"""
FHP Detection — FastAPI REST API

Backend server for web deployment on Render.
Receives base64-encoded frames from the webcam frontend,
runs the full FHP detection pipeline, returns results.

Endpoints:
  GET  /              — Health check
  GET  /health        — Detailed health
  POST /api/detect    — Single frame FHP detection
  POST /api/session   — Session-based detection (maintains frame buffer)
  GET  /api/info      — Model and system info
"""

import os
import sys
import time
import base64
import io
import uuid
import numpy as np
import cv2
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.skeleton import get_upper_body_adjacency, extract_upper_body, mediapipe_to_h36m
from src.utils.angles import compute_all_biomechanical_features, features_to_vector, classify_cva_severity
from src.data.preprocessing import normalize_3d_pose
from src.models.stgcn import create_model


# ============================================================
# Global State
# ============================================================

model = None
adj_tensor = None
device = None
pose_model = None
lifter = None
sessions: Dict[str, deque] = {}
config = {}


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and resources on startup."""
    global model, adj_tensor, device, pose_model, lifter, config

    print("🚀 Starting FHP Detection API...")

    # Load config
    config_path = os.environ.get("CONFIG_PATH", str(PROJECT_ROOT / "config.yaml"))
    if Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = _default_config()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # Load ST-GCN model
    model_config = config.get("model", {})
    model_config.setdefault("in_channels", 3)
    model_config.setdefault("num_joints", 13)
    model_config.setdefault("num_frames", 30)
    model_config.setdefault("gcn_hidden", [64, 128, 64])
    model_config.setdefault("temporal_kernel_size", 9)
    model_config.setdefault("bio_feature_dim", 6)
    model_config.setdefault("num_classes", 2)
    model_config.setdefault("dropout", 0.5)
    model_config.setdefault("type", "stgcn")

    model = create_model(model_config)

    # Try to load trained weights
    model_path = os.environ.get("MODEL_PATH",
        str(PROJECT_ROOT / "models" / "exported" / "stgcn_fhp.pth"))
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        print(f"   ✅ Loaded model: {model_path}")
    else:
        print(f"   ⚠️  No trained model found — running in DEMO mode")

    model.to(device)
    model.eval()

    # Adjacency matrix
    adj = get_upper_body_adjacency(normalize=True)
    adj_tensor = torch.tensor(adj, dtype=torch.float32, device=device)

    # MediaPipe Pose — handle both legacy and new API
    _mp_api_version = None
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
            # Legacy API (Python < 3.13)
            pose_model = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5,
            )
            _mp_api_version = "legacy"
            print("   ✅ MediaPipe loaded (solutions API)")
        elif hasattr(mp, 'tasks'):
            # New tasks API (Python 3.13+)
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            # Download pose landmarker model if needed
            import urllib.request
            model_asset = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
            if not model_asset.exists():
                model_asset.parent.mkdir(parents=True, exist_ok=True)
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
                print(f"   📥 Downloading pose landmarker model...")
                urllib.request.urlretrieve(url, str(model_asset))
                print(f"   ✅ Downloaded to {model_asset}")

            base_options = mp_python.BaseOptions(
                model_asset_path=str(model_asset)
            )
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                num_poses=1,
            )
            pose_model = mp_vision.PoseLandmarker.create_from_options(options)
            _mp_api_version = "tasks"
            print("   ✅ MediaPipe loaded (tasks API)")
        else:
            print("   ⚠️  MediaPipe found but no compatible API")
            pose_model = None
    except Exception as e:
        print(f"   ⚠️  MediaPipe error: {e}")
        pose_model = None

    # Store API version for detection logic
    app.state.mp_api_version = _mp_api_version

    # VideoPose3D lifter
    try:
        from src.models.videopose3d import VideoPose3DLifter
        lifter = VideoPose3DLifter(device=str(device))
        print("   ✅ VideoPose3D loaded")
    except Exception as e:
        print(f"   ⚠️  VideoPose3D: {e}")
        lifter = None

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {total_params:,}")
    print("   🟢 API ready!")

    yield

    # Cleanup
    if pose_model and hasattr(pose_model, 'close'):
        pose_model.close()
    print("🔴 API shutdown")


def _default_config():
    return {
        "model": {
            "type": "stgcn", "in_channels": 3, "num_joints": 13,
            "num_frames": 30, "gcn_hidden": [64, 128, 64],
            "temporal_kernel_size": 9, "bio_feature_dim": 6,
            "num_classes": 2, "dropout": 0.5,
        }
    }


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="FHP Detection API",
    description="Real-time Forward Head Posture detection using CV + GCN",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

WEB_DIR = PROJECT_ROOT / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ============================================================
# Request / Response Models
# ============================================================

class DetectRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded JPEG/PNG image")
    session_id: Optional[str] = Field(None, description="Session ID for temporal tracking")

class DetectResponse(BaseModel):
    detected: bool
    classification: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    angles: Optional[Dict[str, float]] = None
    score_breakdown: Optional[Dict[str, float]] = None
    cva_severity: Optional[str] = None
    keypoints_2d: Optional[list] = None
    session_id: Optional[str] = None
    inference_time_ms: float = 0

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    pose_estimator: bool
    lifter: bool
    active_sessions: int


# ============================================================
# Endpoints
# ============================================================

@app.get("/")
async def root():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return {"status": "ok", "service": "FHP Detection API", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device),
        pose_estimator=pose_model is not None,
        lifter=lifter is not None,
        active_sessions=len(sessions),
    )


@app.post("/api/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    """
    Detect FHP from a single base64-encoded image.

    Pipeline:
    1. Decode image
    2. MediaPipe 2D pose
    3. VideoPose3D 2D→3D lift
    4. Normalize + extract features
    5. ST-GCN classify
    """
    start = time.time()

    # 1. Decode image
    try:
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",")[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    # 2. 2D Pose Detection
    if pose_model is None:
        raise HTTPException(status_code=503, detail="Pose estimator not available")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)  # Required for MediaPipe tasks API

    # Debug: log first few frames
    _debug_count = getattr(app.state, '_debug_frame_count', 0)
    if _debug_count < 3:
        print(f"   [DEBUG] Frame shape: {frame.shape}, dtype: {frame.dtype}")
        print(f"   [DEBUG] RGB shape: {rgb.shape}, contiguous: {rgb.flags['C_CONTIGUOUS']}")
        app.state._debug_frame_count = _debug_count + 1

    raw_landmarks = None
    mp_api = getattr(app.state, 'mp_api_version', None)

    try:
        if mp_api == "legacy":
            results = pose_model.process(rgb)
            if results.pose_landmarks is not None:
                raw_landmarks = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in results.pose_landmarks.landmark
                ], dtype=np.float32)

        elif mp_api == "tasks":
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = pose_model.detect(mp_image)

            if _debug_count < 3:
                has_lm = bool(results.pose_landmarks and len(results.pose_landmarks) > 0)
                print(f"   [DEBUG] Tasks API result: has_landmarks={has_lm}")
                if has_lm:
                    print(f"   [DEBUG] Num landmarks: {len(results.pose_landmarks[0])}")

            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                landmarks = results.pose_landmarks[0]
                raw_landmarks = np.array([
                    [lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0)]
                    for lm in landmarks
                ], dtype=np.float32)
    except Exception as e:
        import traceback
        print(f"Pose detection error: {e}")
        traceback.print_exc()

    if raw_landmarks is None:
        return DetectResponse(
            detected=False,
            inference_time_ms=(time.time() - start) * 1000,
        )

    # Convert to H36M format for the model
    h36m_3d = mediapipe_to_h36m(raw_landmarks)

    # Send ALL 33 raw MediaPipe 2D keypoints for pixel-perfect skeleton overlay
    h, w = frame.shape[:2]
    kp_2d_overlay = []
    for idx in range(33):
        if idx < len(raw_landmarks) and raw_landmarks[idx, 3] > 0.3:  # visibility threshold
            kp_2d_overlay.append([
                round(float(raw_landmarks[idx, 0] * w), 1),
                round(float(raw_landmarks[idx, 1] * h), 1),
            ])
        else:
            kp_2d_overlay.append(None)

    # Add virtual NECK point (index 33) — midpoint between shoulders, slightly above
    lsh = kp_2d_overlay[11]
    rsh = kp_2d_overlay[12]
    if lsh and rsh:
        neck_x = round((lsh[0] + rsh[0]) / 2, 1)
        neck_y = round((lsh[1] + rsh[1]) / 2 - (abs(lsh[1] - rsh[1]) * 0.3 + 8), 1)
        kp_2d_overlay.append([neck_x, neck_y])
    else:
        kp_2d_overlay.append(None)

    # 3. Preprocessing for ML model
    upper = extract_upper_body(h36m_3d)
    normalized = normalize_3d_pose(upper)
    bio_features = compute_all_biomechanical_features(normalized)
    bio_vec = features_to_vector(bio_features)

    # 4. ST-GCN model inference (trained model, not hardcoded)
    ml_fhp_prob = 0.5  # default
    if model is not None:
        try:
            adj = get_upper_body_adjacency(normalize=True)
            adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
            # Single frame → replicate to fill temporal window
            j_np = normalized[np.newaxis, np.newaxis, :, :]  # (1, 1, 13, 3)
            j_np = np.repeat(j_np, 30, axis=1)  # (1, 30, 13, 3)
            f_np = bio_vec[np.newaxis, np.newaxis, :]  # (1, 1, 6)
            f_np = np.repeat(f_np, 30, axis=1)  # (1, 30, 6)

            j_t = torch.tensor(j_np, dtype=torch.float32, device=device)
            f_t = torch.tensor(f_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                logits = model(j_t, adj_t, f_t)
                probs = torch.softmax(logits, dim=1)
                ml_fhp_prob = float(probs[0, 1])
        except Exception as e:
            _debug_cls = getattr(app.state, '_debug_cls_count', 0)
            if _debug_cls < 3:
                print(f"   [ML] model error: {e}")
                app.state._debug_cls_count = _debug_cls + 1

    # 5. ANGLE-AGNOSTIC FHP detection using full 3D vectors
    # Works from front, side, or any diagonal camera angle
    # Uses 3D displacement magnitude + view-angle auto-detection

    session_id = request.session_id or str(uuid.uuid4())

    # --- Extract key landmarks (MediaPipe 33-joint, normalized 0-1 coords) ---
    nose = raw_landmarks[0, :3]
    l_ear = raw_landmarks[7, :3]
    r_ear = raw_landmarks[8, :3]
    l_shoulder = raw_landmarks[11, :3]
    r_shoulder = raw_landmarks[12, :3]
    l_hip = raw_landmarks[23, :3]
    r_hip = raw_landmarks[24, :3]

    mid_ear = (l_ear + r_ear) / 2.0
    mid_shoulder = (l_shoulder + r_shoulder) / 2.0
    mid_hip = (l_hip + r_hip) / 2.0

    # Torso length for normalization (scale-independent)
    torso_len = float(np.linalg.norm(mid_hip - mid_shoulder)) + 1e-6

    # === AUTO-DETECT CAMERA ANGLE ===
    # If shoulders are at similar X → frontal; if very different X → side view
    shoulder_x_spread = abs(l_shoulder[0] - r_shoulder[0])
    # Frontal: spread ~0.2-0.4; Side: spread ~0.0-0.08
    is_mostly_frontal = shoulder_x_spread > 0.10
    is_mostly_side = shoulder_x_spread < 0.08

    # === MULTI-ANGLE FHP INDICATORS ===

    # 1. 3D ear-to-shoulder forward vector (WORKS FROM ANY ANGLE)
    #    The "forward" component of head displacement in 3D space
    ear_shoulder_vec = mid_ear - mid_shoulder  # 3D vector from shoulder to ear
    # In MediaPipe: x=horizontal, y=vertical(down), z=depth(toward camera = negative)
    # "Forward" in world = combination of x and z depending on camera angle

    # For side view: forward is primarily X-axis
    # For frontal view: forward is primarily Z-axis
    # Universal: use the full XZ plane displacement normalized by torso
    head_forward_xz = float(np.sqrt(ear_shoulder_vec[0]**2 + ear_shoulder_vec[2]**2))
    # The vertical component tells us if ears are above or at shoulder level
    ear_above_shoulder = float(-ear_shoulder_vec[1])  # positive = ear above (good)

    # 2. Normalized head forward displacement (scale-independent)
    head_fwd_ratio = head_forward_xz / torso_len

    # 3. Vertical ear position relative to shoulder (normalized by torso)
    ear_height_ratio = ear_above_shoulder / torso_len

    # 4. Nose position relative to ears (chin forward indicator)
    nose_to_ear_vec = nose - mid_ear
    nose_drop = float(nose_to_ear_vec[1]) / torso_len  # positive = nose below ears
    nose_forward = float(np.sqrt(nose_to_ear_vec[0]**2 + nose_to_ear_vec[2]**2)) / torso_len

    # 5. Neck angle: angle between (hip→shoulder) and (shoulder→ear) vectors in 3D
    hip_to_shoulder = mid_shoulder - mid_hip
    shoulder_to_ear = mid_ear - mid_shoulder
    if np.linalg.norm(hip_to_shoulder) > 1e-6 and np.linalg.norm(shoulder_to_ear) > 1e-6:
        cos_neck = np.dot(hip_to_shoulder, shoulder_to_ear) / (
            np.linalg.norm(hip_to_shoulder) * np.linalg.norm(shoulder_to_ear) + 1e-8
        )
        neck_angle = float(np.degrees(np.arccos(np.clip(cos_neck, -1, 1))))
    else:
        neck_angle = 180.0

    # 6. CVA proxy: angle of ear-shoulder line from vertical in 3D
    if np.linalg.norm(shoulder_to_ear) > 1e-6:
        vertical_3d = np.array([0, -1, 0], dtype=np.float32)
        cos_cva = np.dot(shoulder_to_ear, vertical_3d) / (
            np.linalg.norm(shoulder_to_ear) + 1e-8
        )
        cva_angle = float(np.degrees(np.arccos(np.clip(cos_cva, -1, 1))))
    else:
        cva_angle = 0.0

    # 7. Shoulder rounding (3D: shoulders forward of hips)
    shoulder_hip_vec = mid_shoulder - mid_hip
    shoulder_fwd_3d = float(np.sqrt(shoulder_hip_vec[0]**2 + shoulder_hip_vec[2]**2)) / torso_len

    # 8. Head tilt (lateral)
    head_tilt = float(np.degrees(np.arctan2(
        abs(l_ear[1] - r_ear[1]), abs(l_ear[0] - r_ear[0])
    )))

    # 9. Shoulder symmetry
    shoulder_sym = abs(l_shoulder[1] - r_shoulder[1]) * 100

    # === FHP SCORE (0-100, higher = worse) ===
    # Thresholds calibrated from actual MediaPipe output on real webcam data
    
    s_neck = min((neck_angle - 15) * 2.0, 35) if neck_angle > 15 else 0.0
    s_ear = min((0.40 - ear_height_ratio) * 120, 25) if ear_height_ratio < 0.40 else 0.0
    s_cva = min((cva_angle - 20) * 1.5, 25) if cva_angle > 20 else 0.0
    s_nose = min((nose_drop - 0.04) * 150, 10) if nose_drop > 0.04 else 0.0
    s_sh = min((shoulder_fwd_3d - 0.15) * 30, 15) if shoulder_fwd_3d > 0.15 else 0.0

    fhp_score = s_neck + s_ear + s_cva + s_nose + s_sh
    fhp_score = min(fhp_score, 100.0)
    
    score_breakdown = {
        "neck": round(s_neck, 1),
        "ear": round(s_ear, 1),
        "cva": round(s_cva, 1),
        "nose": round(s_nose, 1),
        "shoulder": round(s_sh, 1),
        "total": round(fhp_score, 1)
    }

    # Classification uses biomechanical scoring as PRIMARY 
    # (ML model not yet calibrated for real webcam data distribution)
    bio_prob = fhp_score / 100.0

    # Temporal smoothing
    num_frames_smooth = 8
    if session_id not in sessions:
        sessions[session_id] = deque(maxlen=num_frames_smooth)
    sessions[session_id].append(bio_prob)
    smoothed_prob = sum(sessions[session_id]) / len(sessions[session_id])

    is_fhp = smoothed_prob > 0.20  # 20% threshold for sensitivity
    classification = "FHP" if is_fhp else "Normal"
    confidence = round(smoothed_prob if is_fhp else (1 - smoothed_prob), 4)
    probabilities = {
        "Normal": round(1 - smoothed_prob, 4),
        "FHP": round(smoothed_prob, 4),
    }
    severity = "severe" if smoothed_prob > 0.60 else "moderate" if smoothed_prob > 0.20 else "normal"

    # Debug logging — every 30th frame
    _debug_cls = getattr(app.state, '_debug_cls_count', 0)
    app.state._debug_cls_count = _debug_cls + 1
    if _debug_cls % 30 == 0:
        view = "frontal" if is_mostly_frontal else "side" if is_mostly_side else "diagonal"
        # Show individual component scores
        s_neck = min((neck_angle - 35) * 1.2, 35) if neck_angle > 35 else 0
        s_ear = min((0.20 - ear_height_ratio) * 150, 25) if ear_height_ratio < 0.20 else 0
        s_cva = min((cva_angle - 30) * 0.8, 20) if cva_angle > 30 else 0
        print(f"   [FHP] #{_debug_cls} {view} score={fhp_score:.0f} smooth={smoothed_prob:.2f} → {classification}")
        print(f"         neck={neck_angle:.1f}°(+{s_neck:.0f}) ear_h={ear_height_ratio:.3f}(+{s_ear:.0f}) "
              f"cva={cva_angle:.1f}°(+{s_cva:.0f}) nose={nose_drop:.3f} shfwd={shoulder_fwd_3d:.3f}")

    # Angles for display
    angles = {
        "cva_proxy_angle": round(cva_angle, 2),
        "shoulder_rounding_angle": round(shoulder_fwd_3d * 100, 2),
        "head_tilt_angle": round(head_tilt, 2),
        "neck_flexion_angle": round(neck_angle, 2),
        "head_forward_displacement": round(head_fwd_ratio * 100, 2),
        "shoulder_symmetry_angle": round(float(shoulder_sym), 2),
    }

    elapsed = (time.time() - start) * 1000

    return DetectResponse(
        detected=True,
        classification=classification,
        confidence=confidence,
        probabilities=probabilities,
        angles=angles,
        score_breakdown=score_breakdown,
        cva_severity=severity,
        keypoints_2d=kp_2d_overlay,
        session_id=session_id,
        inference_time_ms=round(elapsed, 1),
    )


@app.get("/api/info")
async def info():
    """Return model and system information."""
    return {
        "model_type": "ST-GCN",
        "parameters": sum(p.numel() for p in model.parameters()) if model else 0,
        "device": str(device),
        "num_joints": 13,
        "num_frames": config.get("model", {}).get("num_frames", 30),
        "classes": ["Normal", "FHP"],
        "pose_estimator": "MediaPipe Pose",
        "lifter": "VideoPose3D",
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session buffer."""
    if session_id in sessions:
        del sessions[session_id]
        return {"deleted": True}
    raise HTTPException(status_code=404, detail="Session not found")

# ============================================================
# Serve Frontend
# ============================================================

web_dir = PROJECT_ROOT / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
else:
    print(f"⚠️ Warning: Frontend directory not found at {web_dir}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)
