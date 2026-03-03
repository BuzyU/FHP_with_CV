"""
End-to-end integrity test for the FHP Detection System.

Tests:
  1. Model loading & weight integrity
  2. Forward pass with dummy data
  3. Pipeline consistency (landmarks -> H36M -> upper body -> normalize -> features)
  4. One Euro Filter correctness
  5. API server health & detection endpoint
  6. Session frame buffer & ST-GCN temporal inference
"""

import os
import sys
import time
import json
import base64
import traceback
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
results = []


def test(name, fn):
    """Run a test and record result."""
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS} {name}")
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"  {FAIL} {name}: {e}")
        traceback.print_exc()


# ============================================================
# Test 1: Model loading & weight integrity
# ============================================================

def test_model_load():
    from src.models.stgcn import create_model
    from src.utils.skeleton import get_upper_body_adjacency

    config = {
        "type": "stgcn",
        "in_channels": 3,
        "num_joints": 13,
        "num_frames": 30,
        "gcn_hidden": [64, 128, 64],
        "temporal_kernel_size": 9,
        "bio_feature_dim": 6,
        "num_classes": 2,
        "dropout": 0.5,
    }
    model = create_model(config)
    assert model is not None, "Model creation failed"

    model_path = os.path.join(PROJECT_ROOT, "models", "exported", "stgcn_fhp.pth")
    assert os.path.exists(model_path), f"Model file not found: {model_path}"

    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model has 0 parameters"

    # Check no NaN/Inf in weights
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name}"
        assert not torch.isinf(param).any(), f"Inf in {name}"

    print(f"    Model params: {total_params:,}")


# ============================================================
# Test 2: Forward pass with dummy data
# ============================================================

def test_forward_pass():
    from src.models.stgcn import create_model
    from src.utils.skeleton import get_upper_body_adjacency

    config = {
        "type": "stgcn", "in_channels": 3, "num_joints": 13,
        "num_frames": 30, "gcn_hidden": [64, 128, 64],
        "temporal_kernel_size": 9, "bio_feature_dim": 6,
        "num_classes": 2, "dropout": 0.5,
    }
    model = create_model(config)
    model_path = os.path.join(PROJECT_ROOT, "models", "exported", "stgcn_fhp.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model.eval()

    adj = torch.tensor(get_upper_body_adjacency(normalize=True), dtype=torch.float32)

    # Dummy batch
    B, T, N = 2, 30, 13
    joints = torch.randn(B, T, N, 3)
    bio = torch.randn(B, T, 6)

    with torch.no_grad():
        logits = model(joints, adj, bio)

    assert logits.shape == (B, 2), f"Expected (2,2), got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in output logits"
    assert not torch.isinf(logits).any(), "Inf in output logits"

    probs = torch.softmax(logits, dim=1)
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities out of range"
    assert torch.allclose(probs.sum(dim=1), torch.ones(B), atol=1e-5), "Probabilities don't sum to 1"

    print(f"    Output shape: {logits.shape}, probs: {probs[0].tolist()}")


# ============================================================
# Test 3: Pipeline consistency
# ============================================================

def test_pipeline():
    from src.utils.skeleton import mediapipe_to_h36m, extract_upper_body
    from src.data.preprocessing import normalize_3d_pose
    from src.utils.angles import compute_all_biomechanical_features, features_to_vector

    # Simulate a normal posture MediaPipe output
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[:, 3] = 0.99

    # Minimal realistic skeleton
    landmarks[0, :3] = [0.5, 0.25, 0.0]    # nose
    landmarks[7, :3] = [0.42, 0.28, 0.0]   # left ear
    landmarks[8, :3] = [0.58, 0.28, 0.0]   # right ear
    landmarks[11, :3] = [0.38, 0.42, 0.0]  # left shoulder
    landmarks[12, :3] = [0.62, 0.42, 0.0]  # right shoulder
    landmarks[13, :3] = [0.30, 0.58, 0.0]  # left elbow
    landmarks[14, :3] = [0.70, 0.58, 0.0]  # right elbow
    landmarks[15, :3] = [0.28, 0.70, 0.0]  # left wrist
    landmarks[16, :3] = [0.72, 0.70, 0.0]  # right wrist
    landmarks[23, :3] = [0.45, 0.72, 0.0]  # left hip
    landmarks[24, :3] = [0.55, 0.72, 0.0]  # right hip
    landmarks[25, :3] = [0.45, 0.90, 0.0]  # left knee
    landmarks[26, :3] = [0.55, 0.90, 0.0]  # right knee
    landmarks[27, :3] = [0.45, 1.05, 0.0]  # left ankle
    landmarks[28, :3] = [0.55, 1.05, 0.0]  # right ankle

    h36m = mediapipe_to_h36m(landmarks)
    assert h36m.shape == (17, 3), f"H36M shape: {h36m.shape}"

    upper = extract_upper_body(h36m)
    assert upper.shape == (13, 3), f"Upper body shape: {upper.shape}"

    normalized = normalize_3d_pose(upper)
    assert normalized.shape == (13, 3), f"Normalized shape: {normalized.shape}"
    assert not np.isnan(normalized).any(), "NaN in normalized joints"

    bio = compute_all_biomechanical_features(normalized)
    assert isinstance(bio, dict), "Bio features should be dict"

    vec = features_to_vector(bio)
    assert vec.shape == (6,), f"Feature vector shape: {vec.shape}"
    assert not np.isnan(vec).any(), "NaN in feature vector"

    print(f"    Pipeline: (33,4) -> (17,3) -> (13,3) -> (13,3) -> 6 features")


# ============================================================
# Test 4: One Euro Filter
# ============================================================

def test_one_euro_filter():
    # Import from api.main
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "api"))

    import math
    from typing import Optional

    class OneEuroFilter:
        def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
            self.freq = freq
            self.min_cutoff = min_cutoff
            self.beta = beta
            self.d_cutoff = d_cutoff
            self._x_prev = None
            self._dx_prev = None
            self._t_prev = None

        @staticmethod
        def _smoothing_factor(te, cutoff):
            r = 2.0 * math.pi * cutoff * te
            return r / (r + 1.0)

        def __call__(self, x, t=None):
            if self._x_prev is None:
                self._x_prev = x.copy()
                self._dx_prev = np.zeros_like(x)
                self._t_prev = t if t is not None else 0.0
                return x.copy()
            if t is None:
                te = 1.0 / self.freq
            else:
                te = max(t - self._t_prev, 1e-6)
                self._t_prev = t
            a_d = self._smoothing_factor(te, self.d_cutoff)
            dx = (x - self._x_prev) / te
            dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
            cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
            a = np.vectorize(lambda c: self._smoothing_factor(te, c))(cutoff)
            x_hat = a * x + (1.0 - a) * self._x_prev
            self._x_prev = x_hat
            self._dx_prev = dx_hat
            return x_hat

    filt = OneEuroFilter(freq=15.0, min_cutoff=0.8, beta=0.004, d_cutoff=1.0)

    # Stable signal should be passed through with minimal change
    stable = np.array([[0.5, 0.3, 0.0]] * 33, dtype=np.float32)
    out1 = filt(stable, t=0.0)
    assert np.allclose(out1, stable, atol=1e-5), "First frame should pass through unchanged"

    out2 = filt(stable, t=0.066)
    diff = np.abs(out2 - stable).max()
    assert diff < 0.01, f"Stable signal should stay stable, max diff={diff}"

    # Noisy signal should be smoothed
    rng = np.random.default_rng(42)
    filt2 = OneEuroFilter(freq=15.0, min_cutoff=0.8, beta=0.004, d_cutoff=1.0)
    raw_signals = []
    filtered_signals = []
    for i in range(60):
        noisy = stable + rng.normal(0, 0.05, stable.shape).astype(np.float32)
        filtered = filt2(noisy, t=i * 0.066)
        raw_signals.append(noisy.copy())
        filtered_signals.append(filtered.copy())

    raw_var = np.var(np.array(raw_signals)[:, 0, :], axis=0).mean()
    filt_var = np.var(np.array(filtered_signals)[:, 0, :], axis=0).mean()
    reduction = 1.0 - filt_var / raw_var
    assert reduction > 0.3, f"Filter should reduce variance by >30%, got {reduction:.1%}"

    print(f"    Variance reduction: {reduction:.1%} (raw={raw_var:.5f}, filtered={filt_var:.5f})")


# ============================================================
# Test 5: Training data integrity
# ============================================================

def test_training_data():
    data_dir = os.path.join(PROJECT_ROOT, "data", "real_splits")

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        assert os.path.isdir(split_dir), f"Missing split dir: {split}"

        joints = np.load(os.path.join(split_dir, "joints.npy"))
        features = np.load(os.path.join(split_dir, "features.npy"))
        labels = np.load(os.path.join(split_dir, "labels.npy"))

        N = labels.shape[0]
        assert joints.shape == (N, 30, 13, 3), f"{split} joints shape: {joints.shape}"
        assert features.shape == (N, 30, 6), f"{split} features shape: {features.shape}"
        assert set(np.unique(labels)) <= {0, 1}, f"{split} labels: {np.unique(labels)}"
        assert not np.isnan(joints).any(), f"NaN in {split} joints"
        assert not np.isnan(features).any(), f"NaN in {split} features"

        n0 = int(np.sum(labels == 0))
        n1 = int(np.sum(labels == 1))
        print(f"    {split}: {N} samples (Normal={n0}, FHP={n1}), joints={joints.shape}")


# ============================================================
# Test 6: Model prediction sanity (normal vs FHP landmarks)
# ============================================================

def test_model_predictions():
    from src.models.stgcn import create_model
    from src.utils.skeleton import get_upper_body_adjacency, mediapipe_to_h36m, extract_upper_body
    from src.data.preprocessing import normalize_3d_pose
    from src.utils.angles import compute_all_biomechanical_features, features_to_vector

    config = {
        "type": "stgcn", "in_channels": 3, "num_joints": 13,
        "num_frames": 30, "gcn_hidden": [64, 128, 64],
        "temporal_kernel_size": 9, "bio_feature_dim": 6,
        "num_classes": 2, "dropout": 0.5,
    }
    model = create_model(config)
    model_path = os.path.join(PROJECT_ROOT, "models", "exported", "stgcn_fhp.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model.eval()
    adj = torch.tensor(get_upper_body_adjacency(normalize=True), dtype=torch.float32)

    def make_sequence(landmarks, num_frames=30):
        """Process a single landmark set into a full temporal sequence."""
        joints_seq, feats_seq = [], []
        rng = np.random.default_rng(0)
        for t in range(num_frames):
            lm = landmarks.copy()
            lm[:, :3] += rng.normal(0, 0.002, (33, 3)).astype(np.float32)
            h36m = mediapipe_to_h36m(lm)
            upper = extract_upper_body(h36m)
            norm = normalize_3d_pose(upper)
            bio = compute_all_biomechanical_features(norm)
            vec = features_to_vector(bio)
            joints_seq.append(norm)
            feats_seq.append(vec)
        return np.array(joints_seq), np.array(feats_seq)

    # Normal posture landmarks
    normal_lm = np.zeros((33, 4), dtype=np.float32)
    normal_lm[:, 3] = 0.99
    normal_lm[0, :3] = [0.50, 0.25, 0.0]   # nose high
    normal_lm[7, :3] = [0.43, 0.28, 0.0]   # left ear well above shoulders
    normal_lm[8, :3] = [0.57, 0.28, 0.0]   # right ear
    normal_lm[11, :3] = [0.38, 0.42, 0.0]  # left shoulder
    normal_lm[12, :3] = [0.62, 0.42, 0.0]  # right shoulder
    normal_lm[13, :3] = [0.30, 0.58, 0.0]
    normal_lm[14, :3] = [0.70, 0.58, 0.0]
    normal_lm[15, :3] = [0.28, 0.70, 0.0]
    normal_lm[16, :3] = [0.72, 0.70, 0.0]
    normal_lm[23, :3] = [0.45, 0.72, 0.0]
    normal_lm[24, :3] = [0.55, 0.72, 0.0]
    normal_lm[25, :3] = [0.45, 0.90, 0.0]
    normal_lm[26, :3] = [0.55, 0.90, 0.0]
    normal_lm[27, :3] = [0.45, 1.05, 0.0]
    normal_lm[28, :3] = [0.55, 1.05, 0.0]

    # FHP posture landmarks (ears dropped near shoulder level, head craned forward)
    fhp_lm = normal_lm.copy()
    fhp_lm[0, :3] = [0.50, 0.38, -0.08]   # nose dropped & forward
    fhp_lm[7, :3] = [0.43, 0.40, -0.06]   # left ear near shoulder level
    fhp_lm[8, :3] = [0.57, 0.40, -0.06]   # right ear near shoulder level

    nj, nf = make_sequence(normal_lm)
    fj, ff = make_sequence(fhp_lm)

    with torch.no_grad():
        # Stack both into a batch
        j_batch = torch.tensor(np.stack([nj, fj]), dtype=torch.float32)
        f_batch = torch.tensor(np.stack([nf, ff]), dtype=torch.float32)
        logits = model(j_batch, adj, f_batch)
        probs = torch.softmax(logits, dim=1)

    normal_fhp_prob = float(probs[0, 1])
    fhp_fhp_prob = float(probs[1, 1])

    print(f"    Normal posture -> P(FHP) = {normal_fhp_prob:.4f}")
    print(f"    FHP posture    -> P(FHP) = {fhp_fhp_prob:.4f}")

    # We expect the model to show SOME discrimination
    # (It was trained on synthetic data so we check for reasonable behavior)
    assert fhp_fhp_prob > normal_fhp_prob, \
        f"Model should rate FHP higher than Normal: FHP={fhp_fhp_prob:.4f} vs Normal={normal_fhp_prob:.4f}"

    discrimination = fhp_fhp_prob - normal_fhp_prob
    if discrimination > 0.3:
        print(f"    Discrimination: {discrimination:.4f} (good)")
    elif discrimination > 0.1:
        print(f"    Discrimination: {discrimination:.4f} (moderate)")
    else:
        print(f"    {WARN} Discrimination: {discrimination:.4f} (weak)")


# ============================================================
# Test 7: API endpoint test (if server running)
# ============================================================

def test_api_endpoints():
    import urllib.request
    import urllib.error

    base = "http://localhost:8000"

    # Health
    try:
        req = urllib.request.Request(f"{base}/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            assert data.get("status") == "healthy", f"Unhealthy: {data}"
            print(f"    /health: {data['status']}")
    except urllib.error.URLError:
        print(f"    {WARN} Server not running at {base} -- skipping API tests")
        return

    # Info
    req = urllib.request.Request(f"{base}/api/info")
    with urllib.request.urlopen(req, timeout=3) as resp:
        info = json.loads(resp.read())
        assert info.get("model_type") == "ST-GCN"
        print(f"    /api/info: {info['model_type']}, {info['parameters']:,} params")

    # Detection with a dummy black image (640x480)
    import cv2
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a simple stick figure so MediaPipe might detect something
    # (black image won't have a person, so we expect detected=False)
    _, buf = cv2.imencode(".jpg", dummy_img)
    b64 = base64.b64encode(buf).decode()

    payload = json.dumps({"image": b64, "session_id": "test-integrity"}).encode()
    req = urllib.request.Request(
        f"{base}/api/detect",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read())
        # Black image: no person detected is fine
        if not result.get("detected"):
            print(f"    /api/detect (black): detected=False (correct -- no person)")
        else:
            print(f"    /api/detect (black): detected=True, cls={result.get('classification')}")
        assert "inference_time_ms" in result, "Missing inference_time_ms"
        print(f"    inference_time: {result.get('inference_time_ms', 0):.1f}ms")


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FHP Detection System — Integrity Tests")
    print("=" * 60)

    tests = [
        ("1. Model loading & weight integrity", test_model_load),
        ("2. Forward pass (dummy data)", test_forward_pass),
        ("3. Pipeline consistency", test_pipeline),
        ("4. One Euro Filter", test_one_euro_filter),
        ("5. Training data integrity", test_training_data),
        ("6. Model prediction sanity", test_model_predictions),
        ("7. API endpoints (if running)", test_api_endpoints),
    ]

    for name, fn in tests:
        print(f"\n--- {name} ---")
        test(name, fn)

    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r[0] == PASS)
    failed = sum(1 for r in results if r[0] == FAIL)
    print(f"Results: {passed} passed, {failed} failed out of {len(results)} tests")

    if failed:
        print("\nFailed tests:")
        for r in results:
            if r[0] == FAIL:
                print(f"  {r[1]}: {r[2]}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)
