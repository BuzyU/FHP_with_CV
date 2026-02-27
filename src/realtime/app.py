"""
Real-time FHP Detection Application

Captures webcam feed, runs the full pipeline (2D pose → 3D lift → GCN classify),
and displays results with skeleton overlay, angle readouts, and posture alerts.
"""

import cv2
import time
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import deque


class FHPRealtimeApp:
    """
    Real-time Forward Head Posture detection and correction application.

    Pipeline (per frame):
      1. Capture webcam frame
      2. Detect 2D pose (MediaPipe)
      3. Lift to 3D (VideoPose3D)
      4. Preprocess & normalize
      5. Classify with ST-GCN
      6. Display results + alerts

    Args:
        config_path: Path to config.yaml
        model_path: Path to trained ST-GCN checkpoint
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        model_path: Optional[str] = None,
    ):
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # State
        self.running = False
        self.frame_buffer = deque(maxlen=self.config["model"]["num_frames"])
        self.prediction_history = deque(maxlen=60)
        self.last_alert_time = 0
        self.fps_history = deque(maxlen=30)
        self.session_stats = {
            "total_frames": 0,
            "fhp_frames": 0,
            "normal_frames": 0,
            "start_time": None,
        }

        # Initialize components
        self._init_pose_estimator()
        self._init_lifter()
        self._init_classifier(model_path)

        print("✅ FHP Real-time App initialized")
        print(f"   Device: {self.device}")
        print(f"   Camera: {self.config['realtime']['camera_id']}")

    def _init_pose_estimator(self):
        """Initialize 2D pose estimation."""
        from src.models.pose_estimator import PoseEstimator
        pe_config = self.config["pose_estimation"]
        self.pose_estimator = PoseEstimator(
            backend=pe_config["backend"],
            model_complexity=pe_config["model_complexity"],
            min_detection_confidence=pe_config["min_detection_confidence"],
            min_tracking_confidence=pe_config["min_tracking_confidence"],
        )

    def _init_lifter(self):
        """Initialize 3D pose lifter."""
        from src.models.videopose3d import VideoPose3DLifter
        lift_config = self.config["lifting"]
        self.lifter = VideoPose3DLifter(
            model_path=lift_config.get("model_path"),
            device=str(self.device),
        )

    def _init_classifier(self, model_path: Optional[str]):
        """Initialize FHP classifier."""
        from src.models.stgcn import create_model
        from src.utils.skeleton import get_upper_body_adjacency

        model_config = self.config["model"]
        self.classifier = create_model(model_config)

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.classifier.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.classifier.load_state_dict(checkpoint)
            print(f"✅ Loaded classifier from {model_path}")
        else:
            print("⚠️  No trained classifier loaded — using random weights (demo mode)")

        self.classifier.to(self.device)
        self.classifier.eval()

        # Pre-compute adjacency matrix
        adj = get_upper_body_adjacency(normalize=True)
        self.adj_tensor = torch.tensor(adj, dtype=torch.float32, device=self.device)

    def run(self):
        """Main application loop."""
        rt_config = self.config["realtime"]
        cap = cv2.VideoCapture(rt_config["camera_id"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, rt_config["window_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rt_config["window_height"])

        self.running = True
        self.session_stats["start_time"] = time.time()

        print("\n🎥 Starting real-time FHP detection...")
        print("   Press 'q' to quit, 'r' to reset stats, 's' for screenshot")

        while self.running and cap.isOpened():
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            result = self._process_frame(frame)

            # Render display
            display = self._render_display(frame, result)

            cv2.imshow("FHP Detection System", display)

            # FPS tracking
            fps = 1.0 / max(time.time() - frame_start, 1e-8)
            self.fps_history.append(fps)

            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.running = False
            elif key == ord("r"):
                self._reset_stats()
            elif key == ord("s"):
                self._save_screenshot(display)

        cap.release()
        cv2.destroyAllWindows()
        self._print_session_summary()

    def _process_frame(self, frame: np.ndarray) -> Dict:
        """
        Full processing pipeline for a single frame.

        Returns result dict with pose, features, classification.
        """
        result = {"detected": False}

        # Step 1: 2D pose estimation
        pose_2d = self.pose_estimator.estimate_2d(frame)
        if not pose_2d.get("detected"):
            return result

        result["pose_2d"] = pose_2d

        # Step 2: 3D lifting
        kp_2d = pose_2d["keypoints_2d_norm"]  # (17, 2)
        kp_3d = self.lifter.lift(kp_2d)       # (17, 3)
        result["pose_3d"] = kp_3d

        # Step 3: Preprocessing
        from src.data.preprocessing import preprocess_single_frame
        from src.utils.skeleton import extract_upper_body
        from src.utils.angles import compute_all_biomechanical_features, features_to_vector

        upper_body = extract_upper_body(kp_3d)
        from src.data.preprocessing import normalize_3d_pose
        normalized = normalize_3d_pose(upper_body)

        features = compute_all_biomechanical_features(normalized)
        feat_vec = features_to_vector(features)

        result["normalized_joints"] = normalized
        result["bio_features"] = features
        result["bio_vector"] = feat_vec
        result["detected"] = True

        # Step 4: Buffer for temporal model
        self.frame_buffer.append({
            "joints": normalized,
            "features": feat_vec,
        })

        # Step 5: Classification
        if len(self.frame_buffer) >= self.config["model"]["num_frames"]:
            prediction = self._classify()
            result.update(prediction)

            # Update stats
            self.session_stats["total_frames"] += 1
            if prediction.get("class") == "FHP":
                self.session_stats["fhp_frames"] += 1
            else:
                self.session_stats["normal_frames"] += 1

        return result

    def _classify(self) -> Dict:
        """Run classification on buffered frames."""
        num_frames = self.config["model"]["num_frames"]

        # Build input tensors
        joints_list = [f["joints"] for f in self.frame_buffer]
        feats_list = [f["features"] for f in self.frame_buffer]

        # Pad if needed
        while len(joints_list) < num_frames:
            joints_list.insert(0, joints_list[0])
            feats_list.insert(0, feats_list[0])

        joints_tensor = torch.tensor(
            np.array(joints_list[-num_frames:]),
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, T, N, 3)

        feats_tensor = torch.tensor(
            np.array(feats_list[-num_frames:]),
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, T, F)

        # Run model
        with torch.no_grad():
            logits = self.classifier(joints_tensor, self.adj_tensor, feats_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs[0, pred].item()

        class_name = "FHP" if pred == 1 else "Normal"
        self.prediction_history.append((class_name, confidence))

        return {
            "class": class_name,
            "confidence": confidence,
            "probabilities": probs[0].cpu().numpy(),
        }

    def _render_display(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Render the full display with overlays."""
        display = frame.copy()
        h, w = display.shape[:2]
        rt_config = self.config["realtime"]

        # Draw pose skeleton
        if result.get("detected") and rt_config["show_skeleton"]:
            pose_2d = result.get("pose_2d", {})
            display = self.pose_estimator.draw_pose(display, pose_2d)

        # Status panel (top-left)
        self._draw_status_panel(display, result)

        # Angle readouts (top-right)
        if result.get("bio_features") and rt_config["show_angles"]:
            self._draw_angle_panel(display, result["bio_features"])

        # Classification indicator (bottom)
        if "class" in result:
            self._draw_classification(display, result)

        # FPS (bottom-right)
        if rt_config["show_fps"] and self.fps_history:
            avg_fps = np.mean(list(self.fps_history))
            cv2.putText(display, f"FPS: {avg_fps:.0f}", (w - 120, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Alert
        if result.get("class") == "FHP" and result.get("confidence", 0) > rt_config["confidence_threshold"]:
            self._trigger_alert(display, result)

        return display

    def _draw_status_panel(self, img: np.ndarray, result: Dict):
        """Draw status panel on top-left."""
        cv2.rectangle(img, (5, 5), (250, 80), (30, 30, 35), -1)
        cv2.rectangle(img, (5, 5), (250, 80), (80, 80, 80), 1)

        status = "Tracking" if result.get("detected") else "No Person"
        color = (80, 220, 100) if result.get("detected") else (80, 80, 240)
        cv2.circle(img, (20, 25), 6, color, -1)
        cv2.putText(img, status, (35, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Session time
        if self.session_stats["start_time"]:
            elapsed = time.time() - self.session_stats["start_time"]
            mins, secs = divmod(int(elapsed), 60)
            cv2.putText(img, f"Session: {mins:02d}:{secs:02d}", (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # FHP percentage
        total = self.session_stats["total_frames"]
        if total > 0:
            fhp_pct = self.session_stats["fhp_frames"] / total * 100
            cv2.putText(img, f"FHP: {fhp_pct:.0f}%", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (80, 80, 240) if fhp_pct > 30 else (80, 220, 100), 1, cv2.LINE_AA)

    def _draw_angle_panel(self, img: np.ndarray, features: Dict):
        """Draw biomechanical angles on top-right."""
        w = img.shape[1]
        panel_x = w - 260

        cv2.rectangle(img, (panel_x, 5), (w - 5, 130), (30, 30, 35), -1)
        cv2.rectangle(img, (panel_x, 5), (w - 5, 130), (80, 80, 80), 1)

        cv2.putText(img, "Biomechanics", (panel_x + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 240), 1, cv2.LINE_AA)

        y = 45
        for name, value in list(features.items())[:4]:
            short_name = name.replace("_angle", "").replace("_", " ").title()
            cv2.putText(img, f"{short_name}: {value:.1f}", (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            y += 22

    def _draw_classification(self, img: np.ndarray, result: Dict):
        """Draw classification result at the bottom."""
        h, w = img.shape[:2]
        cls = result["class"]
        conf = result.get("confidence", 0)

        if cls == "FHP":
            color = (80, 80, 240)
            bg_color = (50, 40, 60)
        else:
            color = (80, 220, 100)
            bg_color = (40, 55, 45)

        # Bottom banner
        cv2.rectangle(img, (0, h - 50), (w, h), bg_color, -1)

        # Class label
        cv2.putText(img, f"{cls}", (20, h - 18),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)

        # Confidence bar
        bar_x = 180
        bar_w = 200
        cv2.rectangle(img, (bar_x, h - 35), (bar_x + bar_w, h - 15), (60, 60, 60), -1)
        cv2.rectangle(img, (bar_x, h - 35), (bar_x + int(bar_w * conf), h - 15), color, -1)
        cv2.putText(img, f"{conf * 100:.0f}%", (bar_x + bar_w + 10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _trigger_alert(self, img: np.ndarray, result: Dict):
        """Trigger visual (and optionally audio) alert for FHP."""
        now = time.time()
        cooldown = self.config["realtime"]["alert_cooldown_seconds"]

        if now - self.last_alert_time < cooldown:
            return

        self.last_alert_time = now

        # Visual alert — red border flash
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 8)

        cv2.putText(img, "! CORRECT YOUR POSTURE !", (w // 2 - 200, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    def _reset_stats(self):
        """Reset session statistics."""
        self.session_stats = {
            "total_frames": 0,
            "fhp_frames": 0,
            "normal_frames": 0,
            "start_time": time.time(),
        }
        self.prediction_history.clear()
        print("📊 Stats reset")

    def _save_screenshot(self, frame: np.ndarray):
        """Save current frame as screenshot."""
        ts = int(time.time())
        path = f"data/screenshots/screenshot_{ts}.png"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, frame)
        print(f"📸 Screenshot saved: {path}")

    def _print_session_summary(self):
        """Print session summary on exit."""
        stats = self.session_stats
        total = stats["total_frames"]

        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)

        if stats["start_time"]:
            elapsed = time.time() - stats["start_time"]
            print(f"Duration: {elapsed / 60:.1f} minutes")

        print(f"Total classified frames: {total}")

        if total > 0:
            fhp_pct = stats["fhp_frames"] / total * 100
            normal_pct = stats["normal_frames"] / total * 100
            print(f"Normal posture: {normal_pct:.1f}%")
            print(f"FHP detected:   {fhp_pct:.1f}%")

            if fhp_pct > 50:
                print("\n⚠️  Warning: You spent more than half the session in FHP!")
                print("   Consider taking breaks and doing neck stretches.")
            elif fhp_pct > 30:
                print("\n💡 Tip: FHP was detected 30%+ of the time.")
                print("   Try to be more conscious of your head position.")
            else:
                print("\n✅ Great posture session! Keep it up!")

        print("=" * 50)


def main():
    """Entry point for the real-time FHP detection application."""
    import argparse

    parser = argparse.ArgumentParser(description="FHP Detection - Real-time App")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", default=None, help="Path to trained ST-GCN checkpoint")
    parser.add_argument("--camera", type=int, default=None, help="Camera ID override")
    args = parser.parse_args()

    app = FHPRealtimeApp(
        config_path=args.config,
        model_path=args.model,
    )

    if args.camera is not None:
        app.config["realtime"]["camera_id"] = args.camera

    app.run()


if __name__ == "__main__":
    main()
