"""
2D Pose Estimation Wrapper

Wraps MediaPipe Pose (and optionally RTMPose) for real-time 2D keypoint detection.
Outputs are converted to Human3.6M 17-joint format for compatibility with VideoPose3D.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class PoseEstimator:
    """
    Real-time 2D human pose estimation using MediaPipe.

    Detects 33 body landmarks from RGB images and converts them
    to the Human3.6M 17-joint format needed by VideoPose3D.

    Args:
        backend: "mediapipe" (default) or "rtmpose"
        model_complexity: 0=lite, 1=full, 2=heavy
        min_detection_confidence: Minimum detection confidence
        min_tracking_confidence: Minimum tracking confidence
    """

    def __init__(
        self,
        backend: str = "mediapipe",
        model_complexity: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ):
        self.backend = backend
        self._init_model(model_complexity, min_detection_confidence, min_tracking_confidence)

    def _init_model(self, complexity, det_conf, track_conf):
        """Initialize the pose estimation model."""
        if self.backend == "mediapipe":
            try:
                import mediapipe as mp
                self.mp_pose = mp.solutions.pose
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles

                self.model = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=complexity,
                    enable_segmentation=False,
                    min_detection_confidence=det_conf,
                    min_tracking_confidence=track_conf,
                )
                print(f"✅ MediaPipe Pose initialized (complexity={complexity})")
            except ImportError:
                print("⚠️  MediaPipe not installed. Run: pip install mediapipe")
                self.model = None

        elif self.backend == "rtmpose":
            # RTMPose would require mmpose setup
            print("⚠️  RTMPose backend not yet implemented. Using MediaPipe fallback.")
            self._init_model("mediapipe", complexity, det_conf, track_conf)

    def estimate_2d(
        self,
        frame: np.ndarray,
        return_raw: bool = False,
    ) -> Optional[Dict]:
        """
        Estimate 2D pose from a single BGR frame.

        Args:
            frame: BGR image from OpenCV (H, W, 3)
            return_raw: If True, also return raw MediaPipe landmarks

        Returns:
            Dict with:
              - "keypoints_2d": (17, 2) H36M-format 2D keypoints (pixel coords)
              - "keypoints_2d_norm": (17, 2) normalized [0, 1]
              - "confidence": (17,) per-joint confidence scores
              - "detected": bool
              - "raw_landmarks": (33, 4) raw MediaPipe output (if return_raw)
        """
        if self.model is None:
            return {"detected": False}

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        results = self.model.process(rgb)

        if results.pose_landmarks is None:
            return {"detected": False}

        # Extract MediaPipe landmarks (33 points × [x, y, z, visibility])
        raw = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ], dtype=np.float32)

        # Convert to H36M 17-joint format
        from src.utils.skeleton import mediapipe_to_h36m
        h36m_joints = mediapipe_to_h36m(raw)  # (17, 3)

        # Pixel coordinates (for visualization)
        keypoints_px = h36m_joints[:, :2].copy()
        keypoints_px[:, 0] *= w
        keypoints_px[:, 1] *= h

        # Confidence scores (mapped from MediaPipe visibility)
        confidence = self._map_confidence(raw[:, 3])

        output = {
            "keypoints_2d": keypoints_px,           # (17, 2) pixel coordinates
            "keypoints_2d_norm": h36m_joints[:, :2], # (17, 2) normalized [0,1]
            "keypoints_3d_mp": h36m_joints,          # (17, 3) MediaPipe 3D estimate
            "confidence": confidence,                # (17,) per-joint confidence
            "detected": True,
        }

        if return_raw:
            output["raw_landmarks"] = raw
            output["pose_results"] = results

        return output

    def _map_confidence(self, mp_visibility: np.ndarray) -> np.ndarray:
        """Map MediaPipe 33-joint visibility to H36M 17-joint confidence."""
        from src.utils.skeleton import _MEDIAPIPE_TO_H36M

        confidence = np.zeros(17, dtype=np.float32)
        for mp_idx, h36m_idx in _MEDIAPIPE_TO_H36M.items():
            confidence[h36m_idx] = mp_visibility[mp_idx]

        # Synthesized joints get average confidence
        avg_conf = np.mean([confidence[i] for i in [1, 4, 11, 14] if confidence[i] > 0])
        for idx in [0, 7, 8, 9]:  # pelvis, spine, chest, neck
            if confidence[idx] == 0:
                confidence[idx] = avg_conf

        return confidence

    def draw_pose(
        self,
        frame: np.ndarray,
        pose_result: Dict,
        draw_connections: bool = True,
    ) -> np.ndarray:
        """
        Draw detected pose on the frame.

        Args:
            frame: BGR image
            pose_result: Output from estimate_2d()
            draw_connections: Whether to draw bone connections

        Returns:
            Annotated frame
        """
        if not pose_result.get("detected", False):
            return frame

        annotated = frame.copy()
        keypoints = pose_result["keypoints_2d"]
        confidence = pose_result["confidence"]

        from src.utils.skeleton import UPPER_BODY_EDGES, UPPER_BODY_INDICES, get_bone_color

        # Draw bones
        if draw_connections:
            for src, dst in UPPER_BODY_EDGES:
                h36m_src = UPPER_BODY_INDICES[src]
                h36m_dst = UPPER_BODY_INDICES[dst]

                if confidence[h36m_src] > 0.3 and confidence[h36m_dst] > 0.3:
                    pt1 = tuple(keypoints[h36m_src].astype(int))
                    pt2 = tuple(keypoints[h36m_dst].astype(int))
                    color = get_bone_color(src, dst)
                    cv2.line(annotated, pt1, pt2, color, 2, cv2.LINE_AA)

        # Draw joints
        for i in UPPER_BODY_INDICES:
            if confidence[i] > 0.3:
                pt = tuple(keypoints[i].astype(int))
                cv2.circle(annotated, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, pt, 3, (0, 200, 255), -1, cv2.LINE_AA)

        return annotated

    def release(self):
        """Release model resources."""
        if hasattr(self, 'model') and self.model is not None:
            if self.backend == "mediapipe":
                self.model.close()

    def __del__(self):
        self.release()


class BatchPoseEstimator:
    """
    Batch 2D pose estimation for processing image datasets.

    Processes a directory of images and outputs 2D keypoints files.
    """

    def __init__(self, **kwargs):
        self.estimator = PoseEstimator(**kwargs)

    def process_directory(
        self,
        image_dir: str,
        output_path: str,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> Dict:
        """
        Process all images in a directory.

        Args:
            image_dir: Path to directory of images
            output_path: Path to save keypoints_2d.npy

        Returns:
            Summary dict with counts and quality info
        """
        image_dir = Path(image_dir)
        images = sorted([
            f for f in image_dir.iterdir()
            if f.suffix.lower() in extensions
        ])

        all_keypoints = []
        all_confidences = []
        failed = []

        from tqdm import tqdm

        for img_path in tqdm(images, desc="Detecting 2D poses"):
            frame = cv2.imread(str(img_path))
            if frame is None:
                failed.append(str(img_path))
                continue

            result = self.estimator.estimate_2d(frame)

            if result.get("detected", False):
                all_keypoints.append(result["keypoints_2d_norm"])
                all_confidences.append(result["confidence"])
            else:
                failed.append(str(img_path))

        keypoints_arr = np.array(all_keypoints)
        confidence_arr = np.array(all_confidences)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output), keypoints_arr)
        np.save(str(output).replace(".npy", "_confidence.npy"), confidence_arr)

        summary = {
            "total_images": len(images),
            "successful": len(all_keypoints),
            "failed": len(failed),
            "output_shape": keypoints_arr.shape,
        }

        print(f"✅ Processed {summary['successful']}/{summary['total_images']} images")
        if failed:
            print(f"⚠️  Failed: {len(failed)} images")

        return summary

    def process_video(
        self,
        video_path: str,
        output_path: str,
        sample_fps: Optional[float] = None,
    ) -> Dict:
        """
        Process a video file frame by frame.

        Args:
            video_path: Path to video file
            output_path: Path to save keypoints_2d.npy
            sample_fps: If set, subsample to this FPS (None = use all frames)

        Returns:
            Summary dict
        """
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_interval = 1
        if sample_fps and video_fps > 0:
            frame_interval = max(1, int(video_fps / sample_fps))

        all_keypoints = []
        frame_idx = 0

        from tqdm import tqdm

        with tqdm(total=total_frames // frame_interval, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    result = self.estimator.estimate_2d(frame)
                    if result.get("detected", False):
                        all_keypoints.append(result["keypoints_2d_norm"])
                    pbar.update(1)

                frame_idx += 1

        cap.release()

        keypoints_arr = np.array(all_keypoints)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output), keypoints_arr)

        return {
            "total_frames": total_frames,
            "processed_frames": len(all_keypoints),
            "video_fps": video_fps,
            "output_shape": keypoints_arr.shape,
        }


if __name__ == "__main__":
    # Quick test — open webcam and detect pose
    estimator = PoseEstimator(model_complexity=1)

    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = estimator.estimate_2d(frame)
        annotated = estimator.draw_pose(frame, result)

        cv2.imshow("Pose Estimation", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    estimator.release()
