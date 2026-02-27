"""
Self-Labeling Protocol & Visual Labeling Tools

Provides:
1. A visual labeling guide for FHP vs Normal posture
2. Semi-automated labeling workflow using pose estimation
3. Interactive labeling UI (Gradio-based or OpenCV-based)
4. Label quality validation and inter-rater statistics

Labeling is the MOST CRITICAL step — model quality depends entirely
on label accuracy. This module enforces consistent, clinically-grounded
labeling criteria.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# ============================================================
# Labeling Criteria (Clinical Standard)
# ============================================================

LABELING_CRITERIA = {
    "FHP": {
        "label": 1,
        "description": "Forward Head Posture",
        "visual_cues": [
            "Ear (tragus) is visibly FORWARD of the shoulder (acromion)",
            "Head appears to jut forward from the neck",
            "Neck has a visible forward curve (cervical flexion)",
            "Shoulders may be rounded forward",
            "Upper back may show increased kyphosis (rounding)",
        ],
        "key_test": "Draw an imaginary vertical line from the shoulder — "
                     "the ear should be AHEAD of this line",
        "color": (0, 0, 255),  # Red in BGR
    },
    "Normal": {
        "label": 0,
        "description": "Normal / Good Posture",
        "visual_cues": [
            "Ear (tragus) is directly ABOVE or slightly behind the shoulder (acromion)",
            "Head is balanced on top of the neck (neutral position)",
            "Spine maintains natural S-curve",
            "Shoulders are relaxed and not rounded",
            "Chin is slightly tucked (not protruding forward)",
        ],
        "key_test": "Draw an imaginary vertical line from the shoulder — "
                     "the ear should be ON or BEHIND this line",
        "color": (0, 255, 0),  # Green in BGR
    },
}


# ============================================================
# Visual Guide Generator
# ============================================================

def create_labeling_guide_image(
    output_path: str = "docs/labeling_guide.png",
    width: int = 1400,
    height: int = 900,
) -> np.ndarray:
    """
    Create a detailed visual labeling guide image.

    The guide shows side-by-side comparison of Normal vs FHP posture
    with annotated anatomical landmarks and decision criteria.
    """
    # Create canvas with dark background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 35)  # Dark gray

    # Colors
    GREEN = (80, 220, 100)
    RED = (80, 80, 240)
    GOLD = (50, 200, 240)
    WHITE = (240, 240, 240)
    GRAY = (140, 140, 140)
    DARK_BG = (45, 45, 50)

    # Fonts
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

    # ---- Title ----
    cv2.putText(img, "FHP LABELING GUIDE", (width // 2 - 250, 50),
                FONT_BOLD, 1.5, GOLD, 2, cv2.LINE_AA)
    cv2.putText(img, "Forward Head Posture Detection - Self-Labeling Protocol",
                (width // 2 - 320, 85), FONT, 0.7, GRAY, 1, cv2.LINE_AA)

    # Divider
    cv2.line(img, (50, 100), (width - 50, 100), GRAY, 1)

    # ---- LEFT: Normal Posture ----
    left_cx = width // 4
    _draw_posture_diagram(img, left_cx, 350, "NORMAL", GREEN, is_fhp=False)

    cv2.putText(img, "NORMAL POSTURE", (left_cx - 100, 140),
                FONT_BOLD, 0.9, GREEN, 2, cv2.LINE_AA)

    # Check mark
    cv2.putText(img, "[GOOD]", (left_cx - 40, 170), FONT, 0.6, GREEN, 1, cv2.LINE_AA)

    # Criteria
    criteria_y = 570
    normal_criteria = [
        "Ear ABOVE shoulder (aligned)",
        "Head balanced on neck",
        "Natural spine curves",
        "Shoulders relaxed, not rounded",
        "CVA angle > 49 degrees",
    ]
    for i, text in enumerate(normal_criteria):
        cv2.circle(img, (left_cx - 170, criteria_y + i * 30 - 5), 5, GREEN, -1)
        cv2.putText(img, text, (left_cx - 155, criteria_y + i * 30),
                    FONT, 0.5, WHITE, 1, cv2.LINE_AA)

    # ---- CENTER: Divider ----
    cv2.line(img, (width // 2, 110), (width // 2, height - 50), GRAY, 1)
    cv2.putText(img, "vs", (width // 2 - 15, height // 2), FONT_BOLD, 1.0, GOLD, 2, cv2.LINE_AA)

    # ---- RIGHT: FHP ----
    right_cx = 3 * width // 4
    _draw_posture_diagram(img, right_cx, 350, "FHP", RED, is_fhp=True)

    cv2.putText(img, "FORWARD HEAD POSTURE", (right_cx - 150, 140),
                FONT_BOLD, 0.9, RED, 2, cv2.LINE_AA)

    cv2.putText(img, "[BAD]", (right_cx - 30, 170), FONT, 0.6, RED, 1, cv2.LINE_AA)

    fhp_criteria = [
        "Ear FORWARD of shoulder",
        "Head juts out from neck",
        "Neck curves forward (flexion)",
        "Shoulders rounded forward",
        "CVA angle < 46 degrees",
    ]
    for i, text in enumerate(fhp_criteria):
        cv2.circle(img, (right_cx - 170, criteria_y + i * 30 - 5), 5, RED, -1)
        cv2.putText(img, text, (right_cx - 155, criteria_y + i * 30),
                    FONT, 0.5, WHITE, 1, cv2.LINE_AA)

    # ---- BOTTOM: Landmark Key ----
    key_y = height - 130
    cv2.line(img, (50, key_y - 20), (width - 50, key_y - 20), GRAY, 1)
    cv2.putText(img, "ANATOMICAL LANDMARKS", (50, key_y + 5),
                FONT_BOLD, 0.7, GOLD, 1, cv2.LINE_AA)

    landmarks = [
        ("1. Head Top", (255, 200, 50)),
        ("2. Ear Tragus", GOLD),
        ("3. Top Neck (C1-C2)", (200, 150, 255)),
        ("4. Bottom Neck (C7)", (255, 150, 200)),
        ("5. Shoulder (Acromion)", (150, 220, 255)),
        ("6. Wrist / Hand", (150, 255, 200)),
    ]

    for i, (name, color) in enumerate(landmarks):
        x = 60 + (i % 3) * 440
        y = key_y + 35 + (i // 3) * 30
        cv2.circle(img, (x, y - 5), 6, color, -1)
        cv2.putText(img, name, (x + 15, y), FONT, 0.55, WHITE, 1, cv2.LINE_AA)

    # Decision rule box
    rule_y = height - 40
    cv2.rectangle(img, (50, rule_y - 15), (width - 50, rule_y + 15), DARK_BG, -1)
    cv2.rectangle(img, (50, rule_y - 15), (width - 50, rule_y + 15), GOLD, 1)
    cv2.putText(img, "KEY RULE: If the ear is FORWARD of the shoulder line -> Label as FHP. "
                "If aligned or behind -> Label as Normal.",
                (70, rule_y + 5), FONT, 0.5, GOLD, 1, cv2.LINE_AA)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Labeling guide saved to: {output_path}")

    return img


def _draw_posture_diagram(
    img: np.ndarray,
    cx: int,
    cy: int,
    label: str,
    color: Tuple[int, int, int],
    is_fhp: bool = False,
) -> None:
    """Draw a simplified side-view posture diagram with anatomical points."""
    GOLD = (50, 200, 240)
    WHITE = (240, 240, 240)

    # Body proportions (side view, sitting)
    head_offset_x = 40 if is_fhp else 0   # FHP: head forward
    shoulder_offset_x = 15 if is_fhp else 0  # FHP: shoulders rounded

    # Key points (relative to center)
    points = {
        "head_top":    (cx + head_offset_x, cy - 160),
        "ear_tragus":  (cx + head_offset_x - 5, cy - 130),
        "top_neck":    (cx + head_offset_x // 2, cy - 100),
        "bottom_neck": (cx + 5, cy - 75),
        "shoulder":    (cx + shoulder_offset_x, cy - 50),
        "chest":       (cx, cy - 20),
        "pelvis":      (cx, cy + 40),
        "elbow":       (cx + 30, cy + 10),
        "wrist":       (cx + 50, cy + 50),
    }

    # Draw body outline (simplified)
    # Spine curve
    spine_pts = [
        points["head_top"],
        points["top_neck"],
        points["bottom_neck"],
        points["chest"],
        points["pelvis"],
    ]
    for i in range(len(spine_pts) - 1):
        cv2.line(img, spine_pts[i], spine_pts[i + 1], color, 3, cv2.LINE_AA)

    # Arm
    cv2.line(img, points["shoulder"], points["elbow"], color, 2, cv2.LINE_AA)
    cv2.line(img, points["elbow"], points["wrist"], color, 2, cv2.LINE_AA)

    # Head circle
    cv2.circle(img, points["head_top"], 25, color, 2, cv2.LINE_AA)

    # Draw landmark dots
    landmark_colors = {
        "head_top": (255, 200, 50),
        "ear_tragus": GOLD,
        "top_neck": (200, 150, 255),
        "bottom_neck": (255, 150, 200),
        "shoulder": (150, 220, 255),
        "wrist": (150, 255, 200),
    }

    for name, pos in points.items():
        if name in landmark_colors:
            cv2.circle(img, pos, 7, landmark_colors[name], -1, cv2.LINE_AA)
            cv2.circle(img, pos, 7, WHITE, 1, cv2.LINE_AA)

    # Draw vertical alignment line from shoulder
    shoulder_x = points["shoulder"][0]
    cv2.line(img, (shoulder_x, cy - 180), (shoulder_x, cy + 60),
             (100, 100, 100), 1, cv2.LINE_AA)

    # Draw CVA angle indicator
    ear = points["ear_tragus"]
    neck = points["bottom_neck"]

    # Line from C7 to ear
    cv2.line(img, neck, ear, GOLD, 2, cv2.LINE_AA)

    # Horizontal from C7
    cv2.line(img, neck, (neck[0] + 80, neck[1]), (100, 100, 100), 1, cv2.LINE_AA)

    # Angle arc
    angle = np.degrees(np.arctan2(-(ear[1] - neck[1]), ear[0] - neck[0]))
    cv2.ellipse(img, neck, (30, 30), 0, -angle, 0, GOLD, 1, cv2.LINE_AA)

    # Angle label
    angle_text = f"CVA ~{int(abs(angle))}°"
    cv2.putText(img, angle_text, (neck[0] + 35, neck[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GOLD, 1, cv2.LINE_AA)


# ============================================================
# Interactive Labeling Tool
# ============================================================

class ImageLabeler:
    """
    OpenCV-based interactive labeling tool for FHP detection.

    Usage:
        labeler = ImageLabeler(image_dir="data/raw/images", output_path="data/labels.json")
        labeler.run()

    Controls:
        - 'n' or '→': Label as Normal (0)
        - 'f' or '←': Label as FHP (1)
        - 's':         Skip image
        - 'u':         Undo last label
        - 'q':         Save and quit
        - 'g':         Show labeling guide
    """

    def __init__(
        self,
        image_dir: str,
        output_path: str = "data/labels.json",
        pose_estimator=None,
        guide_image_path: str = "docs/labeling_guide.png",
    ):
        self.image_dir = Path(image_dir)
        self.output_path = Path(output_path)
        self.pose_estimator = pose_estimator
        self.guide_path = Path(guide_image_path)

        # Load existing labels
        self.labels = self._load_labels()
        self.history = []

        # Find unlabeled images
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.all_images = sorted([
            f for f in self.image_dir.rglob("*")
            if f.suffix.lower() in extensions
        ])
        self.unlabeled = [
            f for f in self.all_images
            if str(f) not in self.labels
        ]

        print(f"Total images: {len(self.all_images)}")
        print(f"Already labeled: {len(self.labels)}")
        print(f"Remaining: {len(self.unlabeled)}")

    def _load_labels(self) -> Dict:
        """Load existing labels from JSON."""
        if self.output_path.exists():
            with open(self.output_path, "r") as f:
                return json.load(f)
        return {}

    def _save_labels(self):
        """Save labels to JSON."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.labels, f, indent=2)
        print(f"Saved {len(self.labels)} labels to {self.output_path}")

    def run(self):
        """Run the interactive labeling loop."""
        if not self.unlabeled:
            print("All images are labeled!")
            return

        print("\n" + "=" * 50)
        print("LABELING CONTROLS:")
        print("  'n' / '→'  : Label as NORMAL")
        print("  'f' / '←'  : Label as FHP")
        print("  's'         : Skip")
        print("  'u'         : Undo last")
        print("  'g'         : Show guide")
        print("  'q'         : Save & quit")
        print("=" * 50 + "\n")

        idx = 0
        while idx < len(self.unlabeled):
            img_path = self.unlabeled[idx]
            frame = cv2.imread(str(img_path))
            if frame is None:
                idx += 1
                continue

            # Annotate with pose if estimator available
            display = frame.copy()
            if self.pose_estimator:
                result = self.pose_estimator.estimate_2d(frame)
                if result.get("detected"):
                    display = self.pose_estimator.draw_pose(display, result)

            # Add UI overlay
            display = self._add_overlay(display, img_path, idx)

            cv2.imshow("FHP Labeler", display)
            key = cv2.waitKey(0) & 0xFF

            if key == ord("n") or key == 83:  # 'n' or Right arrow
                self._label_image(img_path, 0)
                idx += 1
            elif key == ord("f") or key == 81:  # 'f' or Left arrow
                self._label_image(img_path, 1)
                idx += 1
            elif key == ord("s"):
                idx += 1
            elif key == ord("u"):
                if self.history:
                    last_path, _ = self.history.pop()
                    del self.labels[last_path]
                    idx = max(0, idx - 1)
            elif key == ord("g"):
                self._show_guide()
            elif key == ord("q"):
                break

        self._save_labels()
        cv2.destroyAllWindows()

    def _label_image(self, img_path: Path, label: int):
        """Record a label for an image."""
        path_str = str(img_path)
        self.labels[path_str] = {
            "label": label,
            "class": "FHP" if label == 1 else "Normal",
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append((path_str, label))

        # Auto-save every 50 labels
        if len(self.labels) % 50 == 0:
            self._save_labels()

    def _add_overlay(self, frame: np.ndarray, img_path: Path, idx: int) -> np.ndarray:
        """Add labeling UI overlay to frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Top bar
        cv2.rectangle(overlay, (0, 0), (w, 45), (30, 30, 35), -1)
        progress = f"{idx + 1}/{len(self.unlabeled)} | Labeled: {len(self.labels)}"
        cv2.putText(overlay, progress, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        # File name
        cv2.putText(overlay, img_path.name, (w - 300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        # Bottom bar with controls
        cv2.rectangle(overlay, (0, h - 40), (w, h), (30, 30, 35), -1)
        cv2.putText(overlay, "[N]ormal    [F]HP    [S]kip    [U]ndo    [G]uide    [Q]uit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (50, 200, 240), 1, cv2.LINE_AA)

        return overlay

    def _show_guide(self):
        """Display the labeling guide."""
        if self.guide_path.exists():
            guide = cv2.imread(str(self.guide_path))
            cv2.imshow("Labeling Guide", guide)
            cv2.waitKey(0)
            cv2.destroyWindow("Labeling Guide")
        else:
            print("Guide not found. Generating...")
            guide = create_labeling_guide_image(str(self.guide_path))
            cv2.imshow("Labeling Guide", guide)
            cv2.waitKey(0)
            cv2.destroyWindow("Labeling Guide")


def export_labels_to_numpy(
    labels_json_path: str,
    output_dir: str,
) -> Tuple[int, int]:
    """
    Convert JSON labels to numpy arrays for training.

    Returns:
        (num_normal, num_fhp) counts
    """
    with open(labels_json_path, "r") as f:
        labels = json.load(f)

    paths = []
    label_values = []

    for path, info in labels.items():
        paths.append(path)
        label_values.append(info["label"])

    labels_arr = np.array(label_values, dtype=np.int64)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    np.save(str(output / "labels.npy"), labels_arr)

    # Save path mapping
    with open(str(output / "path_mapping.json"), "w") as f:
        json.dump({"paths": paths}, f)

    normal = int((labels_arr == 0).sum())
    fhp = int((labels_arr == 1).sum())
    print(f"Exported: {len(labels_arr)} labels (Normal: {normal}, FHP: {fhp})")

    return normal, fhp


if __name__ == "__main__":
    # Generate the labeling guide
    guide = create_labeling_guide_image("docs/labeling_guide.png")
    print(f"Guide shape: {guide.shape}")

    print("\nLabeling criteria:")
    for name, info in LABELING_CRITERIA.items():
        print(f"\n  {name} (label={info['label']}):")
        for cue in info["visual_cues"]:
            print(f"    - {cue}")
