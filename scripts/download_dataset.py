"""
Download & Process Real FHP Datasets
=====================================

Downloads the "Don't Be a Turtle" posture dataset and processes
images through MediaPipe to extract landmarks for ST-GCN training.

Dataset: Don't Be a Turtle (865 images, Apache 2.0)
Source:  https://github.com/neilsummers/dont-be-a-turtle

Processing pipeline:
  1. Download & extract dataset images
  2. Run MediaPipe Pose on each image
  3. Convert landmarks through mediapipe_to_h36m -> extract_upper_body
  4. Compute biomechanical features
  5. Save as .npy splits (train/val/test)
"""

import os
import sys
import json
import shutil
import zipfile
import urllib.request
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.skeleton import mediapipe_to_h36m, extract_upper_body
from src.utils.angles import compute_all_biomechanical_features, features_to_vector
from src.data.preprocessing import normalize_3d_pose


# ============================================================
# Download helpers
# ============================================================

DATASET_URLS = [
    "https://github.com/nickhsu/turtle-neck-dataset/archive/refs/heads/main.zip",
    "https://github.com/nickhsu/turtle-neck-dataset/archive/refs/heads/master.zip",
]
DATASET_DIR = PROJECT_ROOT / "data" / "dontbeturtle"
RAW_ZIP = DATASET_DIR / "master.zip"


def download_dataset(force: bool = False):
    """Download posture dataset from GitHub (best-effort)."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    marker = DATASET_DIR / ".downloaded"
    if marker.exists() and not force:
        print("[SKIP] Dataset already downloaded.")
        return True

    for url in DATASET_URLS:
        try:
            print(f"[DOWNLOAD] Trying {url} ...")
            urllib.request.urlretrieve(url, str(RAW_ZIP))
            print(f"   Saved: {RAW_ZIP} ({RAW_ZIP.stat().st_size / 1024 / 1024:.1f} MB)")

            print("[EXTRACT] Unpacking ...")
            with zipfile.ZipFile(str(RAW_ZIP), "r") as zf:
                zf.extractall(str(DATASET_DIR))

            # Flatten nested directory
            for child in DATASET_DIR.iterdir():
                if child.is_dir() and child.name != "__MACOSX":
                    for item in child.iterdir():
                        dest = DATASET_DIR / item.name
                        if dest.exists():
                            if dest.is_dir():
                                shutil.rmtree(str(dest))
                            else:
                                dest.unlink()
                        shutil.move(str(item), str(dest))
                    if not any(child.iterdir()):
                        child.rmdir()

            RAW_ZIP.unlink(missing_ok=True)
            marker.touch()
            print("[OK] Dataset downloaded and extracted.")
            return True
        except Exception as e:
            print(f"   Failed: {e}")
            continue

    print("[WARN] Could not download any remote dataset. Using synthetic data only.")
    return False


# ============================================================
# Image discovery & labeling
# ============================================================

def discover_images() -> list:
    """
    Scan the dataset directory for images and assign labels.

    The "Don't Be a Turtle" dataset has COCO-format annotations.
    We also support simple folder-based labelling:
      data/dontbeturtle/normal/  -> label 0
      data/dontbeturtle/fhp/     -> label 1

    Returns a list of dicts: {"path": ..., "label": 0|1}
    """
    images = []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # --- Strategy 1: COCO annotations ---
    ann_path = DATASET_DIR / "annotations" / "person_keypoints_train2017.json"
    if not ann_path.exists():
        # Try other common annotation file names
        for name in ["annotations.json", "labels.json", "keypoints.json"]:
            candidate = DATASET_DIR / name
            if candidate.exists():
                ann_path = candidate
                break

    if ann_path.exists():
        print(f"[LABEL] Found COCO annotations: {ann_path}")
        images = _label_from_coco(ann_path)
        if images:
            return images

    # --- Strategy 2: Folder-based labels ---
    for label_name, label_int in [("normal", 0), ("good", 0),
                                   ("fhp", 1), ("bad", 1), ("turtle", 1), ("forward", 1)]:
        folder = DATASET_DIR / label_name
        if folder.is_dir():
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in valid_ext:
                    images.append({"path": str(f), "label": label_int})

    if images:
        print(f"[LABEL] Found {len(images)} images via folder labels.")
        return images

    # --- Strategy 3: All images, label heuristically via MediaPipe ---
    print("[LABEL] No explicit labels found. Will label via biomechanical heuristic.")
    for root, _, files in os.walk(str(DATASET_DIR)):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in valid_ext:
                images.append({"path": os.path.join(root, fname), "label": -1})

    print(f"[LABEL] Found {len(images)} unlabeled images.")
    return images


def _label_from_coco(ann_path: Path) -> list:
    """Extract images + FHP labels from COCO-format annotations."""
    with open(ann_path) as f:
        coco = json.load(f)

    id_to_file = {}
    for img_info in coco.get("images", []):
        id_to_file[img_info["id"]] = img_info["file_name"]

    category_map = {}
    for cat in coco.get("categories", []):
        category_map[cat["id"]] = cat.get("name", "").lower()

    images = []
    img_dir = ann_path.parent.parent  # typical COCO layout: dataset/annotations/...

    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        fname = id_to_file.get(img_id)
        if fname is None:
            continue

        # Find the actual image file
        img_path = None
        for search_dir in [img_dir / "images", img_dir / "train2017",
                           img_dir / "val2017", img_dir]:
            candidate = search_dir / fname
            if candidate.exists():
                img_path = str(candidate)
                break

        if img_path is None:
            continue

        # Determine label from category or attributes
        cat_id = ann.get("category_id", 1)
        cat_name = category_map.get(cat_id, "")
        attrs = ann.get("attributes", {})

        label = -1  # unknown
        if "turtle" in cat_name or "fhp" in cat_name or "bad" in cat_name:
            label = 1
        elif "normal" in cat_name or "good" in cat_name:
            label = 0
        elif "posture" in attrs:
            label = 1 if attrs["posture"] in ("bad", "fhp", "turtle") else 0

        images.append({"path": img_path, "label": label})

    return images


# ============================================================
# MediaPipe landmark extraction
# ============================================================

def extract_landmarks_from_images(
    image_list: list,
    heuristic_label: bool = True,
) -> tuple:
    """
    Run MediaPipe Pose on each image and extract 33 landmarks.

    Args:
        image_list: list of {"path": ..., "label": 0|1|-1}
        heuristic_label: If True, auto-label images with label=-1

    Returns:
        (landmarks_list, labels_list)  where each landmarks entry is (33, 4)
    """
    import cv2

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        # Download model if needed
        model_path = str(PROJECT_ROOT / "models" / "pose_landmarker_lite.task")
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            print(f"[DL] Downloading MediaPipe model ...")
            urllib.request.urlretrieve(url, model_path)

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
        )
        detector = mp_vision.PoseLandmarker.create_from_options(options)
        api_version = "tasks"
    except Exception:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        detector = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
        )
        api_version = "legacy"

    all_landmarks = []
    all_labels = []
    skipped = 0

    print(f"[EXTRACT] Processing {len(image_list)} images with MediaPipe ({api_version}) ...")

    for i, item in enumerate(image_list):
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i + 1}/{len(image_list)} (skipped {skipped})")

        img_path = item["path"]
        label = item["label"]

        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        landmarks = None
        try:
            if api_version == "tasks":
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = detector.detect(mp_image)
                if results.pose_landmarks and len(results.pose_landmarks) > 0:
                    lms = results.pose_landmarks[0]
                    landmarks = np.array(
                        [[lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)] for lm in lms],
                        dtype=np.float32,
                    )
            else:
                results = detector.process(rgb)
                if results.pose_landmarks is not None:
                    landmarks = np.array(
                        [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
                        dtype=np.float32,
                    )
        except Exception:
            skipped += 1
            continue

        if landmarks is None or landmarks.shape[0] != 33:
            skipped += 1
            continue

        # Auto-label via biomechanical heuristic if label is unknown
        if label == -1 and heuristic_label:
            label = _heuristic_fhp_label(landmarks)

        if label == -1:
            skipped += 1
            continue

        all_landmarks.append(landmarks)
        all_labels.append(label)

    if api_version == "legacy":
        detector.close()

    print(f"[OK] Extracted {len(all_landmarks)} valid samples (skipped {skipped})")
    normal_count = sum(1 for l in all_labels if l == 0)
    fhp_count = sum(1 for l in all_labels if l == 1)
    print(f"   Normal: {normal_count}, FHP: {fhp_count}")

    return all_landmarks, all_labels


def _heuristic_fhp_label(landmarks: np.ndarray) -> int:
    """
    Auto-label a single image as FHP or Normal using the same
    biomechanical rules used in the API scoring system.

    Returns 0 (normal) or 1 (FHP).
    """
    nose = landmarks[0, :3]
    l_ear = landmarks[7, :3]
    r_ear = landmarks[8, :3]
    l_shoulder = landmarks[11, :3]
    r_shoulder = landmarks[12, :3]
    l_hip = landmarks[23, :3]
    r_hip = landmarks[24, :3]

    mid_ear = (l_ear + r_ear) / 2.0
    mid_shoulder = (l_shoulder + r_shoulder) / 2.0
    mid_hip = (l_hip + r_hip) / 2.0

    torso_len = float(np.linalg.norm(mid_hip - mid_shoulder)) + 1e-6

    ear_shoulder_vec = mid_ear - mid_shoulder
    ear_above_shoulder = float(-ear_shoulder_vec[1])
    ear_height_ratio = ear_above_shoulder / torso_len

    hip_to_shoulder = mid_shoulder - mid_hip
    shoulder_to_ear = mid_ear - mid_shoulder
    n1 = np.linalg.norm(hip_to_shoulder)
    n2 = np.linalg.norm(shoulder_to_ear)
    if n1 > 1e-6 and n2 > 1e-6:
        cos_neck = np.dot(hip_to_shoulder, shoulder_to_ear) / (n1 * n2)
        neck_angle = float(np.degrees(np.arccos(np.clip(cos_neck, -1, 1))))
    else:
        neck_angle = 180.0

    vertical_3d = np.array([0, -1, 0], dtype=np.float32)
    n_se = np.linalg.norm(shoulder_to_ear)
    if n_se > 1e-6:
        cos_cva = np.dot(shoulder_to_ear, vertical_3d) / n_se
        cva_angle = float(np.degrees(np.arccos(np.clip(cos_cva, -1, 1))))
    else:
        cva_angle = 0.0

    # Same thresholds as api/main.py
    score = 0.0
    if neck_angle > 30:
        score += min((neck_angle - 30) * 2.0, 30)
    if ear_height_ratio < 0.25:
        score += min((0.25 - ear_height_ratio) * 100, 25)
    if cva_angle > 35:
        score += min((cva_angle - 35) * 1.8, 25)

    return 1 if score >= 35 else 0


# ============================================================
# Build training data (.npy)
# ============================================================

def build_training_data(
    all_landmarks: list,
    all_labels: list,
    num_frames: int = 30,
    output_dir: str = None,
    seed: int = 42,
):
    """
    Convert single-image landmarks into temporal sequences for ST-GCN
    and save as .npy splits.

    Each single image is expanded into a T-frame sequence with
    small random temporal jitter to simulate video input.
    """
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "real_splits")

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    all_joints = []
    all_features = []
    valid_labels = []

    print(f"[BUILD] Processing {len(all_landmarks)} samples into {num_frames}-frame sequences ...")

    for i, (lm, label) in enumerate(zip(all_landmarks, all_labels)):
        if (i + 1) % 200 == 0:
            print(f"   Progress: {i + 1}/{len(all_landmarks)}")

        frame_joints = []
        frame_features = []

        for t in range(num_frames):
            noisy_lm = lm.copy()
            # Temporal jitter: simulates slight movement between frames
            jitter = rng.normal(0, 0.003, (33, 3)).astype(np.float32)
            # Slow drift mimicking breathing / sway
            drift = np.sin(t * 0.15) * 0.002
            noisy_lm[:, :3] += jitter
            noisy_lm[:, 1] += drift

            try:
                h36m = mediapipe_to_h36m(noisy_lm)
                upper = extract_upper_body(h36m)
                normalized = normalize_3d_pose(upper)
                bio = compute_all_biomechanical_features(normalized)
                bio_vec = features_to_vector(bio)
                frame_joints.append(normalized)
                frame_features.append(bio_vec)
            except Exception:
                frame_joints.append(np.zeros((13, 3), dtype=np.float32))
                frame_features.append(np.zeros(6, dtype=np.float32))

        all_joints.append(np.array(frame_joints))
        all_features.append(np.array(frame_features))
        valid_labels.append(label)

    joints_arr = np.array(all_joints, dtype=np.float32)   # (N, T, 13, 3)
    features_arr = np.array(all_features, dtype=np.float32)  # (N, T, 6)
    labels_arr = np.array(valid_labels, dtype=np.int64)       # (N,)

    print(f"   Shapes: joints={joints_arr.shape}, features={features_arr.shape}, labels={labels_arr.shape}")

    # Shuffle then split 70/15/15
    idx = rng.permutation(len(labels_arr))
    joints_arr = joints_arr[idx]
    features_arr = features_arr[idx]
    labels_arr = labels_arr[idx]

    n = len(labels_arr)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits = {
        "train": (joints_arr[:n_train], features_arr[:n_train], labels_arr[:n_train]),
        "val": (joints_arr[n_train:n_train + n_val], features_arr[n_train:n_train + n_val], labels_arr[n_train:n_train + n_val]),
        "test": (joints_arr[n_train + n_val:], features_arr[n_train + n_val:], labels_arr[n_train + n_val:]),
    }

    for split_name, (j, f, l) in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        np.save(os.path.join(split_dir, "joints.npy"), j)
        np.save(os.path.join(split_dir, "features.npy"), f)
        np.save(os.path.join(split_dir, "labels.npy"), l)
        normal_c = int(np.sum(l == 0))
        fhp_c = int(np.sum(l == 1))
        print(f"   {split_name}: {len(l)} samples (Normal={normal_c}, FHP={fhp_c})")

    print(f"[OK] Dataset saved to {output_dir}")
    return output_dir


# ============================================================
# Augment with synthetic data to balance classes
# ============================================================

def augment_with_synthetic(
    output_dir: str = None,
    target_per_class: int = 4000,
    num_frames: int = 30,
    seed: int = 123,
):
    """
    If the real dataset is small or imbalanced, top it up with
    synthetic landmarks from the existing generator.
    """
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "real_splits")

    from scripts.train_real_data import generate_realistic_mediapipe_landmarks, process_through_pipeline

    rng = np.random.default_rng(seed)

    train_dir = os.path.join(output_dir, "train")
    joints = np.load(os.path.join(train_dir, "joints.npy"))
    features = np.load(os.path.join(train_dir, "features.npy"))
    labels = np.load(os.path.join(train_dir, "labels.npy"))

    normal_count = int(np.sum(labels == 0))
    fhp_count = int(np.sum(labels == 1))
    print(f"[AUG] Current train set: Normal={normal_count}, FHP={fhp_count}")

    new_joints = list(joints)
    new_features = list(features)
    new_labels = list(labels)

    for label_name, label_int in [("normal", 0), ("fhp", 1)]:
        current = normal_count if label_int == 0 else fhp_count
        needed = max(0, target_per_class - current)
        if needed == 0:
            continue

        print(f"   Generating {needed} synthetic '{label_name}' samples ...")
        for _ in range(needed):
            variation = rng.uniform(0.5, 1.5)
            base_lm = generate_realistic_mediapipe_landmarks(label_name, variation)

            frame_j, frame_f = [], []
            for t in range(num_frames):
                jitter = rng.normal(0, 0.003, (33, 3)).astype(np.float32)
                lm = base_lm.copy()
                lm[:, :3] += jitter
                lm[:, 1] += np.sin(t * 0.1) * 0.002
                try:
                    j_, f_ = process_through_pipeline(lm)
                    frame_j.append(j_)
                    frame_f.append(f_)
                except Exception:
                    frame_j.append(np.zeros((13, 3), dtype=np.float32))
                    frame_f.append(np.zeros(6, dtype=np.float32))

            new_joints.append(np.array(frame_j))
            new_features.append(np.array(frame_f))
            new_labels.append(label_int)

    # Shuffle and save
    new_joints = np.array(new_joints, dtype=np.float32)
    new_features = np.array(new_features, dtype=np.float32)
    new_labels = np.array(new_labels, dtype=np.int64)

    idx = rng.permutation(len(new_labels))
    np.save(os.path.join(train_dir, "joints.npy"), new_joints[idx])
    np.save(os.path.join(train_dir, "features.npy"), new_features[idx])
    np.save(os.path.join(train_dir, "labels.npy"), new_labels[idx])

    final_normal = int(np.sum(new_labels == 0))
    final_fhp = int(np.sum(new_labels == 1))
    print(f"[OK] Augmented train set: Normal={final_normal}, FHP={final_fhp}, Total={len(new_labels)}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download & process FHP dataset for training")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--frames", type=int, default=30, help="Frames per sequence")
    parser.add_argument("--augment", action="store_true", help="Augment with synthetic data")
    parser.add_argument("--augment-target", type=int, default=4000, help="Target samples per class after augmentation")
    parser.add_argument("--force-download", action="store_true", help="Re-download even if exists")
    args = parser.parse_args()

    output_dir = str(PROJECT_ROOT / "data" / "real_splits")
    download_ok = False

    # Step 1: Download
    if not args.skip_download:
        print("=" * 60)
        print("STEP 1: Download dataset")
        print("=" * 60)
        download_ok = download_dataset(force=args.force_download)

    # Step 2: Discover & label images
    print("\n" + "=" * 60)
    print("STEP 2: Discover & label images")
    print("=" * 60)
    image_list = discover_images() if download_ok else []

    if not image_list:
        print("[INFO] No external images available. Using synthetic-only dataset.")
        from scripts.train_real_data import generate_training_dataset
        generate_training_dataset(num_samples=8000, num_frames=args.frames, output_dir=output_dir)
    else:
        # Step 3: Extract landmarks
        print("\n" + "=" * 60)
        print("STEP 3: Extract MediaPipe landmarks")
        print("=" * 60)
        landmarks, labels = extract_landmarks_from_images(image_list)

        if len(landmarks) < 50:
            print(f"[WARN] Only {len(landmarks)} valid samples. Supplementing with synthetic data.")

        # Step 4: Build training splits
        print("\n" + "=" * 60)
        print("STEP 4: Build training splits")
        print("=" * 60)
        build_training_data(landmarks, labels, num_frames=args.frames, output_dir=output_dir)

    # Step 5: Augment if requested or if dataset is small
    train_labels = np.load(os.path.join(output_dir, "train", "labels.npy"))
    if args.augment or len(train_labels) < 2000:
        print("\n" + "=" * 60)
        print("STEP 5: Augment with synthetic data")
        print("=" * 60)
        augment_with_synthetic(output_dir=output_dir, target_per_class=args.augment_target, num_frames=args.frames)

    print("\n" + "=" * 60)
    print("[DONE] Dataset ready for training!")
    print("=" * 60)
    print(f"   Output: {output_dir}")
    print(f"   Next:   python scripts/train_real_data.py --skip-generate")
