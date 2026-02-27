"""
End-to-End Automation Pipeline

Automates the full data processing workflow:
  1. Collect data (images/videos from directories)
  2. Run 2D pose estimation on all images
  3. Lift 2D → 3D poses
  4. Preprocess and normalize
  5. Generate labeling guide
  6. Split into train/val/test
  7. Prepare for training

Usage:
    python scripts/run_pipeline.py --config config.yaml --stage all
    python scripts/run_pipeline.py --config config.yaml --stage preprocess
"""

import sys
import os
import argparse
import yaml
import numpy as np
import json
import time
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def stage_collect(config: dict, image_dir: Optional[str] = None):
    """
    Stage 1: Collect and organize raw data.

    Verifies that raw data exists and reports statistics.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: DATA COLLECTION")
    print("=" * 60)

    raw_dir = Path(image_dir or config["paths"]["data_raw"])

    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created data directory: {raw_dir}")
        print(f"\n⚠️  No raw data found. Please add images to:")
        print(f"   {raw_dir.absolute()}")
        print(f"\n   Supported formats: .jpg, .jpeg, .png, .bmp")
        print(f"   Tip: Organize into subfolders (e.g., raw/statefarm/, raw/youtube/)")
        return False

    # Count images
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [f for f in raw_dir.rglob("*") if f.suffix.lower() in extensions]
    videos = [f for f in raw_dir.rglob("*") if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]

    print(f"📊 Raw data stats:")
    print(f"   Images found: {len(images)}")
    print(f"   Videos found: {len(videos)}")
    print(f"   Location: {raw_dir.absolute()}")

    if len(images) == 0 and len(videos) == 0:
        print(f"\n⚠️  No data found. Add images to: {raw_dir.absolute()}")
        return False

    return True


def stage_detect_2d(config: dict, image_dir: Optional[str] = None):
    """
    Stage 2: Run 2D pose estimation on all images.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: 2D POSE ESTIMATION")
    print("=" * 60)

    from src.models.pose_estimator import BatchPoseEstimator

    raw_dir = Path(image_dir or config["paths"]["data_raw"])
    output_dir = Path(config["paths"]["data_processed"])
    output_path = output_dir / "keypoints_2d.npy"

    if output_path.exists():
        existing = np.load(str(output_path))
        print(f"⏭️  Found existing 2D keypoints: {existing.shape}")
        resp = input("   Re-process? (y/N): ").strip().lower()
        if resp != "y":
            return True

    estimator = BatchPoseEstimator(
        backend=config["pose_estimation"]["backend"],
        model_complexity=config["pose_estimation"]["model_complexity"],
    )

    summary = estimator.process_directory(
        image_dir=str(raw_dir),
        output_path=str(output_path),
    )

    print(f"\n✅ 2D detection complete:")
    for k, v in summary.items():
        print(f"   {k}: {v}")

    return summary.get("successful", 0) > 0


def stage_lift_3d(config: dict):
    """
    Stage 3: Lift 2D keypoints to 3D.
    """
    print("\n" + "=" * 60)
    print("STAGE 3: 2D → 3D LIFTING")
    print("=" * 60)

    from src.models.videopose3d import VideoPose3DLifter

    processed_dir = Path(config["paths"]["data_processed"])
    kp_2d_path = processed_dir / "keypoints_2d.npy"
    kp_3d_path = processed_dir / "poses_3d.npy"

    if not kp_2d_path.exists():
        print(f"❌ No 2D keypoints found at {kp_2d_path}. Run stage 'detect_2d' first.")
        return False

    keypoints_2d = np.load(str(kp_2d_path))
    print(f"📥 Loaded 2D keypoints: {keypoints_2d.shape}")

    lifter = VideoPose3DLifter(
        model_path=config["lifting"].get("model_path"),
    )

    # Process in batches
    batch_size = 128
    all_3d = []
    N = keypoints_2d.shape[0]

    from tqdm import tqdm

    for i in tqdm(range(0, N, batch_size), desc="Lifting 2D → 3D"):
        batch = keypoints_2d[i:i + batch_size]
        # Add time dim (single frame)
        batch_t = batch[:, np.newaxis, :, :]  # (B, 1, 17, 2)
        result_3d = lifter.lift(batch_t)       # (B, 1, 17, 3)
        all_3d.append(result_3d[:, 0, :, :])   # remove time dim

    poses_3d = np.concatenate(all_3d, axis=0)
    np.save(str(kp_3d_path), poses_3d)

    print(f"\n✅ 3D lifting complete: {poses_3d.shape}")
    print(f"   Saved to: {kp_3d_path}")

    return True


def stage_preprocess(config: dict):
    """
    Stage 4: Preprocess and normalize 3D poses.
    """
    print("\n" + "=" * 60)
    print("STAGE 4: PREPROCESSING & NORMALIZATION")
    print("=" * 60)

    from src.data.preprocessing import preprocess_single_frame
    from src.utils.angles import compute_all_biomechanical_features, features_to_vector, get_feature_names

    processed_dir = Path(config["paths"]["data_processed"])
    poses_3d_path = processed_dir / "poses_3d.npy"

    if not poses_3d_path.exists():
        print(f"❌ No 3D poses found. Run stage 'lift_3d' first.")
        return False

    poses_3d = np.load(str(poses_3d_path))
    N = poses_3d.shape[0]
    print(f"📥 Loaded {N} 3D poses")

    all_normalized = []
    all_features = []
    quality_scores = []

    from tqdm import tqdm
    from src.data.preprocessing import compute_sample_quality

    for i in tqdm(range(N), desc="Preprocessing"):
        norm_j, feat_v = preprocess_single_frame(poses_3d[i], upper_body_only=True)
        quality = compute_sample_quality(norm_j)

        all_normalized.append(norm_j)
        all_features.append(feat_v)
        quality_scores.append(quality)

    normalized = np.array(all_normalized)
    features = np.array(all_features)
    quality = np.array(quality_scores)

    np.save(str(processed_dir / "poses_normalized.npy"), normalized)
    np.save(str(processed_dir / "bio_features.npy"), features)
    np.save(str(processed_dir / "quality_scores.npy"), quality)

    # Quality report
    print(f"\n✅ Preprocessing complete:")
    print(f"   Normalized poses: {normalized.shape}")
    print(f"   Bio features: {features.shape}")
    print(f"   Feature names: {get_feature_names()}")
    print(f"   Quality stats: mean={quality.mean():.2f}, "
          f"min={quality.min():.2f}, max={quality.max():.2f}")

    low_quality = (quality < 0.5).sum()
    if low_quality > 0:
        print(f"   ⚠️  {low_quality} samples have low quality (<0.5)")

    return True


def stage_label(config: dict):
    """
    Stage 5: Generate labeling guide and run labeling tool.
    """
    print("\n" + "=" * 60)
    print("STAGE 5: LABELING")
    print("=" * 60)

    from src.data.label_tools import create_labeling_guide_image, ImageLabeler

    # Generate guide
    guide_path = "docs/labeling_guide.png"
    create_labeling_guide_image(guide_path)

    # Check if labels exist
    labels_path = Path(config["paths"]["data_processed"]) / "labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            existing = json.load(f)
        print(f"📊 Existing labels: {len(existing)}")
        resp = input("   Launch labeling tool for remaining? (y/N): ").strip().lower()
        if resp != "y":
            return True

    raw_dir = config["paths"]["data_raw"]
    labeler = ImageLabeler(
        image_dir=raw_dir,
        output_path=str(labels_path),
        guide_image_path=guide_path,
    )
    labeler.run()

    return True


def stage_split(config: dict):
    """
    Stage 6: Split data into train/val/test sets.
    """
    print("\n" + "=" * 60)
    print("STAGE 6: TRAIN/VAL/TEST SPLIT")
    print("=" * 60)

    from src.data.dataset import create_data_splits

    processed_dir = Path(config["paths"]["data_processed"])
    splits_dir = Path(config["paths"]["data_splits"])

    poses_path = processed_dir / "poses_3d.npy"
    labels_path = processed_dir / "labels.npy"

    if not poses_path.exists():
        print(f"❌ No poses found at {poses_path}")
        return False

    if not labels_path.exists():
        # Check if JSON labels exist and convert
        json_labels = processed_dir / "labels.json"
        if json_labels.exists():
            from src.data.label_tools import export_labels_to_numpy
            export_labels_to_numpy(str(json_labels), str(processed_dir))
        else:
            print(f"❌ No labels found. Run stage 'label' first.")
            return False

    train_ratio = config["training"]["train_split"]
    val_ratio = config["training"]["val_split"]
    seed = config["training"]["seed"]

    create_data_splits(
        poses_path=str(poses_path),
        labels_path=str(labels_path),
        output_dir=str(splits_dir),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    print(f"\n✅ Data splits saved to: {splits_dir}")
    return True


def stage_verify(config: dict):
    """
    Final verification: check all data is ready for training.
    """
    print("\n" + "=" * 60)
    print("VERIFICATION: TRAINING READINESS CHECK")
    print("=" * 60)

    checks = []
    splits_dir = Path(config["paths"]["data_splits"])

    for split in ["train", "val", "test"]:
        split_dir = splits_dir / split
        poses_ok = (split_dir / "poses_3d.npy").exists()
        labels_ok = (split_dir / "labels.npy").exists()

        if poses_ok and labels_ok:
            poses = np.load(str(split_dir / "poses_3d.npy"))
            labels = np.load(str(split_dir / "labels.npy"))
            check = f"✅ {split}: {len(labels)} samples (Normal: {(labels == 0).sum()}, FHP: {(labels == 1).sum()})"
            checks.append(True)
        else:
            check = f"❌ {split}: Missing data"
            checks.append(False)

        print(f"   {check}")

    guide_ok = Path("docs/labeling_guide.png").exists()
    print(f"   {'✅' if guide_ok else '❌'} Labeling guide: {'found' if guide_ok else 'missing'}")

    config_ok = Path("config.yaml").exists()
    print(f"   {'✅' if config_ok else '❌'} Config: {'found' if config_ok else 'missing'}")

    all_ok = all(checks) and guide_ok and config_ok
    print(f"\n{'🎉 READY FOR TRAINING!' if all_ok else '⚠️  Some checks failed. Fix the issues above.'}")

    return all_ok


# ============================================================
# Main CLI
# ============================================================

STAGES = {
    "collect": stage_collect,
    "detect_2d": stage_detect_2d,
    "lift_3d": stage_lift_3d,
    "preprocess": stage_preprocess,
    "label": stage_label,
    "split": stage_split,
    "verify": stage_verify,
}


def main():
    parser = argparse.ArgumentParser(
        description="FHP Detection — End-to-End Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  collect    - Check and organize raw data
  detect_2d  - Run 2D pose estimation on images
  lift_3d    - Lift 2D keypoints to 3D
  preprocess - Normalize and extract features
  label      - Generate guide & run labeling tool
  split      - Create train/val/test splits
  verify     - Check training readiness
  all        - Run all stages sequentially
        """,
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--stage", default="all", choices=list(STAGES.keys()) + ["all"],
                        help="Which pipeline stage to run")
    parser.add_argument("--image-dir", default=None, help="Override raw image directory")
    args = parser.parse_args()

    config = load_config(args.config)

    print("🦴 FHP Detection Pipeline")
    print(f"   Config: {args.config}")
    print(f"   Stage: {args.stage}")

    start_time = time.time()

    if args.stage == "all":
        for stage_name, stage_fn in STAGES.items():
            if stage_name == "collect":
                success = stage_fn(config, args.image_dir)
            elif stage_name == "detect_2d":
                success = stage_fn(config, args.image_dir)
            else:
                success = stage_fn(config)

            if not success:
                print(f"\n❌ Pipeline stopped at stage '{stage_name}'")
                break
    else:
        stage_fn = STAGES[args.stage]
        if args.stage in ("collect", "detect_2d"):
            stage_fn(config, args.image_dir)
        else:
            stage_fn(config)

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
