"""
Generate Synthetic Training Data for FHP Detection

Creates biomechanically realistic 3D skeleton poses for both
Normal and FHP postures. This allows training and testing
the full pipeline without real data collection.

The synthetic data models:
- Normal: ear aligned over shoulder, neutral spine
- FHP: ear forward of shoulder, flexed cervical spine
- Variations: body size, sitting angle, arm position

Usage:
    python scripts/generate_training_data.py --samples 5000
    python scripts/generate_training_data.py --samples 10000 --temporal
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_base_skeleton() -> np.ndarray:
    """
    Generate a base H36M 17-joint skeleton in neutral sitting posture.

    Returns:
        (17, 3) array — [x (forward/back), y (left/right), z (up/down)]
    """
    # H36M joint order:
    # 0=Pelvis, 1=R_Hip, 2=R_Knee, 3=R_Ankle, 4=L_Hip, 5=L_Knee, 6=L_Ankle,
    # 7=Spine, 8=Chest, 9=Neck, 10=Head, 11=L_Shoulder, 12=L_Elbow, 13=L_Wrist,
    # 14=R_Shoulder, 15=R_Elbow, 16=R_Wrist

    skeleton = np.array([
        [0.00,  0.00,  0.00],    # 0: Pelvis (origin)
        [0.00, -0.10,  0.02],    # 1: R_Hip
        [0.00, -0.12, -0.40],    # 2: R_Knee
        [0.00, -0.10, -0.80],    # 3: R_Ankle
        [0.00,  0.10,  0.02],    # 4: L_Hip
        [0.00,  0.12, -0.40],    # 5: L_Knee
        [0.00,  0.10, -0.80],    # 6: L_Ankle
        [0.00,  0.00,  0.20],    # 7: Spine
        [0.00,  0.00,  0.40],    # 8: Chest
        [0.00,  0.00,  0.55],    # 9: Neck
        [0.00,  0.00,  0.70],    # 10: Head
        [0.00,  0.18,  0.52],    # 11: L_Shoulder
        [0.00,  0.22,  0.32],    # 12: L_Elbow
        [0.00,  0.20,  0.12],    # 13: L_Wrist
        [0.00, -0.18,  0.52],    # 14: R_Shoulder
        [0.00, -0.22,  0.32],    # 15: R_Elbow
        [0.00, -0.20,  0.12],    # 16: R_Wrist
    ], dtype=np.float32)

    return skeleton


def apply_fhp_deformation(skeleton: np.ndarray, severity: float = 1.0) -> np.ndarray:
    """
    Apply Forward Head Posture deformation to a normal skeleton.

    Args:
        skeleton: (17, 3) base skeleton
        severity: 0.0 (minimal) to 1.0 (severe FHP)

    Returns:
        Deformed skeleton with FHP characteristics
    """
    s = skeleton.copy()
    sev = np.clip(severity, 0.3, 1.5)

    # 1. Head moves forward (x-axis) — the primary FHP indicator
    head_fwd = 0.06 + 0.12 * sev
    s[10, 0] += head_fwd                       # Head
    s[9, 0] += head_fwd * 0.65                  # Neck moves forward too

    # 2. Cervical flexion — neck bends forward
    neck_drop = 0.02 + 0.04 * sev
    s[10, 2] -= neck_drop                       # Head drops slightly
    s[9, 2] -= neck_drop * 0.3

    # 3. Shoulder rounding — shoulders come forward
    shoulder_fwd = 0.02 + 0.05 * sev
    s[11, 0] += shoulder_fwd                    # L_Shoulder
    s[14, 0] += shoulder_fwd                    # R_Shoulder
    s[12, 0] += shoulder_fwd * 0.8              # L_Elbow
    s[15, 0] += shoulder_fwd * 0.8              # R_Elbow

    # 4. Upper back kyphosis — chest rounds
    s[8, 0] += 0.01 + 0.02 * sev               # Chest forward
    s[8, 2] -= 0.01 * sev                       # Chest drops

    return s


def apply_normal_variation(skeleton: np.ndarray, variation: float = 0.5) -> np.ndarray:
    """
    Apply natural variation to a normal posture skeleton.

    Keeps the ear aligned over the shoulder but adds realistic variation.
    """
    s = skeleton.copy()
    var = np.clip(variation, 0.0, 1.0)

    # Slight natural sway (minimal forward/back)
    s[10, 0] += np.random.uniform(-0.02, 0.02) * var  # Head tiny variation
    s[9, 0] += np.random.uniform(-0.01, 0.01) * var

    # Slight shoulder asymmetry
    asym = np.random.uniform(-0.02, 0.02) * var
    s[11, 2] += asym
    s[14, 2] -= asym

    # Arm position variation
    arm_var = np.random.uniform(-0.05, 0.05, size=3) * var
    s[12] += arm_var * 0.5
    s[13] += arm_var
    s[15] += arm_var * 0.5
    s[16] += arm_var

    return s


def add_global_variation(skeleton: np.ndarray) -> np.ndarray:
    """Add body size and orientation variations."""
    s = skeleton.copy()

    # Body size variation (±15%)
    scale = np.random.uniform(0.85, 1.15)
    s *= scale

    # Slight rotation around vertical axis (camera angle variation)
    angle = np.random.uniform(-0.15, 0.15)  # ±~8.5 degrees
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
    s = (rot @ s.T).T

    # Add Gaussian noise
    noise = np.random.randn(*s.shape).astype(np.float32) * 0.008
    s += noise

    return s


def generate_temporal_sequence(
    skeleton: np.ndarray,
    num_frames: int = 30,
    is_fhp: bool = False,
) -> np.ndarray:
    """
    Generate a temporal sequence with subtle frame-to-frame movement.

    Returns:
        (num_frames, 17, 3) sequence
    """
    frames = []
    base = skeleton.copy()

    for t in range(num_frames):
        frame = base.copy()

        # Small breathing movement
        breath = np.sin(2 * np.pi * t / num_frames * 0.8) * 0.005
        frame[8, 2] += breath  # Chest rises
        frame[9, 2] += breath * 0.5

        # Micro-movements (natural fidgeting)
        jitter = np.random.randn(*frame.shape).astype(np.float32) * 0.003
        frame += jitter

        # FHP: occasional head drift forward
        if is_fhp:
            drift = np.sin(2 * np.pi * t / num_frames * 0.3) * 0.01
            frame[10, 0] += drift
            frame[9, 0] += drift * 0.5

        frames.append(frame)

    return np.array(frames, dtype=np.float32)


def generate_dataset(
    num_samples: int = 5000,
    temporal: bool = True,
    num_frames: int = 30,
    fhp_ratio: float = 0.5,
) -> tuple:
    """
    Generate a complete synthetic dataset.

    Args:
        num_samples: Total number of samples
        temporal: If True, generate temporal sequences
        num_frames: Frames per sequence (if temporal)
        fhp_ratio: Fraction of FHP samples

    Returns:
        (poses, labels) — poses shape depends on temporal flag
    """
    num_fhp = int(num_samples * fhp_ratio)
    num_normal = num_samples - num_fhp

    print(f"Generating {num_samples} synthetic samples...")
    print(f"  Normal: {num_normal}")
    print(f"  FHP: {num_fhp}")
    print(f"  Temporal: {temporal} ({num_frames} frames)")

    all_poses = []
    all_labels = []

    base = generate_base_skeleton()

    # Generate Normal samples
    for i in range(num_normal):
        variation = np.random.uniform(0.2, 0.8)
        s = apply_normal_variation(base, variation)
        s = add_global_variation(s)

        if temporal:
            seq = generate_temporal_sequence(s, num_frames, is_fhp=False)
            all_poses.append(seq)
        else:
            all_poses.append(s)
        all_labels.append(0)

    # Generate FHP samples
    for i in range(num_fhp):
        severity = np.random.uniform(0.3, 1.5)
        s = apply_fhp_deformation(base, severity)
        s = add_global_variation(s)

        if temporal:
            seq = generate_temporal_sequence(s, num_frames, is_fhp=True)
            all_poses.append(seq)
        else:
            all_poses.append(s)
        all_labels.append(1)

    poses = np.array(all_poses, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # Shuffle
    indices = np.random.permutation(len(labels))
    poses = poses[indices]
    labels = labels[indices]

    return poses, labels


def split_and_save(
    poses: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
):
    """Split data and save to train/val/test directories."""
    N = len(labels)
    indices = np.random.permutation(N)

    train_end = int(N * train_ratio)
    val_end = int(N * (train_ratio + val_ratio))

    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }

    for split_name, split_idx in splits.items():
        split_dir = Path(output_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        np.save(str(split_dir / "poses_3d.npy"), poses[split_idx])
        np.save(str(split_dir / "labels.npy"), labels[split_idx])

        normal_count = (labels[split_idx] == 0).sum()
        fhp_count = (labels[split_idx] == 1).sum()
        print(f"  {split_name:5s}: {len(split_idx):5d} samples (Normal: {normal_count}, FHP: {fhp_count})")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic FHP training data")
    parser.add_argument("--samples", type=int, default=5000, help="Total samples")
    parser.add_argument("--temporal", action="store_true", default=True, help="Generate temporal sequences")
    parser.add_argument("--frames", type=int, default=30, help="Frames per sequence")
    parser.add_argument("--output", default="data/splits", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    poses, labels = generate_dataset(
        num_samples=args.samples,
        temporal=args.temporal,
        num_frames=args.frames,
    )

    print(f"\nPoses shape: {poses.shape}")
    print(f"Labels shape: {labels.shape}")

    print(f"\nSplitting data:")
    split_and_save(poses, labels, args.output)

    print(f"\n✅ Data saved to {args.output}/")


if __name__ == "__main__":
    main()
