"""
3D Pose Preprocessing & Normalization

Transforms raw 3D skeleton data into a normalized, view-invariant
representation suitable for the GCN classifier.

Pipeline:
  1. Center on pelvis (translation invariance)
  2. Scale by torso length (body size invariance)
  3. Align spine with z-axis (orientation invariance)
  4. Extract biomechanical features
"""

import numpy as np
from typing import Dict, Tuple, Optional

from src.utils.angles import compute_all_biomechanical_features, features_to_vector
from src.utils.skeleton import extract_upper_body, UPPER_BODY_JOINT_NAMES


def _rotation_matrix_align(source_vec: np.ndarray, target_vec: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix that aligns source_vec to target_vec.
    Uses Rodrigues' rotation formula.
    """
    source = source_vec / (np.linalg.norm(source_vec) + 1e-8)
    target = target_vec / (np.linalg.norm(target_vec) + 1e-8)

    v = np.cross(source, target)
    c = np.dot(source, target)
    s = np.linalg.norm(v)

    if s < 1e-8:
        # Vectors are parallel
        if c > 0:
            return np.eye(3, dtype=np.float32)
        else:
            # 180° rotation — find perpendicular axis
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(source, perp)) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
            axis = np.cross(source, perp)
            axis = axis / np.linalg.norm(axis)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            return (np.eye(3) + 2 * K @ K).astype(np.float32)

    # Skew-symmetric cross-product matrix
    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])

    R = np.eye(3) + K + K @ K * (1 - c) / (s * s + 1e-8)
    return R.astype(np.float32)


def normalize_3d_pose(
    joints_3d: np.ndarray,
    pelvis_idx: int = 0,
    neck_idx: int = 5,
    spine_idx: int = 3,
) -> np.ndarray:
    """
    Normalize a 3D skeleton for view-invariant processing.

    Steps:
      1. Translate to center on pelvis (origin)
      2. Scale by torso length (pelvis → neck)
      3. Rotate to align spine vector with z-axis

    Args:
        joints_3d: (num_joints, 3) array of 3D joint positions

    Returns:
        Normalized joints of same shape
    """
    joints = joints_3d.copy().astype(np.float32)

    # 1. Center on pelvis
    pelvis = joints[pelvis_idx].copy()
    joints -= pelvis

    # 2. Scale normalization
    torso_length = np.linalg.norm(joints[neck_idx] - joints[pelvis_idx])
    if torso_length > 1e-8:
        joints /= torso_length

    # 3. Orientation alignment — align spine with z-axis
    spine_vec = joints[spine_idx] - joints[pelvis_idx]
    if np.linalg.norm(spine_vec) > 1e-8:
        R = _rotation_matrix_align(spine_vec, np.array([0.0, 0.0, 1.0]))
        joints = (R @ joints.T).T

    return joints


def preprocess_single_frame(
    joints_3d_17: np.ndarray,
    upper_body_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a single frame.

    Args:
        joints_3d_17: (17, 3) raw 3D joints from VideoPose3D
        upper_body_only: If True, extract 13 upper body joints

    Returns:
        (normalized_joints, bio_features_vector)
    """
    if upper_body_only:
        joints = extract_upper_body(joints_3d_17)
    else:
        joints = joints_3d_17.copy()

    # Normalize
    normalized = normalize_3d_pose(joints)

    # Extract biomechanical features
    features = compute_all_biomechanical_features(normalized)
    feat_vec = features_to_vector(features)

    return normalized, feat_vec


def preprocess_sequence(
    joints_3d_seq: np.ndarray,
    target_length: int = 30,
    upper_body_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a sequence of frames.

    Args:
        joints_3d_seq: (T, 17, 3) raw 3D joints sequence
        target_length: Desired number of frames (pad or truncate)
        upper_body_only: If True, extract upper body

    Returns:
        (normalized_sequence, bio_features_sequence)
          - normalized: (target_length, num_joints, 3)
          - bio_features: (target_length, num_bio_features)
    """
    T = joints_3d_seq.shape[0]
    all_joints = []
    all_features = []

    for t in range(T):
        norm_j, feat_v = preprocess_single_frame(joints_3d_seq[t], upper_body_only)
        all_joints.append(norm_j)
        all_features.append(feat_v)

    joints_arr = np.array(all_joints)
    feats_arr = np.array(all_features)

    # Pad or truncate to target_length
    joints_arr = _pad_or_truncate(joints_arr, target_length)
    feats_arr = _pad_or_truncate(feats_arr, target_length)

    return joints_arr, feats_arr


def _pad_or_truncate(arr: np.ndarray, target_length: int) -> np.ndarray:
    """Pad (by repeating last frame) or truncate a sequence to target length."""
    T = arr.shape[0]

    if T == target_length:
        return arr
    elif T > target_length:
        # Center crop
        start = (T - target_length) // 2
        return arr[start:start + target_length]
    else:
        # Pad by repeating last frame
        pad_count = target_length - T
        padding = np.tile(arr[-1:], (pad_count,) + (1,) * (arr.ndim - 1))
        return np.concatenate([arr, padding], axis=0)


def compute_sample_quality(joints_3d: np.ndarray) -> float:
    """
    Estimate quality/confidence of a 3D pose sample.

    Low quality indicators:
    - Very small/large torso length
    - Self-intersecting limbs
    - Extreme joint angles

    Returns:
        Quality score 0-1 (1 = high quality)
    """
    score = 1.0

    # Check torso length (should be reasonable)
    if joints_3d.shape[0] >= 6:
        torso = np.linalg.norm(joints_3d[5] - joints_3d[0])
        if torso < 0.01 or torso > 10.0:
            score *= 0.5

    # Check for NaN/Inf
    if np.any(np.isnan(joints_3d)) or np.any(np.isinf(joints_3d)):
        score = 0.0

    # Check for collapsed skeleton (all joints at same point)
    joint_spread = np.std(joints_3d)
    if joint_spread < 0.01:
        score *= 0.3

    return float(np.clip(score, 0.0, 1.0))


if __name__ == "__main__":
    # Self-test with a synthetic 17-joint pose
    np.random.seed(42)

    # Simulate a full H36M skeleton
    joints_17 = np.random.randn(17, 3).astype(np.float32) * 0.3
    joints_17[0] = [0, 0, 0]  # pelvis at origin-ish
    joints_17[7] = [0, 0, 0.5]  # spine above pelvis
    joints_17[9] = [0, 0, 0.8]  # neck above spine

    # Single frame
    norm_j, feat_v = preprocess_single_frame(joints_17, upper_body_only=True)
    print(f"Normalized joints shape: {norm_j.shape}")
    print(f"Bio features shape: {feat_v.shape}")
    print(f"Quality: {compute_sample_quality(norm_j):.2f}")

    # Sequence
    seq = np.random.randn(50, 17, 3).astype(np.float32) * 0.3
    norm_seq, feat_seq = preprocess_sequence(seq, target_length=30)
    print(f"\nSequence joints shape: {norm_seq.shape}")
    print(f"Sequence features shape: {feat_seq.shape}")
