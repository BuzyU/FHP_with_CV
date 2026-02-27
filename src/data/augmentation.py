"""
3D Skeleton Data Augmentation

Augmentation techniques for 3D pose data that preserve anatomical validity:
- Random rotation around each axis
- Gaussian noise on joint positions
- Left-right skeleton mirroring
- Temporal jittering (frame offsets/drops)
- Scale perturbation
"""

import numpy as np
from typing import Optional, Tuple


def augment_rotation(
    joints: np.ndarray,
    max_angle_deg: float = 15.0,
    axes: str = "xyz",
) -> np.ndarray:
    """
    Apply random rotation around specified axes.

    Args:
        joints: (..., num_joints, 3) joint positions
        max_angle_deg: Maximum rotation angle per axis
        axes: Which axes to rotate around ("x", "y", "z", or combination)

    Returns:
        Rotated joints of same shape
    """
    result = joints.copy()

    for axis in axes:
        angle = np.random.uniform(-max_angle_deg, max_angle_deg)
        angle_rad = np.radians(angle)
        c, s = np.cos(angle_rad), np.sin(angle_rad)

        if axis == "x":
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == "y":
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis == "z":
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            continue

        # Flatten all but last dim, apply rotation, reshape back
        original_shape = result.shape
        flat = result.reshape(-1, 3)
        rotated = (R @ flat.T).T
        result = rotated.reshape(original_shape)

    return result.astype(np.float32)


def augment_noise(
    joints: np.ndarray,
    noise_std: float = 0.02,
) -> np.ndarray:
    """
    Add Gaussian noise to joint positions.

    Args:
        joints: (..., num_joints, 3)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Noisy joints of same shape
    """
    noise = np.random.randn(*joints.shape).astype(np.float32) * noise_std
    return joints + noise


def augment_mirror(
    joints: np.ndarray,
    left_indices: Optional[list] = None,
    right_indices: Optional[list] = None,
) -> np.ndarray:
    """
    Mirror skeleton left-right (flips y-axis and swaps left/right joints).

    Default indices are for the 13-joint upper body format.

    Args:
        joints: (..., num_joints, 3)
        left_indices: Joint indices on the left side
        right_indices: Corresponding right side indices

    Returns:
        Mirrored joints of same shape
    """
    if left_indices is None:
        # Upper body: L_Hip=2, L_Shoulder=7, L_Elbow=8, L_Wrist=9
        left_indices = [2, 7, 8, 9]
    if right_indices is None:
        # Upper body: R_Hip=1, R_Shoulder=10, R_Elbow=11, R_Wrist=12
        right_indices = [1, 10, 11, 12]

    result = joints.copy()

    # Flip y-axis (width dimension)
    result[..., 1] = -result[..., 1]

    # Swap left and right joints
    for l_idx, r_idx in zip(left_indices, right_indices):
        temp = result[..., l_idx, :].copy()
        result[..., l_idx, :] = result[..., r_idx, :]
        result[..., r_idx, :] = temp

    return result


def augment_scale(
    joints: np.ndarray,
    scale_range: Tuple[float, float] = (0.9, 1.1),
) -> np.ndarray:
    """
    Apply random uniform scaling.

    Args:
        joints: (..., num_joints, 3)
        scale_range: (min_scale, max_scale)

    Returns:
        Scaled joints
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return (joints * scale).astype(np.float32)


def augment_temporal_jitter(
    sequence: np.ndarray,
    max_jitter_frames: int = 3,
) -> np.ndarray:
    """
    Apply temporal jittering — randomly offset/shift the sequence in time.

    Args:
        sequence: (T, num_joints, 3) — temporal sequence
        max_jitter_frames: Maximum frame offset

    Returns:
        Jittered sequence of same shape
    """
    T = sequence.shape[0]
    offset = np.random.randint(-max_jitter_frames, max_jitter_frames + 1)

    if offset == 0:
        return sequence.copy()

    result = np.zeros_like(sequence)

    if offset > 0:
        result[offset:] = sequence[:T - offset]
        result[:offset] = sequence[0]  # pad start with first frame
    else:
        offset = abs(offset)
        result[:T - offset] = sequence[offset:]
        result[T - offset:] = sequence[-1]  # pad end with last frame

    return result


def augment_temporal_dropout(
    sequence: np.ndarray,
    dropout_prob: float = 0.1,
) -> np.ndarray:
    """
    Randomly drop frames and fill with interpolation.

    Simulates camera frame drops / detection failures.

    Args:
        sequence: (T, num_joints, 3)
        dropout_prob: Probability of dropping each frame

    Returns:
        Augmented sequence
    """
    T = sequence.shape[0]
    result = sequence.copy()

    drop_mask = np.random.rand(T) < dropout_prob

    # Don't drop first or last frame
    drop_mask[0] = False
    drop_mask[-1] = False

    for t in range(1, T - 1):
        if drop_mask[t]:
            # Linear interpolation from nearest non-dropped frames
            prev_t = t - 1
            while prev_t > 0 and drop_mask[prev_t]:
                prev_t -= 1
            next_t = t + 1
            while next_t < T - 1 and drop_mask[next_t]:
                next_t += 1

            alpha = (t - prev_t) / max(next_t - prev_t, 1)
            result[t] = (1 - alpha) * sequence[prev_t] + alpha * sequence[next_t]

    return result


class PoseAugmentor:
    """
    Composable augmentation pipeline for 3D pose data.

    Example:
        augmentor = PoseAugmentor(
            rotation_range=15.0,
            noise_std=0.02,
            mirror_prob=0.5,
            scale_range=(0.9, 1.1),
            temporal_jitter=3,
        )
        aug_joints, aug_features = augmentor(joints, features)
    """

    def __init__(
        self,
        rotation_range: float = 15.0,
        noise_std: float = 0.02,
        mirror_prob: float = 0.5,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        temporal_jitter: int = 3,
        temporal_dropout_prob: float = 0.1,
        enabled: bool = True,
    ):
        self.rotation_range = rotation_range
        self.noise_std = noise_std
        self.mirror_prob = mirror_prob
        self.scale_range = scale_range
        self.temporal_jitter = temporal_jitter
        self.temporal_dropout_prob = temporal_dropout_prob
        self.enabled = enabled

    def __call__(
        self,
        joints: np.ndarray,
        bio_features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply augmentations.

        Args:
            joints: (T, N, 3) or (N, 3) for single frame
            bio_features: Optional corresponding feature vectors

        Returns:
            (augmented_joints, bio_features)
            Note: bio_features should be recomputed after augmentation
        """
        if not self.enabled:
            return joints, bio_features

        is_sequence = joints.ndim == 3
        result = joints.copy()

        # Spatial augmentations
        if self.rotation_range > 0:
            result = augment_rotation(result, self.rotation_range)

        if self.noise_std > 0:
            result = augment_noise(result, self.noise_std)

        if np.random.rand() < self.mirror_prob:
            result = augment_mirror(result)

        if self.scale_range != (1.0, 1.0):
            result = augment_scale(result, self.scale_range)

        # Temporal augmentations (only for sequences)
        if is_sequence:
            if self.temporal_jitter > 0:
                result = augment_temporal_jitter(result, self.temporal_jitter)

            if self.temporal_dropout_prob > 0:
                result = augment_temporal_dropout(result, self.temporal_dropout_prob)

        # Bio features should be recomputed after augmentation
        return result, None  # Caller should recompute features


if __name__ == "__main__":
    np.random.seed(42)

    # Test single frame
    joints_single = np.random.randn(13, 3).astype(np.float32) * 0.3
    print(f"Original: {joints_single[0]}")

    rotated = augment_rotation(joints_single, max_angle_deg=15.0)
    print(f"Rotated:  {rotated[0]}")

    noisy = augment_noise(joints_single, noise_std=0.02)
    print(f"Noisy:    {noisy[0]}")

    mirrored = augment_mirror(joints_single)
    print(f"Mirrored: {mirrored[0]}")

    # Test sequence
    seq = np.random.randn(30, 13, 3).astype(np.float32) * 0.3

    augmentor = PoseAugmentor()
    aug_seq, _ = augmentor(seq)
    print(f"\nSequence augmented: {seq.shape} → {aug_seq.shape}")
    print("Augmentation pipeline works!")
