"""
PyTorch Dataset for FHP Detection

Handles loading, caching, and serving of preprocessed 3D pose data
for training/validation/testing of the ST-GCN classifier.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from src.data.preprocessing import preprocess_sequence, preprocess_single_frame
from src.data.augmentation import PoseAugmentor
from src.utils.angles import compute_all_biomechanical_features, features_to_vector
from src.utils.skeleton import get_upper_body_edge_index


class FHPDataset(Dataset):
    """
    Dataset for Forward Head Posture classification.

    Supports both single-frame and sequence (temporal) modes.

    Data format on disk:
      data_dir/
        poses_3d.npy     — (N, [T,] 17, 3) raw 3D joints
        labels.npy       — (N,) binary labels (0=Normal, 1=FHP)
        metadata.json    — dataset info (optional)

    Args:
        data_dir: Path to processed data directory
        split: "train", "val", or "test"
        temporal: If True, use sequences; if False, single frames
        seq_length: Number of frames per sequence (only if temporal=True)
        augment: Whether to apply data augmentation
        augment_config: Augmentation parameters
    """

    CLASS_NAMES = ["Normal", "FHP"]

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        temporal: bool = True,
        seq_length: int = 30,
        augment: bool = False,
        augment_config: Optional[Dict] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.temporal = temporal
        self.seq_length = seq_length

        # Load data
        self._load_data()

        # Setup augmentation
        if augment and augment_config:
            self.augmentor = PoseAugmentor(**augment_config)
        elif augment:
            self.augmentor = PoseAugmentor()
        else:
            self.augmentor = None

        # Edge index for GCN (shared across all samples)
        self.edge_index = torch.tensor(get_upper_body_edge_index(), dtype=torch.long)

    def _load_data(self):
        """Load poses and labels from disk."""
        split_dir = self.data_dir / self.split

        if split_dir.exists():
            poses_path = split_dir / "poses_3d.npy"
            labels_path = split_dir / "labels.npy"
        else:
            # Fallback: load full data and trust external splitting
            poses_path = self.data_dir / "poses_3d.npy"
            labels_path = self.data_dir / "labels.npy"

        if not poses_path.exists():
            # Create dummy data for testing/development
            print(f"⚠️  No data found at {poses_path}. Creating dummy dataset.")
            self._create_dummy_data()
            return

        self.poses_3d = np.load(str(poses_path)).astype(np.float32)
        self.labels = np.load(str(labels_path)).astype(np.int64)

        print(f"Loaded {self.split}: {len(self.labels)} samples "
              f"(Normal: {(self.labels == 0).sum()}, FHP: {(self.labels == 1).sum()})")

    def _create_dummy_data(self):
        """Create dummy data for development/testing."""
        n_samples = 200
        np.random.seed(42)

        if self.temporal:
            self.poses_3d = np.random.randn(n_samples, self.seq_length, 17, 3).astype(np.float32) * 0.3
        else:
            self.poses_3d = np.random.randn(n_samples, 17, 3).astype(np.float32) * 0.3

        self.labels = np.random.randint(0, 2, n_samples).astype(np.int64)
        print(f"Created dummy {self.split}: {n_samples} samples")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with keys:
              - "joints": (seq_length, num_joints, 3) or (num_joints, 3)
              - "bio_features": (seq_length, num_bio_features) or (num_bio_features,)
              - "label": scalar class label
              - "edge_index": (2, num_edges) graph edges
        """
        raw_pose = self.poses_3d[idx].copy()
        label = self.labels[idx]

        if self.temporal:
            # Preprocess temporal sequence
            norm_joints, bio_feats = preprocess_sequence(
                raw_pose, target_length=self.seq_length, upper_body_only=True
            )

            # Apply augmentation
            if self.augmentor is not None:
                norm_joints, _ = self.augmentor(norm_joints)
                # Recompute bio features after augmentation
                bio_feats = np.array([
                    features_to_vector(compute_all_biomechanical_features(norm_joints[t]))
                    for t in range(norm_joints.shape[0])
                ])
        else:
            # Single frame
            norm_joints, bio_feats = preprocess_single_frame(raw_pose, upper_body_only=True)

            if self.augmentor is not None:
                norm_joints, _ = self.augmentor(norm_joints)
                bio_feats = features_to_vector(
                    compute_all_biomechanical_features(norm_joints)
                )

        return {
            "joints": torch.tensor(norm_joints, dtype=torch.float32),
            "bio_features": torch.tensor(bio_feats, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "edge_index": self.edge_index,
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for balanced training."""
        counts = np.bincount(self.labels, minlength=2)
        weights = 1.0 / (counts + 1e-8)
        weights = weights / weights.sum() * 2  # normalize
        return torch.tensor(weights, dtype=torch.float32)

    def get_sampler(self) -> WeightedRandomSampler:
        """Create a weighted random sampler for class balancing."""
        class_weights = self.get_class_weights()
        sample_weights = class_weights[self.labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.labels),
            replacement=True,
        )


def create_data_splits(
    poses_path: str,
    labels_path: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Split data into train/val/test and save to disk.

    Args:
        poses_path: Path to poses_3d.npy
        labels_path: Path to labels.npy
        output_dir: Directory to save splits
    """
    poses = np.load(poses_path)
    labels = np.load(labels_path)
    N = len(labels)

    np.random.seed(seed)
    indices = np.random.permutation(N)

    train_end = int(N * train_ratio)
    val_end = int(N * (train_ratio + val_ratio))

    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }

    output = Path(output_dir)
    for split_name, split_indices in splits.items():
        split_dir = output / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        np.save(str(split_dir / "poses_3d.npy"), poses[split_indices])
        np.save(str(split_dir / "labels.npy"), labels[split_indices])

        print(f"{split_name}: {len(split_indices)} samples "
              f"(Normal: {(labels[split_indices] == 0).sum()}, "
              f"FHP: {(labels[split_indices] == 1).sum()})")


def create_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    temporal: bool = True,
    seq_length: int = 30,
    augment_config: Optional[Dict] = None,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Returns:
        Dict with "train", "val", "test" DataLoaders
    """
    loaders = {}

    for split in ["train", "val", "test"]:
        is_train = split == "train"
        dataset = FHPDataset(
            data_dir=data_dir,
            split=split,
            temporal=temporal,
            seq_length=seq_length,
            augment=is_train,
            augment_config=augment_config if is_train else None,
        )

        sampler = dataset.get_sampler() if is_train else None

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(is_train and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
        )

    return loaders


if __name__ == "__main__":
    # Self-test with dummy data
    dataset = FHPDataset(
        data_dir="data/processed",
        split="train",
        temporal=True,
        seq_length=30,
        augment=True,
    )

    sample = dataset[0]
    print(f"Joints shape: {sample['joints'].shape}")
    print(f"Bio features shape: {sample['bio_features'].shape}")
    print(f"Label: {sample['label']}")
    print(f"Edge index shape: {sample['edge_index'].shape}")
    print(f"Class weights: {dataset.get_class_weights()}")

    loader = DataLoader(dataset, batch_size=16)
    batch = next(iter(loader))
    print(f"\nBatch joints: {batch['joints'].shape}")
    print(f"Batch labels: {batch['label'].shape}")
