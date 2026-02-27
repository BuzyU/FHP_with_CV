"""
VideoPose3D — 2D→3D Pose Lifting

Wraps the VideoPose3D architecture for lifting 2D keypoints to 3D space.
Pretrained on Human3.6M (3.6 million 3D human poses).

Pipeline: 2D keypoints (17×2) → VideoPose3D → 3D keypoints (17×3)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from pathlib import Path


class TemporalConvBlock(nn.Module):
    """
    A single temporal convolution block with residual connection.

    Architecture:
      Conv1D → BatchNorm → ReLU → Dropout → Conv1D → BatchNorm → ReLU → Dropout + Residual
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dropout: float = 0.25, causal: bool = False):
        super().__init__()

        padding = (kernel_size - 1) // 2
        if causal:
            padding = kernel_size - 1

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Residual projection if channel dims differ
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        self.causal = causal
        self.kernel_size = kernel_size

    def forward(self, x):
        """x: (batch, channels, time)"""
        residual = x

        out = self.conv1(x)
        if self.causal:
            out = out[:, :, :x.shape[2]]  # trim causal padding
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.residual is not None:
            residual = self.residual(residual)

        return out + residual


class VideoPose3DModel(nn.Module):
    """
    VideoPose3D-style 2D→3D lifting network.

    A fully convolutional architecture that lifts 2D joint detections
    to 3D pose predictions. Can operate in single-frame or temporal mode.

    Architecture:
      Input (17×2) → Linear → [TemporalConvBlock × N] → Linear → Output (17×3)

    Args:
        num_joints: Number of joints (17 for H36M)
        in_features: Input features per joint (2 for x,y)
        out_features: Output features per joint (3 for x,y,z)
        hidden_dim: Hidden layer dimension
        num_blocks: Number of temporal conv blocks
        kernel_size: Temporal convolution kernel size
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_joints: int = 17,
        in_features: int = 2,
        out_features: int = 3,
        hidden_dim: int = 1024,
        num_blocks: int = 4,
        kernel_size: int = 1,   # 1 = single-frame mode
        dropout: float = 0.25,
    ):
        super().__init__()

        self.num_joints = num_joints
        self.in_features = in_features
        self.out_features = out_features

        input_dim = num_joints * in_features
        output_dim = num_joints * out_features

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Temporal convolution blocks
        self.blocks = nn.ModuleList([
            TemporalConvBlock(hidden_dim, hidden_dim, kernel_size, dropout)
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, output_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lift 2D keypoints to 3D.

        Args:
            x: (batch, time, num_joints, 2) — 2D keypoints

        Returns:
            (batch, time, num_joints, 3) — 3D keypoints
        """
        B, T, N, C = x.shape

        # Reshape: (B, T, N*C) → (B, N*C, T)
        x = x.reshape(B, T, N * C).permute(0, 2, 1)

        # Process
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_proj(x)

        # Reshape back: (B, N*3, T) → (B, T, N, 3)
        x = x.permute(0, 2, 1).reshape(B, T, N, self.out_features)

        return x

    def single_frame(self, keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Lift a single frame of 2D keypoints.

        Args:
            keypoints_2d: (batch, num_joints, 2) or (num_joints, 2)

        Returns:
            (batch, num_joints, 3) or (num_joints, 3)
        """
        squeeze = False
        if keypoints_2d.dim() == 2:
            keypoints_2d = keypoints_2d.unsqueeze(0)
            squeeze = True

        # Add time dimension
        x = keypoints_2d.unsqueeze(1)  # (B, 1, N, 2)
        out = self.forward(x)          # (B, 1, N, 3)
        out = out.squeeze(1)           # (B, N, 3)

        if squeeze:
            out = out.squeeze(0)

        return out


class VideoPose3DLifter:
    """
    High-level wrapper for VideoPose3D inference.

    Handles model loading, device management, and preprocessing.

    Usage:
        lifter = VideoPose3DLifter(model_path="models/exported/videopose3d.pth")
        joints_3d = lifter.lift(keypoints_2d)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        num_joints: int = 17,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_joints = num_joints

        # Create model
        self.model = VideoPose3DModel(num_joints=num_joints)

        # Load pretrained weights if available
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ Loaded VideoPose3D weights from {model_path}")
        else:
            print("⚠️  Using randomly initialized VideoPose3D "
                  "(weights will be downloaded during training)")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def lift(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """
        Lift 2D keypoints to 3D.

        Args:
            keypoints_2d: (N, 2), (T, N, 2), or (B, T, N, 2)

        Returns:
            3D joints matching input shape with last dim = 3
        """
        original_shape = keypoints_2d.shape

        # Normalize to (B, T, N, 2)
        if keypoints_2d.ndim == 2:
            x = keypoints_2d[np.newaxis, np.newaxis, ...]
        elif keypoints_2d.ndim == 3:
            x = keypoints_2d[np.newaxis, ...]
        else:
            x = keypoints_2d

        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        out = self.model(x)
        result = out.cpu().numpy()

        # Reshape to match input
        if keypoints_2d.ndim == 2:
            return result[0, 0]
        elif keypoints_2d.ndim == 3:
            return result[0]
        return result

    def lift_sequence(self, keypoints_2d_seq: np.ndarray) -> np.ndarray:
        """
        Lift a sequence of 2D keypoints.

        Args:
            keypoints_2d_seq: (T, 17, 2) — temporal sequence

        Returns:
            (T, 17, 3) — 3D sequence
        """
        return self.lift(keypoints_2d_seq)

    def get_model_info(self) -> dict:
        """Return model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
            "num_joints": self.num_joints,
        }


if __name__ == "__main__":
    # Self-test
    print("Testing VideoPose3D model...")

    model = VideoPose3DModel(num_joints=17, kernel_size=1)
    info = {
        "params": sum(p.numel() for p in model.parameters()),
    }
    print(f"Model parameters: {info['params']:,}")

    # Single frame
    x_single = torch.randn(4, 1, 17, 2)
    out_single = model(x_single)
    print(f"Single frame: {x_single.shape} → {out_single.shape}")

    # Temporal
    x_temp = torch.randn(4, 30, 17, 2)
    out_temp = model(x_temp)
    print(f"Temporal: {x_temp.shape} → {out_temp.shape}")

    # High-level wrapper
    lifter = VideoPose3DLifter()
    kp_2d = np.random.randn(17, 2).astype(np.float32)
    kp_3d = lifter.lift(kp_2d)
    print(f"\nLifter: {kp_2d.shape} → {kp_3d.shape}")

    info = lifter.get_model_info()
    print(f"Model info: {info}")
