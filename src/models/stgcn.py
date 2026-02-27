"""
Spatio-Temporal Graph Convolutional Network (ST-GCN) for FHP Detection

The core classification model that learns to detect Forward Head Posture
from 3D skeleton data. Combines:

1. Spatial GCN — learns relationships between anatomically connected joints
2. Temporal Convolution — captures posture changes over time
3. Biomechanical Feature Fusion — integrates computed angles as auxiliary input

The GCN respects skeleton topology (adjacency matrix) so the shoulder
influences the elbow which influences the wrist — this chain is encoded
in the graph structure, unlike a flat FFNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class GraphConvolution(nn.Module):
    """
    Single Graph Convolution layer.

    Implements: H' = σ(D^{-1/2} A D^{-1/2} H W)

    Args:
        in_features: Input feature dimension per node
        out_features: Output feature dimension per node
        bias: Whether to use bias
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None

        # Xavier initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_nodes, in_features)
            adj: (num_nodes, num_nodes) — normalized adjacency

        Returns:
            (batch, num_nodes, out_features)
        """
        # H * W
        support = torch.matmul(x, self.weight)  # (B, N, out)
        # A * (H * W)
        output = torch.matmul(adj, support)  # (B, N, out)

        if self.bias is not None:
            output = output + self.bias

        return output


class SpatialGCNBlock(nn.Module):
    """
    Spatial GCN block with BatchNorm and residual connection.

    Processes a single frame's joint graph.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()

        self.gcn = GraphConvolution(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Residual
        self.residual = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_nodes, in_channels)
            adj: (num_nodes, num_nodes)

        Returns:
            (batch, num_nodes, out_channels)
        """
        res = x

        out = self.gcn(x, adj)
        # BatchNorm expects (B, C, N) — reshape temporarily
        B, N, C = out.shape
        out = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.relu(out)
        out = self.dropout(out)

        if self.residual is not None:
            res = self.residual(res)

        return out + res


class TemporalConvLayer(nn.Module):
    """
    Temporal convolution layer for processing frame sequences.

    Applies 1D convolution along the time dimension after spatial GCN.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 9, dropout: float = 0.3):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)

        Returns:
            (batch, out_channels, time)
        """
        res = x
        out = self.conv(x)

        if self.residual is not None:
            res = self.residual(res)

        return out + res


class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network for FHP Detection.

    Full architecture pipeline:
      1. Per-frame spatial GCN (learns joint relationships)
      2. Temporal convolution (learns posture dynamics)
      3. Biomechanical feature fusion (angles as auxiliary input)
      4. Classification head (Normal vs FHP)

    Args:
        in_channels: Features per joint (3 for x,y,z)
        num_joints: Number of joints (13 for upper body)
        num_frames: Temporal window size
        gcn_channels: List of GCN layer output dimensions
        temporal_kernel_size: Kernel size for temporal conv
        bio_feature_dim: Number of biomechanical features
        num_classes: Number of output classes (2: Normal/FHP)
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_joints: int = 13,
        num_frames: int = 30,
        gcn_channels: list = None,
        temporal_kernel_size: int = 9,
        bio_feature_dim: int = 6,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        if gcn_channels is None:
            gcn_channels = [64, 128, 64]

        self.num_joints = num_joints
        self.num_frames = num_frames
        self.num_classes = num_classes

        # --- Spatial GCN layers ---
        gcn_layers = []
        prev_channels = in_channels
        for channels in gcn_channels:
            gcn_layers.append(SpatialGCNBlock(prev_channels, channels, dropout=dropout * 0.6))
            prev_channels = channels
        self.spatial_gcn = nn.ModuleList(gcn_layers)

        # --- Temporal convolution ---
        temporal_in = gcn_channels[-1] * num_joints
        self.temporal_conv = nn.Sequential(
            TemporalConvLayer(temporal_in, 256, temporal_kernel_size, dropout * 0.6),
            TemporalConvLayer(256, 128, temporal_kernel_size, dropout * 0.6),
        )

        # --- Biomechanical feature stream ---
        self.bio_encoder = nn.Sequential(
            nn.Linear(bio_feature_dim * num_frames, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
        )

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

        # --- Attention (optional: learnable joint importance) ---
        self.joint_attention = nn.Sequential(
            nn.Linear(gcn_channels[-1], 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        joints: torch.Tensor,
        adj: torch.Tensor,
        bio_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            joints: (batch, num_frames, num_joints, 3) — 3D joint positions
            adj: (num_joints, num_joints) — normalized adjacency matrix
            bio_features: (batch, num_frames, bio_feature_dim) — biomechanical features

        Returns:
            (batch, num_classes) — class logits
        """
        B, T, N, C = joints.shape

        # --- Spatial GCN: process each frame ---
        spatial_outputs = []
        for t in range(T):
            x = joints[:, t, :, :]  # (B, N, C)

            for gcn_layer in self.spatial_gcn:
                x = gcn_layer(x, adj)  # (B, N, gcn_out)

            # Apply joint attention
            attn = self.joint_attention(x)  # (B, N, 1)
            x = x * attn

            # Flatten joints: (B, N * gcn_out)
            x = x.reshape(B, -1)
            spatial_outputs.append(x)

        # --- Temporal: stack frames and convolve ---
        # (B, N*gcn_out, T)
        temporal_input = torch.stack(spatial_outputs, dim=2)
        temporal_out = self.temporal_conv(temporal_input)  # (B, 128, T)

        # Global average pooling over time
        temporal_out = temporal_out.mean(dim=2)  # (B, 128)

        # --- Biomechanical feature stream ---
        bio_flat = bio_features.reshape(B, -1)  # (B, bio_dim * T)
        bio_out = self.bio_encoder(bio_flat)     # (B, 64)

        # --- Fuse and classify ---
        combined = torch.cat([temporal_out, bio_out], dim=1)  # (B, 192)
        logits = self.classifier(combined)  # (B, num_classes)

        return logits

    def predict(
        self,
        joints: torch.Tensor,
        adj: torch.Tensor,
        bio_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class with probabilities.

        Returns:
            (predictions, probabilities)
        """
        with torch.no_grad():
            logits = self.forward(joints, adj, bio_features)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        return preds, probs

    def get_model_summary(self) -> Dict:
        """Return model summary information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "architecture": "ST-GCN",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "num_joints": self.num_joints,
            "num_frames": self.num_frames,
            "num_classes": self.num_classes,
            "param_breakdown": {
                "spatial_gcn": sum(p.numel() for p in self.spatial_gcn.parameters()),
                "temporal_conv": sum(p.numel() for p in self.temporal_conv.parameters()),
                "bio_encoder": sum(p.numel() for p in self.bio_encoder.parameters()),
                "classifier": sum(p.numel() for p in self.classifier.parameters()),
                "attention": sum(p.numel() for p in self.joint_attention.parameters()),
            }
        }


class STGCNSingleFrame(nn.Module):
    """
    Simplified ST-GCN for single-frame mode (no temporal convolutions).

    Useful for real-time inference where only the current frame is available,
    or as a baseline comparison against the temporal model.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_joints: int = 13,
        gcn_channels: list = None,
        bio_feature_dim: int = 6,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        if gcn_channels is None:
            gcn_channels = [64, 128, 64]

        # Spatial GCN
        gcn_layers = []
        prev_channels = in_channels
        for channels in gcn_channels:
            gcn_layers.append(SpatialGCNBlock(prev_channels, channels, dropout * 0.6))
            prev_channels = channels
        self.spatial_gcn = nn.ModuleList(gcn_layers)

        # Biomechanical features
        self.bio_fc = nn.Sequential(
            nn.Linear(bio_feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )

        # Classifier
        gcn_flat = gcn_channels[-1] * num_joints
        self.classifier = nn.Sequential(
            nn.Linear(gcn_flat + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, joints, adj, bio_features):
        """
        Args:
            joints: (batch, num_joints, 3)
            adj: (num_joints, num_joints)
            bio_features: (batch, bio_feature_dim)
        """
        B = joints.shape[0]
        x = joints

        for gcn_layer in self.spatial_gcn:
            x = gcn_layer(x, adj)

        x = x.reshape(B, -1)
        bio = self.bio_fc(bio_features)
        combined = torch.cat([x, bio], dim=1)

        return self.classifier(combined)


def create_model(config: dict) -> nn.Module:
    """
    Factory function to create the appropriate model from config.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized model
    """
    model_type = config.get("type", "stgcn")

    if model_type == "stgcn":
        return STGCN(
            in_channels=config.get("in_channels", 3),
            num_joints=config.get("num_joints", 13),
            num_frames=config.get("num_frames", 30),
            gcn_channels=config.get("gcn_hidden", [64, 128, 64]),
            temporal_kernel_size=config.get("temporal_kernel_size", 9),
            bio_feature_dim=config.get("bio_feature_dim", 6),
            num_classes=config.get("num_classes", 2),
            dropout=config.get("dropout", 0.5),
        )
    elif model_type == "stgcn_single":
        return STGCNSingleFrame(
            in_channels=config.get("in_channels", 3),
            num_joints=config.get("num_joints", 13),
            gcn_channels=config.get("gcn_hidden", [64, 128, 64]),
            bio_feature_dim=config.get("bio_feature_dim", 6),
            num_classes=config.get("num_classes", 2),
            dropout=config.get("dropout", 0.5),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    from src.utils.skeleton import get_upper_body_adjacency

    print("=" * 60)
    print("ST-GCN Model Test")
    print("=" * 60)

    # Build adjacency
    adj = torch.tensor(get_upper_body_adjacency(), dtype=torch.float32)
    print(f"Adjacency shape: {adj.shape}")

    # --- Temporal model ---
    model = STGCN()
    summary = model.get_model_summary()
    print(f"\nTemporal ST-GCN:")
    print(f"  Total params: {summary['total_params']:,}")
    for name, count in summary['param_breakdown'].items():
        print(f"  {name}: {count:,}")

    # Test forward
    B, T, N = 4, 30, 13
    joints = torch.randn(B, T, N, 3)
    bio = torch.randn(B, T, 6)

    logits = model(joints, adj, bio)
    print(f"\n  Input:  joints {joints.shape}, bio {bio.shape}")
    print(f"  Output: logits {logits.shape}")

    preds, probs = model.predict(joints, adj, bio)
    print(f"  Preds: {preds.shape}, Probs: {probs.shape}")

    # --- Single-frame model ---
    sf_model = STGCNSingleFrame()
    sf_params = sum(p.numel() for p in sf_model.parameters())
    print(f"\nSingle-frame ST-GCN:")
    print(f"  Total params: {sf_params:,}")

    sf_joints = torch.randn(4, 13, 3)
    sf_bio = torch.randn(4, 6)
    sf_logits = sf_model(sf_joints, adj, sf_bio)
    print(f"  Input:  joints {sf_joints.shape}, bio {sf_bio.shape}")
    print(f"  Output: logits {sf_logits.shape}")

    print("\n✅ All model tests passed!")
