"""
Generate demo ST-GCN model weights for testing.

This creates a randomly initialized model and saves it so the
API can start without requiring training first.

Usage:
    python scripts/generate_demo_model.py
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.stgcn import create_model


def main():
    print("Generating demo ST-GCN model weights...")

    config = {
        "type": "stgcn",
        "in_channels": 3,
        "num_joints": 13,
        "num_frames": 30,
        "gcn_hidden": [64, 128, 64],
        "temporal_kernel_size": 9,
        "bio_feature_dim": 6,
        "num_classes": 2,
        "dropout": 0.5,
    }

    model = create_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Save weights
    output_dir = Path("models/exported")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stgcn_fhp.pth"

    torch.save(model.state_dict(), str(output_path))
    print(f"✅ Saved demo model to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\n⚠️  This is a randomly initialized model (DEMO MODE).")
    print(f"   Train the model for real predictions:")
    print(f"   python scripts/train.py --config config.yaml")


if __name__ == "__main__":
    main()
