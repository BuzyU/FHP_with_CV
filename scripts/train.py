"""
Training Script for ST-GCN FHP Classifier

Supports local and Google Colab execution.
Handles the full training loop with:
- Class-balanced sampling
- Learning rate scheduling
- Early stopping
- TensorBoard logging
- Checkpoint saving
- Metric tracking

Usage:
    python scripts/train.py --config config.yaml
    python scripts/train.py --config config.yaml --epochs 100 --lr 0.001
"""

import sys
import os
import argparse
import yaml
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.stgcn import create_model
from src.data.dataset import create_dataloaders, FHPDataset
from src.utils.skeleton import get_upper_body_adjacency
from src.utils.metrics import MetricsTracker, EarlyStopping


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    adj: torch.Tensor,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    tracker = MetricsTracker(num_classes=2, class_names=["Normal", "FHP"])
    total_loss = 0
    num_batches = 0

    for batch in loader:
        joints = batch["joints"].to(device)
        bio_feats = batch["bio_features"].to(device)
        labels = batch["label"].to(device)

        # Forward
        logits = model(joints, adj, bio_feats)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        tracker.update(preds, labels.cpu().numpy(), probs)
        total_loss += loss.item()
        num_batches += 1

    metrics = tracker.compute()
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    adj: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on validation/test set."""
    model.eval()
    tracker = MetricsTracker(num_classes=2, class_names=["Normal", "FHP"])
    total_loss = 0
    num_batches = 0

    for batch in loader:
        joints = batch["joints"].to(device)
        bio_feats = batch["bio_features"].to(device)
        labels = batch["label"].to(device)

        logits = model(joints, adj, bio_feats)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        tracker.update(preds, labels.cpu().numpy(), probs)
        total_loss += loss.item()
        num_batches += 1

    metrics = tracker.compute()
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics


def train(config: dict, args):
    """Full training pipeline."""
    print("=" * 60)
    print("FHP Detection — ST-GCN Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Override config with CLI args
    train_config = config["training"]
    if args.epochs:
        train_config["epochs"] = args.epochs
    if args.lr:
        train_config["learning_rate"] = args.lr
    if args.batch_size:
        train_config["batch_size"] = args.batch_size

    # Create model
    model_config = config["model"]
    model = create_model(model_config).to(device)
    summary = model.get_model_summary() if hasattr(model, "get_model_summary") else {}
    print(f"Model: {model_config.get('type', 'stgcn')}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Adjacency matrix
    adj = torch.tensor(get_upper_body_adjacency(), dtype=torch.float32, device=device)

    # Data loaders
    aug_config = config.get("augmentation", {})
    loaders = create_dataloaders(
        data_dir=config["paths"]["data_splits"],
        batch_size=train_config["batch_size"],
        temporal=True,
        seq_length=model_config["num_frames"],
        augment_config=aug_config if aug_config.get("enabled") else None,
        num_workers=0,
    )

    # Loss function with class balancing
    train_dataset = loaders["train"].dataset
    if train_config.get("class_weights") == "balanced":
        class_weights = train_dataset.get_class_weights().to(device)
        print(f"Class weights: {class_weights}")
    else:
        class_weights = None

    label_smoothing = train_config.get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    # Scheduler
    if train_config.get("scheduler") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_config["epochs"]
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Early stopping
    early_stopping = EarlyStopping(
        patience=train_config["early_stopping_patience"],
        mode="max",
    )

    # Checkpointing
    ckpt_dir = Path(config["paths"]["model_checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = Path(config["paths"].get("logs", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        print(f"TensorBoard: {log_dir}")
    except ImportError:
        writer = None
        print("TensorBoard not available (optional)")

    # Training history
    history = {"train": [], "val": []}

    print(f"\nTraining for {train_config['epochs']} epochs...")
    print(f"Batch size: {train_config['batch_size']}")
    print(f"Learning rate: {train_config['learning_rate']}")
    print("-" * 60)

    for epoch in range(train_config["epochs"]):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, criterion, adj, device, epoch)

        # Validate
        val_metrics = evaluate(model, loaders["val"], criterion, adj, device)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if writer:
            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
            writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
            writer.add_scalar("F1/train", train_metrics["macro_f1"], epoch)
            writer.add_scalar("F1/val", val_metrics["macro_f1"], epoch)
            writer.add_scalar("LR", current_lr, epoch)

        elapsed = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch + 1:3d}/{train_config['epochs']} | "
              f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.3f}/{val_metrics['accuracy']:.3f} | "
              f"F1: {train_metrics['macro_f1']:.3f}/{val_metrics['macro_f1']:.3f} | "
              f"LR: {current_lr:.6f} | {elapsed:.1f}s")

        # Save best model
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, str(ckpt_dir / "best_model.pth"))
            print(f"  ✅ New best model (F1: {best_val_f1:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, str(ckpt_dir / f"checkpoint_epoch_{epoch + 1}.pth"))

        # Early stopping
        if early_stopping.step(val_metrics["macro_f1"]):
            print(f"\n⏹️  Early stopping at epoch {epoch + 1}")
            break

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (Test Set)")
    print("=" * 60)

    # Load best model
    best_ckpt = torch.load(str(ckpt_dir / "best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate(model, loaders["test"], criterion, adj, device)

    # Detailed report
    test_tracker = MetricsTracker(num_classes=2, class_names=["Normal", "FHP"])
    model.eval()
    with torch.no_grad():
        for batch in loaders["test"]:
            joints = batch["joints"].to(device)
            bio_feats = batch["bio_features"].to(device)
            labels = batch["label"].to(device)
            logits = model(joints, adj, bio_feats)
            preds = logits.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            test_tracker.update(preds, labels.cpu().numpy(), probs)

    print(test_tracker.format_report())

    # Save history
    with open(str(ckpt_dir / "training_history.json"), "w") as f:
        json.dump({
            k: [{kk: float(vv) for kk, vv in m.items()} for m in v]
            for k, v in history.items()
        }, f, indent=2)

    # Export model
    export_dir = Path(config["paths"]["model_exported"])
    export_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(export_dir / "stgcn_fhp.pth"))
    print(f"\n✅ Model exported: {export_dir / 'stgcn_fhp.pth'}")
    print(f"   Best validation F1: {best_val_f1:.4f}")
    print(f"   Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Test F1 (macro): {test_metrics['macro_f1']:.4f}")

    if writer:
        writer.close()

    return history, test_metrics


def main():
    parser = argparse.ArgumentParser(description="FHP Detection — Train ST-GCN")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, args)


if __name__ == "__main__":
    main()
