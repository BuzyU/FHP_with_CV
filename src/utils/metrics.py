"""
Evaluation Metrics Utilities

Provides metrics for model evaluation during training and testing:
- Accuracy, Precision, Recall, F1 (per-class and macro)
- Confusion matrix
- AUC-ROC computation
- Metric logging and formatting
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class MetricsTracker:
    """
    Tracks and computes classification metrics across batches.

    Usage:
        tracker = MetricsTracker(num_classes=2, class_names=["Normal", "FHP"])
        tracker.update(predictions, labels)
        metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(self, num_classes: int = 2, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        """Reset all accumulated counts."""
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total = 0
        self.correct = 0
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def update(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ):
        """
        Update metrics with a batch of predictions.

        Args:
            predictions: (batch,) predicted class indices
            labels: (batch,) ground truth class indices
            probabilities: (batch, num_classes) optional — class probabilities
        """
        predictions = np.asarray(predictions).flatten()
        labels = np.asarray(labels).flatten()

        batch_size = len(predictions)
        self.total += batch_size
        self.correct += int(np.sum(predictions == labels))

        for pred, label in zip(predictions, labels):
            self.confusion[int(label), int(pred)] += 1

        self.all_preds.extend(predictions.tolist())
        self.all_labels.extend(labels.tolist())

        if probabilities is not None:
            self.all_probs.extend(probabilities.tolist())

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated data.

        Returns:
            Dictionary with: accuracy, per-class precision/recall/f1, macro_f1, etc.
        """
        metrics = {}

        # Overall accuracy
        metrics["accuracy"] = self.correct / max(self.total, 1)

        # Per-class metrics
        precisions, recalls, f1s = [], [], []

        for c in range(self.num_classes):
            tp = self.confusion[c, c]
            fp = self.confusion[:, c].sum() - tp
            fn = self.confusion[c, :].sum() - tp

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            name = self.class_names[c]
            metrics[f"precision_{name}"] = precision
            metrics[f"recall_{name}"] = recall
            metrics[f"f1_{name}"] = f1

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        # Macro averages
        metrics["macro_precision"] = np.mean(precisions)
        metrics["macro_recall"] = np.mean(recalls)
        metrics["macro_f1"] = np.mean(f1s)

        # AUC-ROC if probabilities available
        if len(self.all_probs) > 0 and self.num_classes == 2:
            metrics["auc_roc"] = self._compute_auc_roc()

        return metrics

    def _compute_auc_roc(self) -> float:
        """Compute AUC-ROC for binary classification."""
        try:
            probs = np.array(self.all_probs)
            labels = np.array(self.all_labels)

            # Probability of positive class (FHP = class 1)
            if probs.ndim == 2:
                pos_probs = probs[:, 1]
            else:
                pos_probs = probs

            # Sort by descending probability
            sorted_idx = np.argsort(-pos_probs)
            sorted_labels = labels[sorted_idx]

            # Compute TPR and FPR at each threshold
            total_pos = np.sum(labels == 1)
            total_neg = np.sum(labels == 0)

            if total_pos == 0 or total_neg == 0:
                return 0.5

            tp, fp = 0, 0
            auc = 0.0
            prev_fpr = 0.0

            for label in sorted_labels:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
                    fpr = fp / total_neg
                    tpr = tp / total_pos
                    auc += (fpr - prev_fpr) * tpr
                    prev_fpr = fpr

            return float(auc)

        except Exception:
            return 0.5

    def get_confusion_matrix(self) -> np.ndarray:
        """Return the confusion matrix."""
        return self.confusion.copy()

    def format_report(self) -> str:
        """Generate a formatted classification report string."""
        metrics = self.compute()

        lines = []
        lines.append("=" * 60)
        lines.append("Classification Report")
        lines.append("=" * 60)
        lines.append(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        lines.append("-" * 60)

        for c in range(self.num_classes):
            name = self.class_names[c]
            p = metrics[f"precision_{name}"]
            r = metrics[f"recall_{name}"]
            f = metrics[f"f1_{name}"]
            lines.append(f"{name:<15} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

        lines.append("-" * 60)
        lines.append(f"{'Macro Avg':<15} {metrics['macro_precision']:>10.4f} "
                      f"{metrics['macro_recall']:>10.4f} {metrics['macro_f1']:>10.4f}")
        lines.append(f"\nAccuracy: {metrics['accuracy']:.4f}")

        if "auc_roc" in metrics:
            lines.append(f"AUC-ROC:  {metrics['auc_roc']:.4f}")

        lines.append(f"\nConfusion Matrix:")
        lines.append(f"  {'':>12} " + " ".join(f"{n:>10}" for n in self.class_names))
        for i, name in enumerate(self.class_names):
            row = " ".join(f"{int(self.confusion[i, j]):>10}" for j in range(self.num_classes))
            lines.append(f"  {name:>12} {row}")

        lines.append("=" * 60)

        return "\n".join(lines)


class EarlyStopping:
    """
    Early stopping based on a monitored metric.

    Args:
        patience: Number of epochs to wait for improvement
        mode: 'min' (loss) or 'max' (accuracy/f1)
        min_delta: Minimum change to qualify as improvement
    """

    def __init__(self, patience: int = 15, mode: str = "max", min_delta: float = 0.001):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def step(self, value: float) -> bool:
        """
        Check if training should stop.

        Returns:
            True if should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = value
            return False

        improved = False
        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    @property
    def status_message(self) -> str:
        return (f"EarlyStopping: best={self.best_value:.4f}, "
                f"counter={self.counter}/{self.patience}")


if __name__ == "__main__":
    # Self-test
    tracker = MetricsTracker(num_classes=2, class_names=["Normal", "FHP"])

    np.random.seed(42)
    for _ in range(10):
        batch_preds = np.random.randint(0, 2, size=32)
        batch_labels = np.random.randint(0, 2, size=32)
        batch_probs = np.random.rand(32, 2)
        batch_probs = batch_probs / batch_probs.sum(axis=1, keepdims=True)

        tracker.update(batch_preds, batch_labels, batch_probs)

    print(tracker.format_report())
