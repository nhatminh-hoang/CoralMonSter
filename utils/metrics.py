"""
Segmentation Metrics — SegmentationMeter

Accumulates a confusion matrix over batches and computes:
  - Mean IoU (Eq. 17 in paper)
  - Pixel Accuracy (Eq. 16 in paper)
  - Per-class IoU
"""

from __future__ import annotations

from typing import List, Optional

import torch
import numpy as np


class SegmentationMeter:
    """
    Accumulates a confusion matrix over batches for semantic segmentation.

    Usage:
        meter = SegmentationMeter(num_classes=7, ignore_index=255)
        for preds, targets in dataloader:
            meter.update(preds.argmax(1), targets)
        print(meter.mean_iou(), meter.pixel_accuracy())
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        eval_ignore_classes: Optional[List[int]] = None,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eval_ignore_classes = eval_ignore_classes or []

        # Confusion matrix: shape (num_classes, num_classes)
        # cm[i, j] = number of pixels with true class i predicted as class j
        self._confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.int64
        )

    def reset(self) -> None:
        """Clear accumulated statistics."""
        self._confusion_matrix[:] = 0

    @torch.no_grad()
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Add one batch of predictions to the confusion matrix.

        Args:
            predictions: (B, H, W) integer class indices (argmax of logits)
            targets:     (B, H, W) integer ground-truth labels
        """
        preds_flat = predictions.flatten().cpu().numpy()
        targets_flat = targets.flatten().cpu().numpy()

        # Mask out ignored pixels
        valid = targets_flat != self.ignore_index
        preds_flat = preds_flat[valid]
        targets_flat = targets_flat[valid]

        # Accumulate via bincount trick
        indices = targets_flat * self.num_classes + preds_flat
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self._confusion_matrix += counts.reshape(
            self.num_classes, self.num_classes
        )

    def confusion_matrix(self) -> np.ndarray:
        """Return the raw confusion matrix (num_classes x num_classes)."""
        return self._confusion_matrix.copy()

    def per_class_iou(self) -> np.ndarray:
        """
        Compute IoU for each class.

        IoU_n = TP_n / (TP_n + FP_n + FN_n)
              = cm[n,n] / (Σ_j cm[n,j] + Σ_j cm[j,n] - cm[n,n])

        Returns:
            Array of shape (num_classes,) with per-class IoU values.
        """
        cm = self._confusion_matrix
        tp = np.diag(cm)                          # true positives
        fp = cm.sum(axis=0) - tp                   # false positives
        fn = cm.sum(axis=1) - tp                   # false negatives

        denominator = tp + fp + fn
        iou = np.where(denominator > 0, tp / denominator, 0.0)
        return iou

    def mean_iou(self) -> float:
        """
        Mean Intersection over Union — Eq. 17.

        Excludes classes in eval_ignore_classes from the average.
        """
        iou = self.per_class_iou()
        mask = np.ones(self.num_classes, dtype=bool)
        for cls in self.eval_ignore_classes:
            if 0 <= cls < self.num_classes:
                mask[cls] = False

        valid_iou = iou[mask]
        if len(valid_iou) == 0:
            return 0.0
        return float(valid_iou.mean())

    def pixel_accuracy(self) -> float:
        """
        Pixel Accuracy — Eq. 16.

        PA = Σ TP_n / Σ (all pixels)
        """
        cm = self._confusion_matrix
        correct = np.diag(cm).sum()
        total = cm.sum()
        if total == 0:
            return 0.0
        return float(correct / total)

    def summary_string(self, class_names: Optional[List[str]] = None) -> str:
        """Pretty-print metrics with optional class names."""
        iou = self.per_class_iou()
        lines = []
        lines.append(f"mIoU: {self.mean_iou() * 100:.2f}%")
        lines.append(f"Pixel Accuracy: {self.pixel_accuracy() * 100:.2f}%")
        lines.append("Per-class IoU:")

        for i in range(self.num_classes):
            name = class_names[i] if class_names else f"Class {i}"
            marker = " (ignored)" if i in self.eval_ignore_classes else ""
            lines.append(f"  {name}: {iou[i] * 100:.2f}%{marker}")

        return "\n".join(lines)
