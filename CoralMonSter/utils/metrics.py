"""
Metric utilities for CoralMonSter training/evaluation loops.
"""

from __future__ import annotations

import torch


class SegmentationMeter:
    """
    Accumulates a confusion matrix to compute mean IoU and pixel accuracy over multiple batches.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64)
        self.correct = 0.0
        self.total = 0.0

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.detach().view(-1).to(torch.int64)
        target = target.detach().view(-1).to(torch.int64)
        mask = target != self.ignore_index
        if mask.sum() == 0:
            return
        preds = preds[mask]
        target = target[mask]
        hist = torch.bincount(
            self.num_classes * target + preds,
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes).double()
        self.confusion += hist.cpu()
        self.correct += (preds == target).sum().item()
        self.total += target.numel()

    def mean_iou(self) -> float:
        intersection = torch.diag(self.confusion)
        union = self.confusion.sum(1) + self.confusion.sum(0) - intersection
        valid = union > 0
        if not valid.any():
            return 0.0
        iou = intersection[valid] / union[valid]
        return iou.mean().item()

    def pixel_accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total
