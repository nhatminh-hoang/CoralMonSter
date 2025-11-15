"""
Metric utilities for CoralMonSter training/evaluation loops.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch


class SegmentationMeter:
    """
    Accumulates a confusion matrix to compute mean IoU and pixel accuracy over multiple batches.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        eval_ignore_classes: Optional[Iterable[int]] = None,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eval_ignore_set = set(eval_ignore_classes or [])
        mask = torch.ones(self.num_classes, dtype=torch.bool)
        for cls_id in self.eval_ignore_set:
            if 0 <= cls_id < self.num_classes:
                mask[cls_id] = False
        self.eval_class_mask = mask
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
        if self.eval_ignore_set:
            eval_mask = torch.ones_like(target, dtype=torch.bool)
            for cls_id in self.eval_ignore_set:
                eval_mask &= target != cls_id
        else:
            eval_mask = None
        if eval_mask is not None:
            preds_eval = preds[eval_mask]
            target_eval = target[eval_mask]
        else:
            preds_eval = preds
            target_eval = target
        self.correct += (preds_eval == target_eval).sum().item()
        self.total += target_eval.numel()

    def mean_iou(self) -> float:
        intersection = torch.diag(self.confusion)
        union = self.confusion.sum(1) + self.confusion.sum(0) - intersection
        valid = (union > 0) & self.eval_class_mask
        if not valid.any():
            return 0.0
        iou = intersection[valid] / union[valid]
        return iou.mean().item()

    def per_class_iou(self) -> List[float]:
        intersection = torch.diag(self.confusion)
        union = self.confusion.sum(1) + self.confusion.sum(0) - intersection
        iou = torch.full_like(intersection, float("nan"))
        valid = (union > 0) & self.eval_class_mask
        iou[valid] = intersection[valid] / union[valid]
        return iou.tolist()

    def confusion_matrix(self) -> torch.Tensor:
        return self.confusion.clone()

    def pixel_accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total
