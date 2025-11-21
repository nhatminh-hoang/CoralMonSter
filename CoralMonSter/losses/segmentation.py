"""
Segmentation and distillation losses for CoralMonSter.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CoralSegmentationLoss(nn.Module):
    """
    Combines Dice + Cross-Entropy to enforce HKCoral supervision on the student.
    """

    def __init__(self, dice_weight: float, ce_weight: float, ignore_index: int = 255) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.last_dice_loss: torch.Tensor | None = None
        self.last_ce_loss: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ce_weight > 0:
            ce_loss = self.ce(logits, target)
        else:
            ce_loss = torch.tensor(0.0, device=logits.device)

        if self.dice_weight > 0:
            dice_loss = multi_class_dice_loss(logits, target, self.ignore_index)
        else:
            dice_loss = torch.tensor(0.0, device=logits.device)

        self.last_ce_loss = ce_loss.detach()
        self.last_dice_loss = dice_loss.detach()

        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


def multi_class_dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """
    Computes the Dice loss for multi-class segmentation with ignore_index.
    
    Args:
        logits: Predicted logits of shape (B, C, H, W).
        target: Ground truth labels of shape (B, H, W).
        ignore_index: Label index to ignore in the loss computation.
    
    Returns:
        Dice loss averaged over classes.
    """
    # 1. Create a binary mask for valid pixels (B, H, W)
    mask = target != ignore_index
    
    # 2. Expand mask for broadcasting to probabilities (B, 1, H, W)
    mask_expanded = mask.unsqueeze(1)

    # 3. Softmax to get probabilities
    probs = torch.softmax(logits, dim=1)
    
    # 4. Mask the probabilities! This is the step you were missing.
    #    We zero out predictions at ignored locations so they don't count in the denominator.
    probs = probs * mask_expanded

    # 5. One-hot encode target (B, H, W) -> (B, C, H, W)
    #    Note: F.one_hot expects classes at the last dim, so we permute.
    num_classes = logits.shape[1]
    
    # Create a cleaned target where ignore_index is mapped to a safe index (e.g., 0)
    # The specific value doesn't matter because we mask it out immediately after.
    target_safe = target.clone()
    target_safe[~mask] = 0 
    
    target_one_hot = F.one_hot(target_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # 6. Mask the target one-hot
    target_one_hot = target_one_hot * mask_expanded

    # 7. Compute Dice
    #    Sum over Batch, H, W (Global Batch Dice). 
    #    Standard practice usually sums over (H, W) then averages over Batch, 
    #    but your implementation used (0, 2, 3) which is valid for Batch Dice.
    dims = (0, 2, 3) 
    
    numerator = 2 * torch.sum(probs * target_one_hot, dims)
    
    # The denominator now correctly excludes probability mass from ignored pixels
    denominator = torch.sum(probs + target_one_hot, dims).clamp_min(1e-6)
    
    dice_score = numerator / denominator
    loss = 1 - dice_score.mean()
    
    return loss


def mask_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """
    Match per-class probability maps between the student and teacher.
    """

    if student.shape != teacher.shape:
        teacher = F.interpolate(
            teacher,
            size=student.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return F.mse_loss(student, teacher)


def token_kl_divergence(
    student: torch.Tensor,
    teacher: torch.Tensor,
    student_temp: float,
    teacher_temp: float,
) -> torch.Tensor:
    log_student = F.log_softmax(student / student_temp, dim=-1)
    soft_teacher = F.softmax(teacher / teacher_temp, dim=-1)
    return F.kl_div(log_student, soft_teacher, reduction="batchmean") * (student_temp ** 2)


def token_cross_entropy(
    student: torch.Tensor,
    teacher: torch.Tensor,
    student_temp: float,
    teacher_temp: float,
) -> torch.Tensor:
    log_student = F.log_softmax(student / student_temp, dim=-1)
    soft_teacher = F.softmax(teacher / teacher_temp, dim=-1)
    loss = -torch.sum(soft_teacher * log_student, dim=-1).mean()
    loss = loss * student_temp
    return loss
