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
    probs = torch.softmax(logits, dim=1)
    target_one_hot = torch.zeros_like(probs)
    mask = target != ignore_index
    masked_target = target.clone()
    masked_target[~mask] = 0
    target_one_hot.scatter_(1, masked_target.unsqueeze(1), 1.0)
    target_one_hot = target_one_hot * mask.unsqueeze(1)

    dims = (0, 2, 3)
    numerator = 2 * torch.sum(probs * target_one_hot, dims)
    denominator = torch.sum(probs + target_one_hot, dims).clamp_min(1e-6)
    loss = 1 - numerator / denominator
    return loss.mean()


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
