from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def multi_class_dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Dice loss with ignore_index support for multi-class segmentation."""
    mask = target != ignore_index
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    mask_expanded = mask.unsqueeze(1)
    probs = torch.softmax(logits, dim=1) * mask_expanded

    num_classes = logits.shape[1]
    target_safe = target.clone()
    target_safe[~mask] = 0
    target_one_hot = F.one_hot(target_safe, num_classes=num_classes).permute(0, 3, 1, 2).float() * mask_expanded

    dims = (0, 2, 3)
    numerator = 2 * torch.sum(probs * target_one_hot, dims)
    denominator = torch.sum(probs + target_one_hot, dims).clamp_min(1e-6)
    dice_score = numerator / denominator
    return 1.0 - dice_score.mean()

def mask_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Match per-class probability maps between the student and teacher using MSE."""
    if student.shape != teacher.shape:
        teacher = F.interpolate(
            teacher,
            size=student.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    # Use probabilities for MSE as per standard distillation
    student_probs = torch.softmax(student, dim=1)
    teacher_probs = torch.softmax(teacher, dim=1)
    return F.mse_loss(student_probs, teacher_probs)

def token_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    student_temp: float,
    teacher_temp: float,
) -> torch.Tensor:
    """Token distillation using Softmax Cross Entropy."""
    # Teacher is centered/sharpened outside or we just handle temperature here
    log_student = F.log_softmax(student / student_temp, dim=-1)
    soft_teacher = F.softmax(teacher / teacher_temp, dim=-1)
    loss = -torch.sum(soft_teacher * log_student, dim=-1).mean()
    return loss * student_temp # Scale by temperature as per standard practice

class CoralMonsterLoss(nn.Module):
    """
    Unified loss for CoralMonSter:
    - Dice Loss (Student vs GT)
    - Mask KD (Student vs Teacher) - MSE
    - Token KD (Student vs Teacher) - Cross Entropy
    """

    def __init__(self, cfg):
        super().__init__()
        self.ignore_index = cfg.get("ignore_index", 255)
        
        # Loss weights
        self.dice_weight = cfg.get("dice_weight", 1.5)
        self.mask_kd_weight = cfg.get("mask_kd_weight", 1.0)
        self.token_kd_weight = cfg.get("token_kd_weight", 1.5)
        
        # Distillation params
        self.student_temp = cfg.get("student_temp", 0.1)
        self.warmup_steps = cfg.get("warmup_steps", 0)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        masks: torch.Tensor,
        current_step: int,
        teacher_temp: float,
    ) -> Dict[str, torch.Tensor]:
        
        losses: Dict[str, torch.Tensor] = {}
        
        # 1. Dice Loss (Always active)
        pred_masks = outputs["student_logits"]
        dice_loss = multi_class_dice_loss(pred_masks, masks, self.ignore_index)
        losses["dice_loss"] = dice_loss * self.dice_weight

        # Check warmup
        if current_step < self.warmup_steps:
            losses["total_loss"] = losses["dice_loss"]
            return losses

        # 2. Distillation Losses
        teacher_logits = outputs.get("teacher_logits")
        if teacher_logits is not None:
            # Mask KD
            losses["mask_kd"] = mask_distillation_loss(pred_masks, teacher_logits) * self.mask_kd_weight

            # Token KD
            student_tokens = outputs.get("student_tokens")
            teacher_tokens = outputs.get("teacher_tokens")
            teacher_center = outputs.get("teacher_center")
            
            if student_tokens is not None and teacher_tokens is not None:
                # Apply centering to teacher tokens if available
                if teacher_center is not None:
                    centered_teacher = teacher_tokens - teacher_center
                else:
                    centered_teacher = teacher_tokens
                
                losses["token_kd"] = token_distillation_loss(
                    student_tokens,
                    centered_teacher,
                    self.student_temp,
                    teacher_temp
                ) * self.token_kd_weight

        losses["total_loss"] = sum(losses.values())
        return losses
