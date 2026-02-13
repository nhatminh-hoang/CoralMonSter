"""
CoralMonSter Loss Functions — Paper Section 3.1, 3.2.4, 3.3

Total objective (Eq. 15):
    L_total = L_seg + S(t > T_warmup) * (λ_mask · L_mask_kd + λ_token · L_token_kd)

Where:
    L_seg      = λ_dice · L_dice + λ_CE · L_CE     (Eq. 5)
    L_dice     = 1 - (2·Σ(P·Y)) / (Σ(P) + Σ(Y))   (Eq. 3)
    L_CE       = -Σ Y·log(P)                         (Eq. 4)
    L_mask_kd  = ||M_student - M_teacher||²           (Eq. 14)
    L_token_kd = -Σ softmax((H'_t - c)/τ_t) · log(softmax(H'_s/τ_s))  (Eq. 13)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Individual loss functions
# =====================================================================

def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """
    Multi-class Dice loss — Eq. 3.

    L_dice = 1 - mean_over_classes( 2·Σ(p·y) / (Σp + Σy) )

    Args:
        logits: (B, Ncls, H, W) raw logits from model
        target: (B, H, W)      integer class labels
        ignore_index: label value to ignore (e.g. 255)

    Returns:
        Scalar Dice loss.
    """
    # Build valid-pixel mask
    valid = target != ignore_index  # (B, H, W)
    if not valid.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    valid_mask = valid.unsqueeze(1)  # (B, 1, H, W)

    # Softmax probabilities, zero out invalid pixels
    probs = torch.softmax(logits, dim=1) * valid_mask  # (B, Ncls, H, W)

    # One-hot encode target (replace ignore pixels with 0 first)
    num_classes = logits.shape[1]
    target_safe = target.clone()
    target_safe[~valid] = 0
    one_hot = (
        F.one_hot(target_safe, num_classes=num_classes)
        .permute(0, 3, 1, 2)
        .float()
        * valid_mask
    )  # (B, Ncls, H, W)

    # Dice coefficient per class (sum over B, H, W dimensions)
    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dim=dims)
    cardinality = torch.sum(probs + one_hot, dim=dims).clamp_min(1e-6)
    dice_per_class = 2.0 * intersection / cardinality

    return 1.0 - dice_per_class.mean()


def cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """
    Standard pixel-wise cross-entropy loss — Eq. 4.

    L_CE = -Σ Y(i) · log(softmax(M_pred)(i))

    Args:
        logits: (B, Ncls, H, W) raw logits
        target: (B, H, W)      integer class labels
        ignore_index: label value to ignore

    Returns:
        Scalar CE loss.
    """
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def mask_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Mask distillation — Eq. 14.

    L_mask_kd = ||M_student - M_teacher||²

    Operates on softmax probabilities for numerical stability.
    Teacher logits are explicitly detached to prevent gradient flow.

    Args:
        student_logits: (B, Ncls, H, W)
        teacher_logits: (B, Ncls, H, W) — already detached from arch.py,
                        but we add .detach() again for safety

    Returns:
        Scalar MSE loss.
    """
    # Align spatial dimensions if necessary
    if student_logits.shape[-2:] != teacher_logits.shape[-2:]:
        teacher_logits = F.interpolate(
            teacher_logits,
            size=student_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    student_probs = torch.softmax(student_logits, dim=1)
    teacher_probs = torch.softmax(teacher_logits.detach(), dim=1)

    return F.mse_loss(student_probs, teacher_probs)


def token_distillation_loss(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    teacher_center: torch.Tensor,
    student_temp: float,
    teacher_temp: float,
) -> torch.Tensor:
    """
    Token distillation with DINO-style centering — Eq. 12-13.

    L_token_kd = -Σ_k softmax((H'_teacher(k) - c) / τ_t)
                      · log(softmax(H'_student(k) / τ_s))

    Following DINO [6], the teacher output is centered (subtract running
    mean `c`) and sharpened (divide by τ_t < τ_s) to prevent mode collapse.

    Args:
        student_tokens: (B, D) projected student class token means
        teacher_tokens: (B, D) projected teacher class token means
        teacher_center: (D,)   running mean of teacher tokens
        student_temp:   τ_s (e.g. 0.1)
        teacher_temp:   τ_t (e.g. 0.04–0.07)

    Returns:
        Scalar cross-entropy distillation loss, scaled by τ_s.
    """
    # Center and sharpen teacher (Eq. 13)
    centered_teacher = (teacher_tokens.detach() - teacher_center) / teacher_temp
    soft_teacher = F.softmax(centered_teacher, dim=-1)

    # Student log-probabilities
    log_student = F.log_softmax(student_tokens / student_temp, dim=-1)

    # Cross-entropy: -Σ p_teacher · log(p_student)
    loss = -torch.sum(soft_teacher * log_student, dim=-1).mean()

    # Scale by student temperature (standard distillation practice)
    return loss * student_temp


# =====================================================================
# Unified loss module
# =====================================================================

class CoralMonsterLoss(nn.Module):
    """
    Unified loss for CoralMonSter training — Eq. 15.

    L_total = L_seg + S(t > T_warmup) · (λ_mask · L_mask_kd + λ_token · L_token_kd)

    where L_seg = λ_dice · L_dice + λ_CE · L_CE

    Config keys (from distillation section of YAML):
        dice_weight, ce_weight, mask_kd_weight, token_kd_weight,
        student_temp, warmup_epochs, ignore_index
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()

        dist_cfg = cfg.get("distillation", cfg)

        # ── Loss weights (Table 1) ───────────────────────────────────────
        self.dice_weight = dist_cfg["dice_weight"]          # λ_dice = 1.5
        self.ce_weight = dist_cfg["ce_weight"]              # λ_CE   = 1.5
        self.mask_kd_weight = dist_cfg["mask_kd_weight"]    # λ_mask = 1.0
        self.token_kd_weight = dist_cfg["token_kd_weight"]  # λ_token = 1.5

        # ── Distillation parameters ──────────────────────────────────────
        self.student_temp = dist_cfg["student_temp"]        # τ_s = 0.1
        self.warmup_epochs = dist_cfg.get("warmup_epochs", 0)

        # ── Segmentation parameters ──────────────────────────────────────
        self.ignore_index = cfg.get("ignore_index",
                                     dist_cfg.get("ignore_index", 255))

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        masks: torch.Tensor,
        current_epoch: int,
        teacher_temp: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for one batch.

        Args:
            outputs: dict from CoralMonSter.forward(), expected keys:
                - student_logits   (B, Ncls, H, W)
                - teacher_logits   (B, Ncls, H, W)  [optional]
                - student_tokens   (B, D)            [optional]
                - teacher_tokens   (B, D)            [optional]
                - teacher_center   (D,)              [optional]
            masks:         (B, H, W) integer ground-truth labels
            current_epoch: current epoch index (0-based)
            teacher_temp:  τ_t for this epoch

        Returns:
            Dict with individual loss components and total_loss.
        """
        losses: Dict[str, torch.Tensor] = {}
        pred = outputs["student_logits"]

        # ── Segmentation losses (always active) — Eq. 5 ─────────────────
        losses["dice_loss"] = dice_loss(pred, masks, self.ignore_index)
        losses["ce_loss"] = cross_entropy_loss(pred, masks, self.ignore_index)

        seg_loss = (
            self.dice_weight * losses["dice_loss"]
            + self.ce_weight * losses["ce_loss"]
        )
        losses["seg_loss"] = seg_loss

        # ── Distillation losses (after warmup) — Eq. 15 ─────────────────
        distillation_active = current_epoch >= self.warmup_epochs
        teacher_logits = outputs.get("teacher_logits")

        if distillation_active and teacher_logits is not None:
            # Mask KD — Eq. 14
            losses["mask_kd"] = mask_distillation_loss(pred, teacher_logits)

            # Token KD — Eq. 13
            student_tokens = outputs.get("student_tokens")
            teacher_tokens = outputs.get("teacher_tokens")
            center = outputs.get("teacher_center")

            if student_tokens is not None and teacher_tokens is not None and center is not None:
                losses["token_kd"] = token_distillation_loss(
                    student_tokens=student_tokens,
                    teacher_tokens=teacher_tokens,
                    teacher_center=center,
                    student_temp=self.student_temp,
                    teacher_temp=teacher_temp,
                )

            # Combine all losses
            total = seg_loss
            if "mask_kd" in losses:
                total = total + self.mask_kd_weight * losses["mask_kd"]
            if "token_kd" in losses:
                total = total + self.token_kd_weight * losses["token_kd"]
            losses["total_loss"] = total
        else:
            # Before warmup: only segmentation loss
            losses["total_loss"] = seg_loss

        return losses
