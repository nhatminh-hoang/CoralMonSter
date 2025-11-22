from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from CoralMonSter.losses.segmentation import (
    CoralSegmentationLoss,
    mask_distillation_loss,
    token_cross_entropy,
    token_kl_divergence,
)
from CoralMonSter.configs.hkcoral_config import HKCoralConfig

class CoralCriterion(nn.Module):
    """
    Computes all losses for CoralMonSter:
    - Segmentation loss (Dice + CE)
    - Mask distillation loss
    - Token distillation loss
    - Token classification loss
    """

    def __init__(self, cfg: HKCoralConfig):
        super().__init__()
        self.cfg = cfg
        self.seg_loss = CoralSegmentationLoss(
            dice_weight=cfg.distillation.dice_weight,
            ce_weight=cfg.distillation.ce_weight,
            ignore_index=cfg.ignore_label,
        )
        self.teacher_temperature = cfg.distillation.teacher_temperature_start

    def set_teacher_temperature(self, value: float):
        self.teacher_temperature = float(value)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        aux_losses: Dict[str, torch.Tensor] = {}
        
        logits = outputs["student_logits"]
        masks = batch.get("masks")
        
        # 1. Segmentation Loss
        if masks is not None:
            masks = masks.to(logits.device)
            losses["seg_loss"] = self.seg_loss(logits, masks)
            
            dice_loss = getattr(self.seg_loss, "last_dice_loss", None)
            ce_loss = getattr(self.seg_loss, "last_ce_loss", None)
            if dice_loss is not None:
                aux_losses["dice_loss"] = dice_loss
            if ce_loss is not None:
                aux_losses["ce_loss"] = ce_loss

        # 2. Token Classification Loss
        token_logits = outputs.get("student_token_logits")
        if token_logits is not None and self.cfg.distillation.token_classification_weight > 0:
            batch_size = logits.shape[0]
            class_targets = torch.arange(
                self.cfg.num_classes,
                device=token_logits.device,
            ).unsqueeze(0).expand(batch_size, -1).reshape(-1)
            token_logits_flat = token_logits.reshape(batch_size * self.cfg.num_classes, self.cfg.num_classes)
            token_cls_loss = F.cross_entropy(token_logits_flat, class_targets, label_smoothing=0.1)
            losses["token_cls_loss"] = (
                token_cls_loss * self.cfg.distillation.token_classification_weight
            )

        # 3. Distillation Losses
        teacher_logits = outputs.get("teacher_logits")
        if teacher_logits is not None:
            # Mask Distillation
            student_probs = torch.softmax(logits, dim=1)
            teacher_probs = torch.softmax(teacher_logits, dim=1).detach()
            mask_kd = mask_distillation_loss(student_probs, teacher_probs)
            losses["mask_kd_loss"] = mask_kd * self.cfg.distillation.mask_kd_weight

            # Token Distillation
            student_tokens = outputs.get("student_tokens")
            teacher_tokens = outputs.get("teacher_tokens")
            teacher_center = outputs.get("teacher_center")
            
            if student_tokens is not None and teacher_tokens is not None:
                if teacher_center is not None:
                    centered_teacher = teacher_tokens - teacher_center
                else:
                    centered_teacher = teacher_tokens

                metric = getattr(self.cfg.distillation, "token_kd_metric", "ce").lower()
                if metric == "kl":
                    token_kd = token_kl_divergence(
                        student_tokens,
                        centered_teacher,
                        student_temp=self.cfg.distillation.student_temperature,
                        teacher_temp=self.teacher_temperature,
                    )
                else:
                    token_kd = token_cross_entropy(
                        student_tokens,
                        centered_teacher,
                        student_temp=self.cfg.distillation.student_temperature,
                        teacher_temp=self.teacher_temperature,
                    )
                losses["token_kd_loss"] = token_kd * self.cfg.distillation.token_kd_weight

        total_loss = (
            torch.stack([v for v in losses.values()]).sum() if losses else torch.tensor(0.0, device=logits.device)
        )
        
        return {
            "total_loss": total_loss,
            **losses,
            **aux_losses
        }
