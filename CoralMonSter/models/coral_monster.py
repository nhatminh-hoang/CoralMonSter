"""
Main CoralMonSter model definition.
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from CoralMonSter.configs.hkcoral_config import HKCoralConfig
from CoralMonSter.losses import (
    CoralSegmentationLoss,
    mask_distillation_loss,
    token_cross_entropy,
    token_kl_divergence,
)
from CoralMonSter.models.student_decoder import PromptFreeMaskDecoder
from CoralMonSter.segment_anything import sam_model_registry


class CoralMonSter(nn.Module):
    """
    Teacher-student semantic segmentation framework for HKCoral.
    """

    def __init__(self, cfg: HKCoralConfig) -> None:
        super().__init__()
        self.cfg = cfg.resolve_paths()
        sam = self._build_sam_backbone()

        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.student_decoder = PromptFreeMaskDecoder(sam.mask_decoder, num_classes=self.cfg.num_classes)
        self.teacher_decoder: Optional[PromptFreeMaskDecoder] = None
        self.teacher_ready = False
        self.momentum = self.cfg.optimization.ema_momentum_min

        if self.cfg.freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)
        if self.cfg.freeze_prompt_encoder:
            for p in self.prompt_encoder.parameters():
                p.requires_grad_(False)
            self.prompt_encoder.eval()

        self.seg_loss = CoralSegmentationLoss(
            dice_weight=self.cfg.distillation.dice_weight,
            ce_weight=self.cfg.distillation.ce_weight,
            ignore_index=self.cfg.ignore_label,
        )
        self.student_token_proj = nn.Linear(self.student_decoder.transformer_dim, 256)
        self.teacher_token_proj = nn.Identity()
        self.teacher_temperature = self.cfg.distillation.teacher_temperature_start
        self.register_buffer(
            "teacher_center",
            torch.zeros(self.student_decoder.transformer_dim, dtype=torch.float32),
        )
        self.distillation_enabled = False

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        images = batch["images"].to(self.device)
        masks = batch.get("masks")
        batch_size = images.shape[0]

        enc_start = time.time()
        image_embeddings = self.image_encoder(images)
        encoder_time = time.time() - enc_start
        image_pe = self._image_pe(batch_size, images.device)

        student_start = time.time()
        student_out = self.student_decoder(image_embeddings, image_pe)
        student_time = time.time() - student_start
        logits = F.interpolate(
            student_out["mask_logits"],
            size=masks.shape[-2:] if masks is not None else (self.cfg.image_size, self.cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )
        student_probs = torch.softmax(logits, dim=1)

        losses: Dict[str, torch.Tensor] = {}
        aux_losses: Dict[str, torch.Tensor] = {}
        teacher_time = 0.0
        if masks is not None:
            masks = masks.to(images.device)
            losses["seg_loss"] = self.seg_loss(logits, masks)
            dice_loss = getattr(self.seg_loss, "last_dice_loss", None)
            ce_loss = getattr(self.seg_loss, "last_ce_loss", None)
            if dice_loss is not None:
                aux_losses["dice_loss"] = dice_loss
            if ce_loss is not None:
                aux_losses["ce_loss"] = ce_loss
            if self.training and self.distillation_enabled and self.teacher_ready:
                teacher_points, teacher_labels = self._sample_teacher_prompts(
                    batch.get("prompt_sets"), images.device
                )
                if teacher_points is not None:
                    teacher_start = time.time()
                    teacher_out = self._teacher_forward(
                        image_embeddings,
                        image_pe,
                        teacher_points,
                        teacher_labels,
                        batch.get("boxes"),
                        return_logits=True,
                    )
                    teacher_time = time.time() - teacher_start
                    teacher_logits = teacher_out.get("mask_logits")
                    if teacher_logits is not None:
                        teacher_probs = torch.softmax(teacher_logits, dim=1).detach()
                        mask_kd = mask_distillation_loss(student_probs, teacher_probs)
                        losses["mask_kd_loss"] = mask_kd * self.cfg.distillation.mask_kd_weight

                    student_tokens = self.student_token_proj(student_out["token_embeddings"]).mean(dim=1)
                    teacher_tokens = self.teacher_token_proj(teacher_out["token_features"])
                    if self.cfg.distillation.center_momentum > 0:
                        centered_teacher = teacher_tokens - self.teacher_center
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
                    self._update_teacher_center(teacher_tokens.detach())

        total_loss = (
            torch.stack([v for v in losses.values()]).sum() if losses else torch.tensor(0.0, device=self.device)
        )
        outputs = {
            "logits": logits,
            "total_loss": total_loss,
            "encoder_time": encoder_time,
            "student_time": student_time,
            "teacher_time": teacher_time,
        }
        outputs.update(losses)
        outputs.update(aux_losses)
        return outputs

    @torch.no_grad()
    def predict(self, images: torch.Tensor, original_sizes: torch.Tensor) -> torch.Tensor:
        self.eval()
        embeddings = self.image_encoder(images.to(self.device))
        pe = self._image_pe(images.shape[0], images.device)
        out = self.student_decoder(embeddings, pe)
        logits = F.interpolate(out["mask_logits"], size=(self.cfg.image_size, self.cfg.image_size), mode="bilinear", align_corners=False)
        results = []
        for idx in range(images.shape[0]):
            h, w = original_sizes[idx].tolist()
            rescaled = F.interpolate(logits[idx : idx + 1], size=(h, w), mode="bilinear", align_corners=False)
            results.append(rescaled.squeeze(0))
        return torch.stack(results)

    @torch.no_grad()
    def teacher_predict(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Produce teacher semantic masks (argmax over logits) for visualization.
        """

        prompts = batch.get("prompt_sets")
        if not prompts:
            return None

        points, labels = self._sample_teacher_prompts(prompts, self.device)
        if points is None:
            return None

        if not self.teacher_ready:
            return None
        images = batch["images"].to(self.device)
        embeddings = self.image_encoder(images)
        image_pe = self._image_pe(images.shape[0], images.device)
        teacher_out = self._teacher_forward(
            embeddings,
            image_pe,
            points,
            labels,
            batch.get("boxes"),
            return_logits=True,
        )
        if "masks" in batch and batch["masks"] is not None:
            target_size = batch["masks"].shape[-2:]
        else:
            target_size = (self.cfg.image_size, self.cfg.image_size)
        logits = teacher_out["mask_logits"]
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)

    def _teacher_forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        points: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        boxes_data: Optional[List[Optional[torch.Tensor]]],
        return_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if not self.teacher_ready:
            raise RuntimeError("Teacher decoder not initialized yet.")
        boxes = self._stack_boxes(boxes_data or [], image_embeddings.device)

        sparse_embeddings, _ = self.prompt_encoder(
            points=(points, labels) if points is not None else None,
            boxes=boxes,
            masks=None,
        )

        decoder_out = self.teacher_decoder(image_embeddings, image_pe, prompt_embeddings=sparse_embeddings)
        teacher_logits = decoder_out["mask_logits"]
        teacher_probs = torch.softmax(teacher_logits, dim=1)
        token_features = decoder_out["token_embeddings"].mean(dim=1)

        result = {
            "mask_prob": teacher_probs.max(dim=1, keepdim=True)[0].detach(),
            "token_features": token_features.detach(),
        }
        if return_logits:
            result["mask_logits"] = teacher_logits.detach()
        return result

    def _image_pe(self, batch_size: int, device: torch.device) -> torch.Tensor:
        pe = self.prompt_encoder.get_dense_pe().to(device)
        if pe.shape[0] == 1:
            pe = pe.expand(batch_size, -1, -1, -1)
        return pe

    @staticmethod
    def _stack_boxes(boxes: List[Optional[torch.Tensor]], device: torch.device) -> Optional[torch.Tensor]:
        if not any(b is not None for b in boxes):
            return None
        stacked = []
        for box in boxes:
            if box is None:
                stacked.append(torch.zeros(4, device=device))
            else:
                stacked.append(box.to(device))
        return torch.stack(stacked, dim=0)

    def _sample_teacher_prompts(
        self,
        prompt_sets: Optional[List[Optional[Dict[int, Any]]]],
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not prompt_sets:
            return None, None
        selected: List[Optional[Any]] = []
        max_points = 0
        for prompts in prompt_sets:
            if not prompts:
                selected.append(None)
                continue
            available = [k for k, v in prompts.items() if v.coords.numel() > 0]
            if not available:
                selected.append(None)
                continue
            choice = random.choice(available)
            sample = prompts[choice]
            selected.append(sample)
            max_points = max(max_points, sample.coords.shape[0])

        if max_points == 0:
            return None, None

        coords = torch.zeros((len(selected), max_points, 2), device=device)
        labels = -torch.ones((len(selected), max_points), device=device)
        for idx, sample in enumerate(selected):
            if sample is None or sample.coords.numel() == 0:
                continue
            count = min(sample.coords.shape[0], max_points)
            coords[idx, :count] = sample.coords[:count].to(device)
            labels[idx, :count] = sample.labels[:count].to(device)
        return coords, labels

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def update_teacher(self) -> None:
        if not self.teacher_ready or self.teacher_decoder is None:
            return
        for teacher_param, student_param in zip(self.teacher_decoder.parameters(), self.student_decoder.parameters()):
            teacher_param.data.mul_(self.momentum).add_(student_param.data, alpha=1 - self.momentum)

    def set_momentum(self, momentum: float) -> None:
        self.momentum = float(momentum)

    def set_teacher_temperature(self, value: float) -> None:
        self.teacher_temperature = float(value)

    def set_distillation_enabled(self, enabled: bool) -> None:
        self.distillation_enabled = bool(enabled)
        if self.distillation_enabled and not self.teacher_ready:
            self._initialize_teacher_from_student()

    def _initialize_teacher_from_student(self) -> None:
        self.teacher_decoder = copy.deepcopy(self.student_decoder)
        for p in self.teacher_decoder.parameters():
            p.requires_grad_(False)
        self.teacher_decoder.eval()
        self.teacher_ready = True

    @torch.no_grad()
    def _update_teacher_center(self, teacher_tokens: torch.Tensor) -> None:
        if self.cfg.distillation.center_momentum <= 0:
            return
        center_m = self.cfg.distillation.center_momentum
        batch_center = teacher_tokens.mean(dim=0)
        self.teacher_center.mul_(center_m).add_(batch_center, alpha=1 - center_m)

    def _build_sam_backbone(self):
        checkpoint_path = self.cfg.sam_checkpoint
        if checkpoint_path.exists():
            checkpoint_arg = str(checkpoint_path)
        else:
            print(f"[Warning] SAM checkpoint '{checkpoint_path}' not found. Initializing from scratch.")
            checkpoint_arg = None
        try:
            return sam_model_registry[self.cfg.model_type](checkpoint=checkpoint_arg)
        except RuntimeError as exc:
            if checkpoint_arg and "state_dict" in str(exc):
                print(
                    "[Warning] Failed to load checkpoint into SAM backbone due to mismatched keys. "
                    "Falling back to random initialization. Make sure '--model_type' matches the checkpoint."
                )
                return sam_model_registry[self.cfg.model_type](checkpoint=None)
            raise
