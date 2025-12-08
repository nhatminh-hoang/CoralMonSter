"""
Main CoralMonSter model definition.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from CoralMonSter.configs.hkcoral_config import HKCoralConfig
from CoralMonSter.models.student_decoder import SemanticQueryDecoder
from CoralMonSter.models.backbones import build_sam_backbone
from peft import LoraConfig, get_peft_model


class CoralMonSter(nn.Module):
    """
    Teacher-student semantic segmentation framework for HKCoral.
    """

    def __init__(self, cfg: HKCoralConfig) -> None:
        super().__init__()
        self.cfg = cfg.resolve_paths()
        sam = build_sam_backbone(
            self.cfg.model_type,
            self.cfg.sam_checkpoint,
            use_gradient_checkpointing=self.cfg.use_gradient_checkpointing,
            use_flash_attention=self.cfg.use_flash_attention,
        )

        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder

        self.freeze_image_encoder = bool(self.cfg.freeze_image_encoder)
        self.freeze_prompt_encoder = bool(self.cfg.freeze_prompt_encoder)

        if self.cfg.use_lora:
            lora_config = LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                target_modules=self.cfg.lora_target_modules,
                lora_dropout=self.cfg.lora_dropout,
                bias="none",
                modules_to_save=[],
            )
            self.image_encoder = get_peft_model(self.image_encoder, lora_config)
            self.image_encoder.print_trainable_parameters()
            self.freeze_image_encoder = False
        elif self.freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)
            self.image_encoder.eval()

        if self.freeze_prompt_encoder:
            for p in self.prompt_encoder.parameters():
                p.requires_grad_(False)
            self.prompt_encoder.eval()

        self.class_queries = nn.Parameter(
            torch.randn(self.cfg.num_classes, sam.mask_decoder.transformer_dim)
        )

        self.student_decoder = SemanticQueryDecoder(
            sam.mask_decoder, num_classes=self.cfg.num_classes
        )
        self.teacher_decoder: Optional[SemanticQueryDecoder] = None
        self.teacher_ready = False
        self.momentum = self.cfg.optimization.ema_momentum_min

        self.student_token_proj = nn.Linear(self.student_decoder.transformer_dim, 256)
        self.teacher_token_proj = nn.Identity()
        self.register_buffer(
            "teacher_center",
            torch.zeros(self.student_decoder.transformer_dim, dtype=torch.float32),
        )
        self.distillation_enabled = False

    def forward(self, batch: Dict[str, Any], compute_distillation: bool = False) -> Dict[str, torch.Tensor]:
        images = batch["images"].to(self.device)
        masks = batch.get("masks")
        batch_size = images.shape[0]

        image_embeddings, image_pe, class_queries, encoder_time, encoder_snapshot = self._prepare_batch(images)
        # print(
        #     f"[CoralMonSter] images={tuple(images.shape)} embeddings={tuple(image_embeddings.shape)} "
        #     f"class_queries={tuple(class_queries.shape)}"
        # )

        student_start = time.time()
        student_out = self.student_decoder(image_embeddings, image_pe, class_queries)
        student_time = time.time() - student_start

        target_size = masks.shape[-2:] if masks is not None else (self.cfg.image_size, self.cfg.image_size)
        logits = F.interpolate(
            student_out["mask_logits"],
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        # print(
        #     f"[CoralMonSter] student mask_logits={tuple(student_out['mask_logits'].shape)} "
        #     f"interp logits={tuple(logits.shape)}"
        # )

        outputs = {
            "student_logits": logits,
            "student_token_logits": student_out.get("token_logits"),
            "encoder_time": encoder_time,
            "student_time": student_time,
            "teacher_time": 0.0,
        }
        if encoder_snapshot is not None:
            outputs["encoder_embeddings"] = encoder_snapshot

        distill_active = (
            (self.training or compute_distillation)
            and self.distillation_enabled
            and self.teacher_ready
        )

        if distill_active:
            teacher_outputs = self._maybe_run_teacher(
                batch=batch,
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                class_queries=class_queries,
                target_size=target_size,
            )
            if teacher_outputs is not None:
                outputs.update(teacher_outputs)
                student_tokens = self.student_token_proj(student_out["token_embeddings"]).mean(dim=1)
                outputs["student_tokens"] = student_tokens
                outputs["teacher_center"] = self.teacher_center.clone()
                if self.training:
                    self._update_teacher_center(teacher_outputs["teacher_tokens"].detach())

        return outputs

    @torch.no_grad()
    def predict(self, images: torch.Tensor, original_sizes: torch.Tensor) -> torch.Tensor:
        self.eval()
        embeddings, pe, class_queries, _, _ = self._prepare_batch(images)
        out = self.student_decoder(embeddings, pe, class_queries)
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
        if not self.teacher_ready:
            return None

        prompt_tokens = self._resolve_teacher_prompts(batch, self.device, require_gt=True)
        if prompt_tokens is None:
            return None

        images = batch["images"].to(self.device)
        embeddings, image_pe, class_queries, _, _ = self._prepare_batch(images)
        if "masks" in batch and batch["masks"] is not None:
            target_size = batch["masks"].shape[-2:]
        else:
            target_size = (self.cfg.image_size, self.cfg.image_size)

        teacher_result = None
        teacher_result = self._run_teacher_concat(
            image_embeddings=embeddings,
            image_pe=image_pe,
            class_queries=class_queries,
            prompt_tokens=prompt_tokens,
            target_size=target_size,
        )
        if teacher_result is None:
            return None
        logits, _, _ = teacher_result
        return logits.detach().argmax(dim=1)

    def _image_pe(self, batch_size: int, device: torch.device) -> torch.Tensor:
        pe = self.prompt_encoder.get_dense_pe().to(device)
        if pe.shape[0] == 1:
            pe = pe.expand(batch_size, -1, -1, -1)
        return pe

    def _prepare_batch(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]]:
        """Encode images once and prepare shared positional encodings and class queries."""
        # images (B, 3, H, W)
        image_embeddings, image_pe, encoder_time, encoder_snapshot = self._encode_images(images) # (B, D, H, W), (B, D, H, W)
        class_queries = self.class_queries.unsqueeze(0).expand(images.shape[0], -1, -1) # (B, D) -> (B, K, D)
        return image_embeddings, image_pe, class_queries, encoder_time, encoder_snapshot

    def _resolve_teacher_prompts(
        self,
        batch: Dict[str, Any],
        device: torch.device,
        require_gt: bool = False,
    ) -> Optional[torch.Tensor]:
        gt_points = batch.get("gt_points")
        if gt_points is None:
            if require_gt:
                raise ValueError("gt_points is required for teacher prompts but was not provided")
            return None
        return self._encode_gt_prompts(gt_points.to(device))

    def _encode_gt_prompts(self, gt_points: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Encode per-class ground-truth points for teacher guidance.

        Expects gt_points shaped (B, num_classes, points_per_class, 2).
        Padding slots should be filled with -1; these are masked out.
        """

        if gt_points is None:
            return None

        b, num_classes, points_per_class, _ = gt_points.shape
        coords = gt_points.to(self.device, dtype=torch.float32).view(
            b, num_classes * points_per_class, 2
        )
        labels = torch.ones(
            (b, num_classes * points_per_class),
            device=self.device,
            dtype=torch.int64,
        )

        invalid = coords.lt(0).any(dim=-1)
        labels[invalid] = -1
        coords = coords.clone()
        coords[invalid] = 0.0

        sparse_embeddings, _ = self.prompt_encoder(
            points=(coords, labels),
            boxes=None,
            masks=None,
        )
        if sparse_embeddings.numel() == 0:
            return None
        # print(
        #     f"[CoralMonSter] gt_points coords={tuple(coords.shape)} labels={tuple(labels.shape)} "
        #     f"prompt_embeddings={tuple(sparse_embeddings.shape)}"
        # )
        return sparse_embeddings

    def _encode_images(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]]:
        enc_start = time.time()
        if self.freeze_image_encoder:
            with torch.no_grad():
                image_embeddings = self.image_encoder(images)
        else:
            image_embeddings = self.image_encoder(images)
        encoder_time = time.time() - enc_start
        image_pe = self._image_pe(images.shape[0], images.device)
        encoder_snapshot = image_embeddings.detach() if getattr(self.cfg, "enable_pca_logging", False) else None
        return image_embeddings, image_pe, encoder_time, encoder_snapshot

    def _maybe_run_teacher(
        self,
        batch: Dict[str, Any],
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        class_queries: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> Optional[Dict[str, torch.Tensor]]:
        prompt_tokens = self._resolve_teacher_prompts(batch, image_embeddings.device, require_gt=True)
        if prompt_tokens is None:
            return None

        teacher_start = time.time()
        teacher_result = self._run_teacher_concat(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            class_queries=class_queries,
            prompt_tokens=prompt_tokens,
            target_size=target_size,
        )
        if teacher_result is None:
            return None

        teacher_logits, teacher_class_tokens, teacher_tokens = teacher_result
        return {
            "teacher_time": time.time() - teacher_start,
            "teacher_logits": teacher_logits.detach(),
            "teacher_tokens": teacher_tokens,
        }

    def _run_teacher_concat(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        class_queries: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Run privileged teacher path: concat queries + prompts, attend, slice, mask.
        Returns interpolated mask logits, class tokens, and pooled token features.
        """

        # class_queries: (B, K, D), prompt_tokens: (B, K*P, D)

        teacher_input = torch.cat([class_queries, prompt_tokens], dim=1)
        refined_tokens, upscaled_feats = self.teacher_decoder.run_transformer(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            input_tokens=teacher_input,
        )

        teacher_class_tokens = refined_tokens[:, : self.cfg.num_classes, :]
        teacher_mask_logits = self.teacher_decoder.generate_masks(
            teacher_class_tokens,
            upscaled_feats,
        )
        teacher_logits = F.interpolate(
            teacher_mask_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        teacher_tokens = self.teacher_token_proj(teacher_class_tokens).mean(dim=1)
        # print(
        #     f"[CoralMonSter] prompt_tokens={tuple(prompt_tokens.shape)} concat={tuple(teacher_input.shape)} "
        #     f"refined={tuple(refined_tokens.shape)} class_tokens={tuple(teacher_class_tokens.shape)} "
        #     f"teacher_logits={tuple(teacher_logits.shape)} teacher_tokens={tuple(teacher_tokens.shape)}"
        # )
        return teacher_logits, teacher_class_tokens, teacher_tokens


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
