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
from CoralMonSter.models.student_decoder import PromptFreeMaskDecoder
from CoralMonSter.models.backbones import build_sam_backbone
from CoralMonSter.data.hkcoral_dataset import sample_prompts_gpu
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
        self.student_decoder = PromptFreeMaskDecoder(sam.mask_decoder, num_classes=self.cfg.num_classes)
        self.teacher_decoder: Optional[PromptFreeMaskDecoder] = None
        self.teacher_ready = False
        self.momentum = self.cfg.optimization.ema_momentum_min

        if self.cfg.use_lora:
            print(f"Enabling LoRA for image encoder with r={self.cfg.lora_r}, alpha={self.cfg.lora_alpha}, dropout={self.cfg.lora_dropout}, target={self.cfg.lora_target_modules}")
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
        elif self.cfg.freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)
        if self.cfg.freeze_prompt_encoder:
            for p in self.prompt_encoder.parameters():
                p.requires_grad_(False)
            self.prompt_encoder.eval()

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

        enc_start = time.time()
        image_embeddings = self.image_encoder(images)
        encoder_time = time.time() - enc_start
        image_pe = self._image_pe(batch_size, images.device)
        encoder_snapshot = None
        if getattr(self.cfg, "enable_pca_logging", False):
            encoder_snapshot = image_embeddings.detach()

        student_start = time.time()
        student_out = self.student_decoder(image_embeddings, image_pe)
        student_time = time.time() - student_start
        logits = F.interpolate(
            student_out["mask_logits"],
            size=masks.shape[-2:] if masks is not None else (self.cfg.image_size, self.cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )
        
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
            # Get prompt_sets from batch or compute on GPU
            prompt_sets = batch.get("prompt_sets")
            if prompt_sets is None or prompt_sets[0] is None:
                # Compute prompts on GPU
                prompt_sets = sample_prompts_gpu(
                    masks.to(images.device),
                    self.cfg.num_classes,
                    self.cfg.ignore_label,
                    self.cfg.prompt_bins,
                )
            
            teacher_points, teacher_labels = self._sample_teacher_prompts(
                prompt_sets, images.device
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
                outputs["teacher_time"] = time.time() - teacher_start
                outputs["teacher_logits"] = teacher_out.get("mask_logits")
                
                student_tokens = self.student_token_proj(student_out["token_embeddings"]).mean(dim=1)
                teacher_tokens = self.teacher_token_proj(teacher_out["token_features"])
                
                outputs["student_tokens"] = student_tokens
                outputs["teacher_tokens"] = teacher_tokens
                outputs["teacher_center"] = self.teacher_center.clone()

                if self.training:
                    self._update_teacher_center(teacher_tokens.detach())

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

        query_embeddings = self._build_teacher_queries(image_embeddings.shape[0], sparse_embeddings)
        decoder_out = self.teacher_decoder(
            image_embeddings,
            image_pe,
            prompt_embeddings=query_embeddings,
            prepend_class_queries=False,
        )
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

    def _build_teacher_queries(
        self,
        batch_size: int,
        prompt_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.teacher_decoder is None:
            raise RuntimeError("Teacher decoder not initialized")
        class_queries = self.teacher_decoder.class_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        if prompt_embeddings is None:
            return class_queries
        return torch.cat([prompt_embeddings, class_queries], dim=1)

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
