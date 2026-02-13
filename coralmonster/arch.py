"""
CoralMonSter — Coral Image Momentum Segmenter

Paper: "CoralMonSter: Prompt-Free Segment Anything via Momentum Distillation
        for Automated Coral Reef Monitoring"

Architecture overview (Fig. 1):
  - Image Encoder:   ViT-B from SAM [4]
  - Student Decoder:  SQD with input T_student = Q            (Eq. 6)
  - Teacher Decoder:  SQD with input T_teacher = [Q; P_gt]    (Eq. 7)
  - EMA:              θ_t ← α·θ_t + (1-α)·θ_s                (Section 3.2.4)
  - DINO Centering:   running mean of teacher token outputs     [6]
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from coralmonster.decoder import SemanticQueryDecoder


class CoralMonSter(nn.Module):
    """
    CoralMonSter: teacher-student momentum segmentation framework.

    Args:
        sam_model:             Pre-built SAM model (image encoder + prompt encoder + mask decoder)
        num_classes:           Number of semantic classes (Ncls)
        image_size:            Input image resolution (H = W)
        freeze_image_encoder:  Whether to freeze backbone (controlled via config)
        initial_momentum:      Starting EMA momentum α (default: 0.996)
    """

    def __init__(
        self,
        sam_model,
        num_classes: int,
        image_size: int = 1024,
        freeze_image_encoder: bool = False,
        initial_momentum: float = 0.996,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.freeze_image_encoder = freeze_image_encoder

        # ── Extract components from SAM ──────────────────────────────────
        self.image_encoder = sam_model.image_encoder
        self.prompt_encoder = sam_model.prompt_encoder

        # Freeze image encoder if configured
        if self.freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)
            self.image_encoder.eval()

        # Prompt encoder is always frozen (standard SAM usage)
        for p in self.prompt_encoder.parameters():
            p.requires_grad_(False)
        self.prompt_encoder.eval()

        # ── Learnable Class Queries Q ∈ R^(Ncls × C) — Section 3.2.1 ────
        self.class_queries = nn.Parameter(
            torch.randn(num_classes, sam_model.mask_decoder.transformer_dim)
        )

        # ── Student Decoder — Eq. 6 ─────────────────────────────────────
        self.student_decoder = SemanticQueryDecoder(
            sam_model.mask_decoder, num_classes=num_classes
        )

        # ── Teacher Decoder (initialized lazily via EMA) — Section 3.2.4 ─
        self.teacher_decoder: Optional[SemanticQueryDecoder] = None
        self.teacher_ready = False
        self.momentum = initial_momentum

        # ── Token projection heads for distillation — Eq. 12 ────────────
        dim = self.student_decoder.transformer_dim
        self.student_token_proj = nn.Linear(dim, 256)
        self.teacher_token_proj = nn.Identity()

        # ── DINO centering buffer — prevents mode collapse [6] ──────────
        self.register_buffer(
            "teacher_center",
            torch.zeros(dim, dtype=torch.float32),
        )

        self.distillation_enabled = False

    # ==================================================================
    # Forward pass
    # ==================================================================

    def forward(
        self,
        images: torch.Tensor,
        gt_points: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
        compute_distillation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training and inference.

        Args:
            images:     (B, 3, H, W) input images
            gt_points:  (B, Ncls, K, 2) ground-truth point prompts per class
            gt_masks:   (B, H, W) ground-truth masks (for determining target size)
            compute_distillation: if True, run teacher path when distillation is enabled

        Returns:
            Dict with keys:
              student_logits          (B, Ncls, H_out, W_out) — upscaled mask logits
              student_token_logits    (B, Ncls, Ncls)
              student_token_embeddings (B, N_tokens, D)
              teacher_logits          (B, Ncls, H_out, W_out) [if distilling]
              student_tokens          (B, D')                  [if distilling]
              teacher_tokens          (B, D')                  [if distilling]
              teacher_center          (D,)                     [if distilling]
        """
        # ── 1. Encode image — Eq. 1 ─────────────────────────────────────
        image_embeddings, image_pe = self._encode_images(images)

        # ── 2. Prepare class queries Q — Section 3.2.1 ──────────────────
        batch_size = images.shape[0]
        class_queries = self.class_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # ── 3. Student forward — Eq. 6, 8-11 ────────────────────────────
        student_out = self.student_decoder(image_embeddings, image_pe, class_queries)

        # Determine output size (match GT mask if available, else image_size)
        if gt_masks is not None:
            target_size = gt_masks.shape[-2:]
        else:
            target_size = (self.image_size, self.image_size)

        # Upscale mask logits to target resolution
        logits = F.interpolate(
            student_out["mask_logits"],
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        outputs: Dict[str, torch.Tensor] = {
            "student_logits": logits,
            "student_token_logits": student_out["token_logits"],
            "student_token_embeddings": student_out["token_embeddings"],
        }

        # ── 4. Teacher forward (if distilling) — Eq. 7, Section 3.2.4 ──
        should_distill = (
            (self.training or compute_distillation)
            and self.distillation_enabled
            and self.teacher_ready
            and gt_points is not None
        )

        if should_distill:
            teacher_outputs = self._run_teacher(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                class_queries=class_queries,
                gt_points=gt_points,
                target_size=target_size,
            )

            if teacher_outputs is not None:
                outputs.update(teacher_outputs)

                # Project student tokens for distillation — Eq. 12
                # Use class tokens only (first Ncls), average over class dim
                student_class_tokens = student_out["token_embeddings"][:, :self.num_classes, :]
                student_tokens = self.student_token_proj(student_class_tokens).mean(dim=1)
                outputs["student_tokens"] = student_tokens
                outputs["teacher_center"] = self.teacher_center.clone()

                # Update DINO center buffer during training
                if self.training:
                    self._update_teacher_center(
                        teacher_outputs["teacher_tokens"].detach()
                    )

        return outputs

    # ==================================================================
    # Teacher path
    # ==================================================================

    def _run_teacher(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        class_queries: torch.Tensor,
        gt_points: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Run teacher decoder with privileged GT prompts — Eq. 7.

        T_teacher = [Q; P_gt]

        The teacher is NOT optimized by gradients; all outputs are detached.
        """
        # Encode GT points into prompt embeddings P_gt
        prompt_tokens = self._encode_gt_prompts(gt_points)
        if prompt_tokens is None:
            return None

        # Concatenate: [Class Queries; Prompt Tokens] — Eq. 7
        teacher_input = torch.cat([class_queries, prompt_tokens], dim=1)

        # Run teacher transformer — Eq. 8
        refined_tokens, upscaled_feats = self.teacher_decoder.run_transformer(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            input_tokens=teacher_input,
        )

        # Slice class tokens (first Ncls) for mask generation
        teacher_class_tokens = refined_tokens[:, :self.num_classes, :]

        # Hypernetwork → mask kernels — Eq. 10
        hyper_in = torch.stack(
            [self.teacher_decoder.hypernets[i](teacher_class_tokens[:, i, :])
             for i in range(self.num_classes)],
            dim=1,
        )

        # Dynamic mask synthesis — Eq. 11
        teacher_mask_logits = self.teacher_decoder._generate_masks(
            hyper_in, upscaled_feats
        )

        # Upscale to target resolution
        teacher_logits = F.interpolate(
            teacher_mask_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        # Project teacher tokens for distillation — Eq. 12
        teacher_tokens = self.teacher_token_proj(teacher_class_tokens).mean(dim=1)

        return {
            "teacher_logits": teacher_logits.detach(),
            "teacher_tokens": teacher_tokens.detach(),
        }

    # ==================================================================
    # Encoding helpers
    # ==================================================================

    def _encode_images(
        self, images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images through SAM's ViT backbone — Eq. 1.

        When freeze_image_encoder is True, uses torch.no_grad() to save memory.
        """
        if self.freeze_image_encoder:
            with torch.no_grad():
                embeddings = self.image_encoder(images)
        else:
            embeddings = self.image_encoder(images)

        # Get positional encoding from prompt encoder
        pe = self.prompt_encoder.get_dense_pe().to(images.device)
        if pe.shape[0] == 1:
            pe = pe.expand(images.shape[0], -1, -1, -1)

        return embeddings, pe

    def _encode_gt_prompts(
        self, gt_points: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Encode per-class ground-truth points for teacher guidance.

        Args:
            gt_points: (B, Ncls, K, 2) — K points per class,
                       negative coords indicate invalid/absent points

        Returns:
            (B, Ncls*K, C) sparse prompt embeddings, or None if empty
        """
        b, num_classes, k, _ = gt_points.shape

        # Flatten: (B, Ncls*K, 2)
        coords = gt_points.to(self.device, dtype=torch.float32).view(b, num_classes * k, 2)

        # Label all points as foreground (1), invalid points as -1
        labels = torch.ones(
            (b, num_classes * k), device=self.device, dtype=torch.int64,
        )
        invalid = coords.lt(0).any(dim=-1)
        labels[invalid] = -1
        coords = coords.clone()
        coords[invalid] = 0.0

        # Encode through SAM's prompt encoder
        sparse_embeddings, _ = self.prompt_encoder(
            points=(coords, labels), boxes=None, masks=None,
        )

        if sparse_embeddings.numel() == 0:
            return None

        return sparse_embeddings

    # ==================================================================
    # EMA and distillation management
    # ==================================================================

    def update_teacher(self) -> None:
        """
        EMA update of teacher parameters — Section 3.2.4.

        θ_t ← α·θ_t + (1-α)·θ_s
        """
        if not self.teacher_ready or self.teacher_decoder is None:
            return

        with torch.no_grad():
            for t_param, s_param in zip(
                self.teacher_decoder.parameters(),
                self.student_decoder.parameters(),
            ):
                t_param.data.mul_(self.momentum).add_(
                    s_param.data, alpha=1 - self.momentum
                )

    def _update_teacher_center(
        self,
        teacher_tokens: torch.Tensor,
        center_momentum: float = 0.9,
    ) -> None:
        """Update DINO-style running center of teacher tokens."""
        batch_center = teacher_tokens.mean(dim=0)
        self.teacher_center.mul_(center_momentum).add_(
            batch_center, alpha=1 - center_momentum
        )

    def set_distillation_enabled(self, enabled: bool) -> None:
        """Enable/disable distillation. Lazily initializes teacher on first enable."""
        self.distillation_enabled = bool(enabled)
        if self.distillation_enabled and not self.teacher_ready:
            self._initialize_teacher_from_student()

    def set_momentum(self, momentum: float) -> None:
        """Set EMA momentum α for teacher updates."""
        self.momentum = momentum

    def _initialize_teacher_from_student(self) -> None:
        """Create teacher decoder as a frozen deep copy of student."""
        self.teacher_decoder = copy.deepcopy(self.student_decoder)
        for p in self.teacher_decoder.parameters():
            p.requires_grad_(False)
        self.teacher_decoder.eval()
        self.teacher_ready = True

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
