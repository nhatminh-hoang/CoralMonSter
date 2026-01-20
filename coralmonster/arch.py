from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from coralmonster.decoder import SemanticQueryDecoder

class CoralMonSter(nn.Module):
    """
    CoralMonSter: Coral Image Momentum Segmenter.
    
    Architecture:
    - Backbone: ViT-B (SAM)
    - Decoder: Semantic Query Decoder (SQD)
    - Distillation: Student (Classes) -> Teacher (Classes + Points)
    - Update: EMA
    """

    def __init__(
        self,
        sam_model,
        num_classes: int,
        image_size: int = 1024,
        freeze_image_encoder: bool = False, # Default should be false as per plan
        initial_momentum: float = 0.996,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Extract components from SAM
        self.image_encoder = sam_model.image_encoder
        self.prompt_encoder = sam_model.prompt_encoder
        
        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)
            self.image_encoder.eval()
        
        # Freeze Prompt Encoder (always frozen as per standard SAM usage usually)
        for p in self.prompt_encoder.parameters():
            p.requires_grad_(False)
        self.prompt_encoder.eval()

        # Learnable Class Queries
        self.class_queries = nn.Parameter(
            torch.randn(num_classes, sam_model.mask_decoder.transformer_dim)
        )

        # Student Decoder
        self.student_decoder = SemanticQueryDecoder(
            sam_model.mask_decoder, num_classes=num_classes
        )
        
        # Teacher Decoder (EMA Copy)
        self.teacher_decoder: Optional[SemanticQueryDecoder] = None
        self.teacher_ready = False
        self.momentum = initial_momentum

        # Projections for Token Distillation
        self.student_token_proj = nn.Linear(self.student_decoder.transformer_dim, 256)
        self.teacher_token_proj = nn.Identity()

        # DINO Centering Buffer
        self.register_buffer(
            "teacher_center",
            torch.zeros(self.student_decoder.transformer_dim, dtype=torch.float32),
        )
        
        self.distillation_enabled = False

    def forward(self, images: torch.Tensor, gt_points: Optional[torch.Tensor] = None, compute_distillation: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        if compute_distillation is True (training), runs Teacher path if gt_points provided.
        """
        # 1. Encode Image
        image_embeddings, image_pe = self._encode_images(images)
        
        # 2. Prepare Class Queries
        batch_size = images.shape[0]
        class_queries = self.class_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 3. Student Forward
        student_out = self.student_decoder(image_embeddings, image_pe, class_queries)
        
        # Upscale masks
        logits = F.interpolate(
            student_out["mask_logits"],
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        
        outputs = {
            "student_logits": logits,
            "student_token_logits": student_out.get("token_logits"),
            "student_token_embeddings": student_out.get("token_embeddings"),
        }

        # 4. Teacher Forward (if training/distilling)
        if (self.training or compute_distillation) and self.distillation_enabled and self.teacher_ready and gt_points is not None:
             teacher_outputs = self._run_teacher(
                 image_embeddings=image_embeddings,
                 image_pe=image_pe,
                 class_queries=class_queries,
                 gt_points=gt_points,
                 target_size=(self.image_size, self.image_size)
             )
             
             if teacher_outputs:
                 outputs.update(teacher_outputs)
                 
                 # Process student tokens for distillation match
                 student_tokens = self.student_token_proj(student_out["token_embeddings"]).mean(dim=1)
                 outputs["student_tokens"] = student_tokens
                 outputs["teacher_center"] = self.teacher_center.clone()
                 
                 if self.training:
                     self._update_teacher_center(teacher_outputs["teacher_tokens"].detach())
        
        return outputs

    def _run_teacher(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        class_queries: torch.Tensor,
        gt_points: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> Optional[Dict[str, torch.Tensor]]:
        
        prompt_tokens = self._encode_gt_prompts(gt_points)
        if prompt_tokens is None:
            return None
            
        # Concat: [Class Queries, Prompt Tokens]
        teacher_input = torch.cat([class_queries, prompt_tokens], dim=1)
        
        # Run Teacher Decoder
        # refined_tokens: (B, N_all, D)
        refined_tokens, upscaled_feats, teacher_token_embed = self.teacher_decoder.run_transformer(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            input_tokens=teacher_input,
        )
        
        # Slice out Class Tokens (first Num_Classes)
        teacher_class_tokens = refined_tokens[:, :self.num_classes, :]
        
        # Generate Masks using Teacher's Hypernetworks on Class Tokens
        hyper_in = torch.stack(
            [hyper(teacher_class_tokens[:, i, :]) for i, hyper in enumerate(self.teacher_decoder.hypernets)],
            dim=1,
        )
        
        teacher_mask_logits = self.teacher_decoder.generate_masks(
            teacher_class_tokens,
            hyper_in,
            upscaled_feats,
        )
        
        teacher_logits = F.interpolate(
            teacher_mask_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        
        # Average tokens for distillation
        teacher_token_embed = self.teacher_token_proj(teacher_token_embed).mean(dim=1)
        
        return {
            "teacher_logits": teacher_logits.detach(),
            "teacher_tokens": teacher_token_embed,
        }

    def _encode_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_embeddings = self.image_encoder(images)
        
        # Generate PE
        # SAM PromptEncoder has get_dense_pe method
        pe = self.prompt_encoder.get_dense_pe().to(images.device)
        if pe.shape[0] == 1:
            pe = pe.expand(images.shape[0], -1, -1, -1)
            
        return image_embeddings, pe

    def _encode_gt_prompts(self, gt_points: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Encode per-class ground-truth points for teacher guidance.
        gt_points: (B, num_classes, points_per_class, 2)
        """
        # This logic is adapted from existing coral_monster.py
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
        return sparse_embeddings

    def update_teacher(self) -> None:
        """EMA Update of Teacher Parameters."""
        if not self.teacher_ready or self.teacher_decoder is None:
            return
        # teacher_params = dict(self.teacher_decoder.named_parameters())
        # student_params = dict(self.student_decoder.named_parameters())
        
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_decoder.parameters(), self.student_decoder.parameters()):
                teacher_param.data.mul_(self.momentum).add_(student_param.data, alpha=1 - self.momentum)

    def _update_teacher_center(self, teacher_tokens: torch.Tensor, center_momentum: float = 0.9) -> None:
        batch_center = teacher_tokens.mean(dim=0)
        self.teacher_center.mul_(center_momentum).add_(batch_center, alpha=1 - center_momentum)

    def set_distillation_enabled(self, enabled: bool) -> None:
        self.distillation_enabled = bool(enabled)
        if self.distillation_enabled and not self.teacher_ready:
            self._initialize_teacher_from_student()

    def set_momentum(self, momentum: float) -> None:
        """Set the EMA momentum value for teacher updates."""
        self.momentum = momentum

    def _initialize_teacher_from_student(self) -> None:
        self.teacher_decoder = copy.deepcopy(self.student_decoder)
        for p in self.teacher_decoder.parameters():
            p.requires_grad_(False)
        self.teacher_decoder.eval()
        self.teacher_ready = True

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
