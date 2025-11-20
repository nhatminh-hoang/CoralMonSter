"""
Prompt-free mask decoder that reuses SAM's transformer and hyper-net modules.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
from torch import nn

from CoralMonSter.segment_anything.modeling.mask_decoder import MaskDecoder


class PromptFreeMaskDecoder(nn.Module):
    """
    Wrapper that converts SAM's promptable mask decoder into a prompt-free,
    class-aware decoder. It keeps the original transformer and upscaling heads
    so that the student inherits SAM's strong mask priors.
    """

    def __init__(self, base_decoder: MaskDecoder, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.transformer_dim = base_decoder.transformer_dim
        self.transformer = copy.deepcopy(base_decoder.transformer)
        self.output_upscaling = copy.deepcopy(base_decoder.output_upscaling)

        proto_hyper = base_decoder.output_hypernetworks_mlps[0]
        self.hypernet = copy.deepcopy(proto_hyper)

        self.class_tokens = nn.Parameter(torch.randn(num_classes, self.transformer_dim))
        nn.init.orthogonal_(self.class_tokens)
        self.semantic_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim, num_classes),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        prompt_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, c, h, w = image_embeddings.shape
        pe = image_pe
        if pe.shape[0] == 1:
            pe = pe.expand(b, -1, -1, -1)
        class_queries = self.class_tokens.unsqueeze(0).expand(b, -1, -1)
        if prompt_embeddings is not None:
            queries = torch.cat([prompt_embeddings, class_queries], dim=1)
        else:
            queries = class_queries
        hs, src = self.transformer(
            image_embedding=image_embeddings,
            image_pe=pe,
            point_embedding=queries,
        )

        src = src.transpose(1, 2).view(b, self.transformer_dim, h, w)
        upscaled = self.output_upscaling(src)
        upscaled_flat = upscaled.flatten(2)  # B x C' x HW

        token_block = hs[:, -self.num_classes :, :]
        hyper_in = self.hypernet(token_block.reshape(b * self.num_classes, -1))
        hyper_in = hyper_in.view(b, self.num_classes, -1)
        logits = torch.matmul(hyper_in, upscaled_flat).view(
            b, self.num_classes, upscaled.shape[-2], upscaled.shape[-1]
        )

        semantic_tokens = token_block
        semantic_logits = self.semantic_head(semantic_tokens)

        return {
            "mask_logits": logits,
            "token_embeddings": semantic_tokens,
            "token_logits": semantic_logits,
        }
