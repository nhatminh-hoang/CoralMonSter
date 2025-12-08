"""
Prompt-agnostic semantic decoder that consumes pre-built query tokens.
"""

from __future__ import annotations

import copy
from typing import Dict

import torch
from torch import nn

from CoralMonSter.segment_anything.modeling.mask_decoder import MaskDecoder


class SemanticQueryDecoder(nn.Module):
    """
    Transformer + hypernetwork head that operates on provided query tokens.

    All prompt encoding must be performed outside this module; it only processes
    image embeddings and query tokens to produce semantic mask logits and token
    outputs.
    """

    def __init__(self, base_decoder: MaskDecoder, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.transformer_dim = base_decoder.transformer_dim
        self.transformer = copy.deepcopy(base_decoder.transformer)
        self.output_upscaling = copy.deepcopy(base_decoder.output_upscaling)

        proto_hyper = base_decoder.output_hypernetworks_mlps[0]
        self.hypernets = nn.ModuleList(
            [copy.deepcopy(proto_hyper) for _ in range(num_classes)]
        )

        self.semantic_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim, num_classes),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        queries: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        token_block, upscaled = self.run_transformer(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            input_tokens=queries,
        )
        logits = self.generate_masks(token_block, upscaled)
        # print(
        #     f"[SemanticQueryDecoder] input_tokens={tuple(queries.shape)} token_block={tuple(token_block.shape)} "
        #     f"upscaled={tuple(upscaled.shape)}"
        # )
        # print(
        #     f"[SemanticQueryDecoder] mask_logits={tuple(logits.shape)}"
        # )
        semantic_logits = self.semantic_head(token_block)

        return {
            "mask_logits": logits,
            "token_embeddings": token_block,
            "token_logits": semantic_logits,
        }

    def run_transformer(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        input_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the TwoWayTransformer and upscale image features.

        Returns the refined tokens from the transformer and the upscaled
        image feature map used for mask generation.
        """

        b, _, h, w = image_embeddings.shape

        pe = image_pe
        if pe.shape[0] == 1:
            pe = pe.expand(b, -1, -1, -1)

        hs, src = self.transformer(
            image_embedding=image_embeddings,
            image_pe=pe,
            point_embedding=input_tokens,
        )

        # hs: B x N_tokens x D, src: B x HW x D
        # print(
        #     f"[SemanticQueryDecoder] transformer_out hs={tuple(hs.shape)} src={tuple(src.shape)}"
        # )

        src = src.transpose(1, 2).view(b, self.transformer_dim, h, w)
        upscaled = self.output_upscaling(src)
        return hs, upscaled

    def generate_masks(
        self,
        refined_tokens: torch.Tensor,
        upscaled_image_features: torch.Tensor,
    ) -> torch.Tensor:
        """Generate semantic masks from refined tokens and upscaled features."""

        b = refined_tokens.shape[0]
        token_block = refined_tokens
        if token_block.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} refined tokens but received {token_block.shape[1]}"
            )

        upscaled_flat = upscaled_image_features.flatten(2)  # B x C' x HW
        hyper_in = torch.stack(
            [hyper(token_block[:, i, :]) for i, hyper in enumerate(self.hypernets)],
            dim=1,
        )
        logits = torch.matmul(hyper_in, upscaled_flat).view(
            b,
            self.num_classes,
            upscaled_image_features.shape[-2],
            upscaled_image_features.shape[-1],
        )
        return logits
