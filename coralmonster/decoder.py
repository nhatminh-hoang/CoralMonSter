from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
from torch import nn

# Import from the moved segment_anything package
from coralmonster.segment_anything.modeling.mask_decoder import MLP, MaskDecoder

class SemanticQueryDecoder(nn.Module):
    """
    Transformer + hypernetwork head that operates on provided query tokens.
    
    This decoder is used for both:
    1. Student Path: Input is Class Queries only.
    2. Teacher Path: Input is Class Queries + Ground Truth Prompt Tokens.
    """

    def __init__(self, base_decoder: MaskDecoder, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.transformer_dim = base_decoder.transformer_dim
        self.transformer = copy.deepcopy(base_decoder.transformer)
        self.output_upscaling = copy.deepcopy(base_decoder.output_upscaling)

        # Hypernetworks for each class
        self.hypernets = nn.ModuleList(
            [
                MLP(self.transformer_dim, self.transformer_dim, self.transformer_dim // 8, 3)
                for _ in range(self.num_classes)
            ]
        )

        # Semantic token projection
        self.semantic_out = MLP(
            self.transformer_dim,
            self.transformer_dim,
            self.num_classes,
            1,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        queries: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the Student (or simple inference).
        
        Args:
            image_embeddings: (B, Dim, H_emb, W_emb)
            image_pe: (B, Dim, H_emb, W_emb)
            queries: (B, NumClasses, Dim)
        """
        token_block, upscaled = self.run_transformer(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            input_tokens=queries,
        )

        # Generate masks via hypernetworks
        hyper_in = torch.stack(
            [hyper(token_block[:, i, :]) for i, hyper in enumerate(self.hypernets)],
            dim=1,
        ) # B x num_classes x D'

        logits = self.generate_masks(token_block, hyper_in, upscaled)
        
        # Semantic token logits (optional usage)
        semantic_logits = self.semantic_out(token_block)

        return {
            "mask_logits": logits,
            "token_embeddings": token_block,
            "token_logits": semantic_logits,
            "hyper_inputs": hyper_in,
        }

    def run_transformer(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        input_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the TwoWayTransformer and upscale image features.
        
        Args:
            input_tokens: Can be just Class Queries (Student) or Concat(Queries, Prompts) (Teacher)
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
        
        # hs: B x N_tokens x D
        # src: B x HW x D

        src = src.transpose(1, 2).view(b, self.transformer_dim, h, w)
        upscaled = self.output_upscaling(src)
        return hs, upscaled

    def generate_masks(
        self,
        token_block: torch.Tensor,
        hyper_in: torch.Tensor,
        upscaled_image_features: torch.Tensor,
    ) -> torch.Tensor:
        """Generate semantic masks from refined tokens and upscaled features."""
        
        b = token_block.shape[0]
        # Note: logic behaves differently if token_block has more tokens than classes (Teacher path)
        # But this function expects hyper_in to be prepared accordingly or handled outside for Teacher.
        # Actually for Teacher, we usually slice the tokens first.
        
        upscaled_flat = upscaled_image_features.flatten(2)  # B x C' x HW
        
        # hyper_in: B x NumClasses x C'
        # upscaled_flat: B x C' x HW
        
        logits = (hyper_in @ upscaled_flat).view(
            b,
            hyper_in.shape[1],
            upscaled_image_features.shape[-2],
            upscaled_image_features.shape[-1],
        ) # B x num_classes x H' x W'
        return logits
