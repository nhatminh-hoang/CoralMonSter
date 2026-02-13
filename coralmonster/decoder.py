"""
Semantic Query Decoder (SQD) — Paper Section 3.2

Replaces SAM's MaskDecoder with a query-based decoder that produces:
  - mask_logits   (B, Ncls, H', W')  — dense mask predictions
  - token_embeddings (B, N_tokens, D) — refined class/prompt tokens
  - token_logits  (B, N_tokens, Ncls) — semantic classification logits
  - hyper_inputs  (B, Ncls, D')       — hypernetwork outputs

Architecture (Eq. 8-11):
  1. Two-way transformer ξ(·) refines queries against image embeddings
  2. Shared hypernetwork ϕ(·) maps each class token → mask kernel
  3. Upscaling head ψ_up(·) recovers spatial resolution
  4. Dynamic convolution: M_pred(i)(h,w) = w_i · F(h,w)
"""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
from torch import nn

from coralmonster.segment_anything.modeling.mask_decoder import MLP, MaskDecoder


class SemanticQueryDecoder(nn.Module):
    """
    Transformer + hypernetwork head that operates on provided query tokens.

    Used for both:
      - Student path: queries = class queries Q  (Eq. 6)
      - Teacher path: queries = [Q; P_gt]        (Eq. 7)
    """

    def __init__(self, base_decoder: MaskDecoder, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.transformer_dim = base_decoder.transformer_dim

        # ── Transformer layers (from SAM TwoWayTransformer) ──────────────
        base_transformer = base_decoder.transformer
        self.layers = copy.deepcopy(base_transformer.layers)
        self.final_attn_token_to_image = copy.deepcopy(
            base_transformer.final_attn_token_to_image
        )
        self.norm_final_attn = copy.deepcopy(base_transformer.norm_final_attn)

        # ── Upscaling head ψ_up(·) — Eq. 9 ──────────────────────────────
        self.output_upscaling = copy.deepcopy(base_decoder.output_upscaling)

        # ── Shared Hypernetwork ϕ(·) — Eq. 10 ───────────────────────────
        # One MLP per class, all sharing the same architecture:
        #   ϕ : R^D → R^(D/8)
        self.hypernets = nn.ModuleList(
            [
                MLP(self.transformer_dim, self.transformer_dim,
                    self.transformer_dim // 8, 3)
                for _ in range(self.num_classes)
            ]
        )

        # ── Semantic output projection ───────────────────────────────────
        # Maps each token → class logit vector (used for token classification)
        self.semantic_out = MLP(
            self.transformer_dim,
            self.transformer_dim,
            self.num_classes,
            1,  # single-layer (linear projection)
        )

    # ==================================================================
    # Forward
    # ==================================================================

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        queries: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full decoder forward pass.

        Args:
            image_embeddings: (B, C, H_emb, W_emb) from image encoder
            image_pe:         (B, C, H_emb, W_emb) positional encoding
            queries:          (B, N_tokens, C)  — class queries or [Q; P_gt]

        Returns:
            dict with keys:
              mask_logits      (B, Ncls, H', W')
              token_embeddings (B, N_tokens, C)   — refined tokens from ξ
              token_logits     (B, N_tokens, Ncls) — semantic logits
              hyper_inputs     (B, Ncls, C')       — hypernetwork outputs
        """
        # ── Step 1: Run two-way transformer (Eq. 8) ──────────────────────
        refined_tokens, upscaled_features = self.run_transformer(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            input_tokens=queries,
        )
        # refined_tokens: (B, N_tokens, D)
        # upscaled_features: (B, D/8, 4*H_emb, 4*W_emb)

        # ── Step 2: Hypernetwork — generate mask kernels (Eq. 10) ────────
        # Use only the first `num_classes` tokens (class queries)
        class_tokens = refined_tokens[:, :self.num_classes, :]
        hyper_in = torch.stack(
            [self.hypernets[i](class_tokens[:, i, :])
             for i in range(self.num_classes)],
            dim=1,
        )  # (B, Ncls, D')

        # ── Step 3: Dynamic mask synthesis (Eq. 11) ──────────────────────
        mask_logits = self._generate_masks(hyper_in, upscaled_features)

        # ── Step 4: Semantic token logits ────────────────────────────────
        semantic_logits = self.semantic_out(class_tokens)

        return {
            "mask_logits": mask_logits,           # (B, Ncls, H', W')
            "token_embeddings": refined_tokens,   # (B, N_tokens, D)
            "token_logits": semantic_logits,       # (B, Ncls, Ncls)
            "hyper_inputs": hyper_in,              # (B, Ncls, D')
        }

    # ==================================================================
    # Transformer execution
    # ==================================================================

    def run_transformer(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        input_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the TwoWayTransformer layers and upscale image features.

        Implements ξ(T, Z) from Eq. 8 using the individual layers
        copied from SAM's TwoWayTransformer.

        Args:
            image_embeddings: (B, C, H, W)
            image_pe:         (B, C, H, W) or (1, C, H, W) — will be expanded
            input_tokens:     (B, N_tokens, C)

        Returns:
            refined_tokens:    (B, N_tokens, D) — H_token
            upscaled_features: (B, D/8, 4H, 4W) — F = ψ_up(H_out)
        """
        b, c, h, w = image_embeddings.shape

        # Expand PE if batch dim is 1
        pe = image_pe
        if pe.shape[0] == 1:
            pe = pe.expand(b, -1, -1, -1)

        # Flatten spatial dims: (B, C, H, W) → (B, HW, C)
        image_flat = image_embeddings.flatten(2).permute(0, 2, 1)
        pe_flat = pe.flatten(2).permute(0, 2, 1)

        # Initialize queries and keys for the two-way attention
        queries = input_tokens  # token side
        keys = image_flat       # image side

        # ── Two-way transformer blocks ───────────────────────────────────
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=input_tokens,
                key_pe=pe_flat,
            )

        # ── Final attention: tokens → image ──────────────────────────────
        q = queries + input_tokens
        k = keys + pe_flat
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        # ── Upscale image features (Eq. 9) ───────────────────────────────
        src = keys.transpose(1, 2).view(b, self.transformer_dim, h, w)
        upscaled = self.output_upscaling(src)  # (B, D/8, 4H, 4W)

        return queries, upscaled  # (refined_tokens, upscaled_features)

    # ==================================================================
    # Mask generation
    # ==================================================================

    def _generate_masks(
        self,
        hyper_in: torch.Tensor,
        upscaled_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dynamic mask synthesis — Eq. 11.

        M_pred(i)(h,w) = w_i · F(h,w)

        Args:
            hyper_in:          (B, Ncls, D') — mask kernels from hypernetwork
            upscaled_features: (B, D', H', W') — upscaled feature map F

        Returns:
            mask_logits: (B, Ncls, H', W')
        """
        b, n_cls, d_prime = hyper_in.shape
        h_out, w_out = upscaled_features.shape[-2:]

        # Flatten spatial: (B, D', H'W')
        features_flat = upscaled_features.flatten(2)

        # Dynamic convolution: (B, Ncls, D') @ (B, D', H'W') → (B, Ncls, H'W')
        logits = (hyper_in @ features_flat).view(b, n_cls, h_out, w_out)

        return logits
