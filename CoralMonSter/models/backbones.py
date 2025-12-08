from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from CoralMonSter.segment_anything import sam_model_registry

def build_sam_backbone(
    model_type: str,
    checkpoint_path: Path,
    use_gradient_checkpointing: bool = False,
    use_flash_attention: bool = False,
    image_size: int = 1024,
    random_init: bool = False,
):
    """
    Builds the SAM backbone model.
    """
    if random_init:
        print("[Info] SAM random initialization requested; ignoring checkpoint.")
        checkpoint_arg = None
    elif checkpoint_path.exists():
        checkpoint_arg = str(checkpoint_path)
    else:
        print(f"[Warning] SAM checkpoint '{checkpoint_path}' not found. Initializing from scratch.")
        checkpoint_arg = None
    
    try:
        sam = sam_model_registry[model_type](
            checkpoint=checkpoint_arg,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_flash_attention=use_flash_attention,
        )
        _retarget_sam_image_size(sam, image_size)
        return sam
    except RuntimeError as exc:
        if checkpoint_arg and "state_dict" in str(exc):
            print(
                "[Warning] Failed to load checkpoint into SAM backbone due to mismatched keys. "
                "Falling back to random initialization. Make sure '--model_type' matches the checkpoint."
            )
            sam = sam_model_registry[model_type](
                checkpoint=None,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_flash_attention=use_flash_attention,
            )
            _retarget_sam_image_size(sam, image_size)
            return sam
        raise


def _retarget_sam_image_size(sam, image_size: int) -> None:
    """Resize SAM positional embeddings and prompt encoder sizes for a new input resolution."""
    encoder = getattr(sam, "image_encoder", None)
    prompt_encoder = getattr(sam, "prompt_encoder", None)
    if encoder is None or prompt_encoder is None:
        return

    # Derive target grid from stride (assumes square patches and images)
    patch = encoder.patch_embed.proj.stride[0]
    target_grid = image_size // patch

    # Resize absolute positional embeddings if they exist and shapes differ
    if getattr(encoder, "pos_embed", None) is not None:
        pos = encoder.pos_embed
        if pos.shape[1] != target_grid:
            with torch.no_grad():
                pos_4d = pos.permute(0, 3, 1, 2)
                resized = F.interpolate(pos_4d, size=(target_grid, target_grid), mode="bilinear", align_corners=False)
            encoder.pos_embed = torch.nn.Parameter(resized.permute(0, 2, 3, 1))

    # Update prompt encoder spatial expectations
    prompt_encoder.image_embedding_size = (target_grid, target_grid)
    prompt_encoder.input_image_size = (image_size, image_size)
    prompt_encoder.mask_input_size = (4 * target_grid, 4 * target_grid)
