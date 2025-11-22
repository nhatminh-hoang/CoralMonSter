from __future__ import annotations

from pathlib import Path
from typing import Optional

from CoralMonSter.segment_anything import sam_model_registry

def build_sam_backbone(
    model_type: str,
    checkpoint_path: Path,
    use_gradient_checkpointing: bool = False,
    use_flash_attention: bool = False,
):
    """
    Builds the SAM backbone model.
    """
    if checkpoint_path.exists():
        checkpoint_arg = str(checkpoint_path)
    else:
        print(f"[Warning] SAM checkpoint '{checkpoint_path}' not found. Initializing from scratch.")
        checkpoint_arg = None
    
    try:
        return sam_model_registry[model_type](
            checkpoint=checkpoint_arg,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_flash_attention=use_flash_attention,
        )
    except RuntimeError as exc:
        if checkpoint_arg and "state_dict" in str(exc):
            print(
                "[Warning] Failed to load checkpoint into SAM backbone due to mismatched keys. "
                "Falling back to random initialization. Make sure '--model_type' matches the checkpoint."
            )
            return sam_model_registry[model_type](
                checkpoint=None,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_flash_attention=use_flash_attention,
            )
        raise
