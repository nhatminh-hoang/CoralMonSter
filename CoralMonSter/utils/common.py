from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, List
import argparse

def infer_model_type_from_checkpoint(path: Path) -> Optional[str]:
    """Infer model type (vit_h, vit_l, vit_b) from checkpoint filename."""
    name = Path(path).name.lower()
    for candidate in ("vit_h", "vit_l", "vit_b"):
        if candidate in name:
            return candidate
    return None

def parse_prompt_bins(value: str) -> Tuple[int, ...]:
    """Parse comma-separated prompt bins string into a tuple of integers."""
    bins = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            count = int(chunk)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Prompt bin '{chunk}' is not a valid integer."
            ) from exc
        if count <= 0:
            raise argparse.ArgumentTypeError(
                f"Prompt bin '{chunk}' must be a positive integer."
            )
        bins.append(count)
    if not bins:
        raise argparse.ArgumentTypeError(
            "--prompt_bins requires at least one positive integer."
        )
    return tuple(bins)
