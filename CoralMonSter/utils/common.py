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


def collect_checkpoints(root: Path, dataset_keyword: Optional[str] = None) -> List[Path]:
    """Return sorted checkpoint paths under root, optionally filtered by keyword."""
    checkpoints = [p for p in sorted(Path(root).rglob("*.pth")) if not p.name.endswith("_last.pth")]
    if dataset_keyword:
        keyword = dataset_keyword.lower()
        checkpoints = [p for p in checkpoints if keyword in str(p).lower()]
    return checkpoints


def infer_dataset_from_path(path: Path, dataset_keyword: Optional[str], fallback: str = "hkcoral") -> str:
    """Infer dataset choice from checkpoint path/keyword fallback to provided default."""
    lower_path = str(path).lower()
    keyword_lower = (dataset_keyword or "").lower()
    if "coralscapes" in lower_path or "coralscapes" in keyword_lower:
        return "coralscapes"
    return fallback
