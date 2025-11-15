"""
Data utilities for CoralMonSter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .hkcoral_dataset import HKCoralDataset, PromptSample, sample_prompt
from .coralscapes_dataset import CoralScapesDataset


def hkcoral_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    original_sizes = torch.stack([b["original_size"] for b in batch])
    file_names = [b["file_name"] for b in batch]
    point_coords = [b["points"] for b in batch]
    point_labels = [b["point_labels"] for b in batch]
    boxes: List[Optional[torch.Tensor]] = [b["box"] for b in batch]
    prompt_sets = [b.get("prompt_sets") for b in batch]

    return {
        "images": images,
        "masks": masks,
        "original_sizes": original_sizes,
        "file_names": file_names,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "boxes": boxes,
        "prompt_sets": prompt_sets,
    }


__all__ = [
    "HKCoralDataset",
    "CoralScapesDataset",
    "PromptSample",
    "sample_prompt",
    "hkcoral_collate_fn",
]
