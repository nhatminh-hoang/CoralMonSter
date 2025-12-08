"""
Data utilities for CoralMonSter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .base_dataset import BaseCoralDataset
from .hkcoral_dataset import HKCoralDataset
from .coralscapes_dataset import CoralScapesDataset
from .prompt_utils import (
    PromptSample,
    sample_prompt,
    sample_prompts_gpu,
    get_class_coordinates_gpu,
)


def hkcoral_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    original_sizes = torch.stack([b["original_size"] for b in batch])
    file_names = [b["file_name"] for b in batch]
    
    # Handle optional prompt fields (may not be present if compute_prompts=False)
    point_coords = [b.get("points") for b in batch]
    point_labels = [b.get("point_labels") for b in batch]
    boxes: List[Optional[torch.Tensor]] = [b.get("box") for b in batch]
    prompt_sets = [b.get("prompt_sets") for b in batch]
    gt_points_list = [b.get("gt_points") for b in batch]

    timing_totals: Dict[str, float] = {}
    for sample in batch:
        sample_timing = sample.get("timings")
        if not sample_timing:
            continue
        for key, value in sample_timing.items():
            timing_totals[key] = timing_totals.get(key, 0.0) + float(value)

    collated = {
        "images": images,
        "masks": masks,
        "original_sizes": original_sizes,
        "file_names": file_names,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "boxes": boxes,
        "prompt_sets": prompt_sets,
    }

    if all(t is not None for t in gt_points_list):
        collated["gt_points"] = torch.stack(gt_points_list)

    if timing_totals:
        collated["timings"] = timing_totals

    return collated


__all__ = [
    "HKCoralDataset",
    "CoralScapesDataset",
    "BaseCoralDataset",
    "PromptSample",
    "sample_prompt",
    "sample_prompts_gpu",
    "get_class_coordinates_gpu",
    "hkcoral_collate_fn",
]
