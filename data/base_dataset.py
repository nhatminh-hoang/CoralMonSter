"""
Lightweight base dataset to centralize shared preprocessing and prompt construction.
"""

from __future__ import annotations

import random
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset

from .prompt_utils import (
    build_gt_points_from_mask,
    build_prompt_sets,
    get_class_coordinates,
    sample_prompt,
)


class BaseCoralDataset(Dataset):
    """Shared helpers for CoralMonSter datasets (HKCoral, CoralScapes, ...)."""

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        ignore_label: int,
        prompt_points: int,
        prompt_bins: Tuple[int, ...],
        compute_prompts: bool,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.prompt_points = prompt_points
        self.prompt_bins = prompt_bins
        self.compute_prompts = compute_prompts

    def _finalize_sample(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        original_size: torch.Tensor,
        file_name: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Assemble the common output dict, including gt_points and optional CPU prompts.
        """
        result: Dict[str, torch.Tensor] = {
            "image": image,
            "mask": mask,
            "original_size": original_size,
            "file_name": file_name,
        }

        points_to_use = random.choice(self.prompt_bins) if self.prompt_bins else self.prompt_points
        gt_points = build_gt_points_from_mask(mask, self.num_classes, self.prompt_points)
        if points_to_use < self.prompt_points:
            for cls_idx in range(gt_points.shape[0]):
                cls_points = gt_points[cls_idx, :points_to_use]
                valid = cls_points[cls_points[:, 0] >= 0]
                if valid.numel() == 0:
                    continue
                tile = valid.repeat(int((self.prompt_points + valid.shape[0] - 1) // valid.shape[0]), 1)
                gt_points[cls_idx] = tile[: self.prompt_points]
        result["gt_points"] = gt_points

        if self.compute_prompts:
            class_indices = get_class_coordinates(mask, self.ignore_label)
            prompt = sample_prompt(class_indices, self.prompt_points, self.num_classes, mask.shape)
            prompt_sets = build_prompt_sets(class_indices, self.prompt_bins, self.num_classes, mask.shape)
            result.update({
                "points": prompt.coords,
                "point_labels": prompt.labels,
                "box": prompt.box,
                "prompt_sets": prompt_sets,
            })

        return result
