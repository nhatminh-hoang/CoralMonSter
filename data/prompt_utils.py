"""
Dataset-agnostic prompt sampling utilities shared across CoralMonSter datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class PromptSample:
    """Container holding the sampled positive/negative points and optional box prompt."""

    coords: torch.Tensor
    labels: torch.Tensor
    box: Optional[torch.Tensor]


def build_gt_points_from_mask(
    mask: torch.Tensor,
    num_classes: int,
    points_per_class: int,
) -> torch.Tensor:
    """
    Build a dense gt_points tensor shaped (num_classes, points_per_class, 2).

    Uses -1 sentinels for classes with insufficient points; coordinates are (x, y).
    """

    h, w = mask.shape[-2:]
    device = mask.device
    gt_points = torch.full(
        (num_classes, points_per_class, 2),
        fill_value=-1.0,
        device=device,
        dtype=torch.float32,
    )

    flat = mask.flatten()
    for cls in range(num_classes):
        cls_idx = (flat == cls).nonzero(as_tuple=True)[0]
        if cls_idx.numel() == 0:
            continue
        if cls_idx.numel() > points_per_class:
            perm = torch.randperm(cls_idx.numel(), device=device)[:points_per_class]
            cls_idx = cls_idx[perm]
        y = torch.div(cls_idx, w, rounding_mode="floor")
        x = cls_idx % w
        coords = torch.stack([x, y], dim=1).to(torch.float32)
        gt_points[cls, : coords.shape[0]] = coords

    return gt_points


def get_class_coordinates(mask: torch.Tensor, ignore_label: int) -> Dict[int, torch.Tensor]:
    """
    Pre-compute flat indices for each class in the mask (CPU path).
    Returns a dictionary mapping class_id to a tensor of flat indices.
    """
    flat_mask = mask.flatten()
    valid_mask = flat_mask != ignore_label
    valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

    if valid_indices.numel() == 0:
        return {}

    valid_labels = flat_mask[valid_indices]
    sorted_labels, sort_idx = torch.sort(valid_labels)
    sorted_indices = valid_indices[sort_idx]

    unique_labels, counts = torch.unique_consecutive(sorted_labels, return_counts=True)

    class_indices: Dict[int, torch.Tensor] = {}
    start_idx = 0
    for i, label in enumerate(unique_labels):
        count = counts[i].item()
        class_indices[label.item()] = sorted_indices[start_idx : start_idx + count]
        start_idx += count

    return class_indices


def sample_prompt(
    class_indices: Dict[int, torch.Tensor],
    points_per_label: int,
    num_classes: int,
    shape: Tuple[int, int],
) -> PromptSample:
    """
    Sample an equal number of points per semantic label (including background).
    """
    H, W = shape
    coords_all = []
    labels_all = []

    def _sample_indices(indices: torch.Tensor, count: int) -> torch.Tensor:
        if indices.numel() == 0 or count == 0:
            return torch.zeros(0, dtype=torch.long)
        if indices.shape[0] <= count:
            return indices
        idx = torch.randperm(indices.shape[0])[:count]
        return indices[idx]

    # Foreground classes
    for cls_id in range(1, num_classes):
        if cls_id in class_indices:
            sampled_idx = _sample_indices(class_indices[cls_id], points_per_label)
            if sampled_idx.numel():
                y = torch.div(sampled_idx, W, rounding_mode="floor")
                x = sampled_idx % W
                coords = torch.stack([x, y], dim=1).float()
                coords_all.append(coords)
                labels_all.append(torch.ones(coords.shape[0], dtype=torch.long))

    # Background as negatives (class 0)
    if 0 in class_indices:
        sampled_idx_bg = _sample_indices(class_indices[0], points_per_label)
        if sampled_idx_bg.numel():
            y = torch.div(sampled_idx_bg, W, rounding_mode="floor")
            x = sampled_idx_bg % W
            coords = torch.stack([x, y], dim=1).float()
            coords_all.append(coords)
            labels_all.append(torch.zeros(coords.shape[0], dtype=torch.long))

    if coords_all:
        coords = torch.cat(coords_all, dim=0)
        labels = torch.cat(labels_all, dim=0)
    else:
        coords = torch.zeros((0, 2), dtype=torch.float32)
        labels = torch.zeros(0, dtype=torch.long)

    # Calculate box from all positive points (foreground)
    pos_indices_list = [class_indices[c] for c in class_indices if c > 0]
    if pos_indices_list:
        all_pos_indices = torch.cat(pos_indices_list, dim=0)
        y = torch.div(all_pos_indices, W, rounding_mode="floor")
        x = all_pos_indices % W
        all_pos_coords = torch.stack([y, x], dim=1)
        box = _mask_to_box(all_pos_coords)
    else:
        box = None

    return PromptSample(coords=coords, labels=labels, box=box)


def build_prompt_sets(
    class_indices: Dict[int, torch.Tensor],
    counts: Tuple[int, ...],
    num_classes: int,
    shape: Tuple[int, int],
) -> Dict[int, PromptSample]:
    prompt_sets: Dict[int, PromptSample] = {}
    for count in counts:
        prompt_sets[count] = sample_prompt(class_indices, count, num_classes, shape)
    return prompt_sets


def _mask_to_box(coords: torch.Tensor) -> Optional[torch.Tensor]:
    if coords.numel() == 0:
        return None
    y_min, x_min = coords.min(dim=0).values
    y_max, x_max = coords.max(dim=0).values
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32, device=coords.device)


@torch.no_grad()
def get_class_coordinates_gpu(mask: torch.Tensor, ignore_label: int) -> Dict[int, torch.Tensor]:
    """
    GPU-optimized version of get_class_coordinates.
    Returns dictionary mapping class_id to flat indices tensor (on same device as mask).
    """
    flat_mask = mask.flatten()
    valid_mask = flat_mask != ignore_label
    valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

    if valid_indices.numel() == 0:
        return {}

    valid_labels = flat_mask[valid_indices]
    sorted_labels, sort_idx = torch.sort(valid_labels)
    sorted_indices = valid_indices[sort_idx]
    unique_labels, counts = torch.unique_consecutive(sorted_labels, return_counts=True)

    class_indices: Dict[int, torch.Tensor] = {}
    start_idx = 0
    for i in range(unique_labels.shape[0]):
        label = unique_labels[i].item()
        count = counts[i].item()
        class_indices[label] = sorted_indices[start_idx : start_idx + count]
        start_idx += count

    return class_indices


@torch.no_grad()
def sample_prompt_gpu(
    class_indices: Dict[int, torch.Tensor],
    points_per_label: int,
    num_classes: int,
    shape: Tuple[int, int],
    device: torch.device,
) -> PromptSample:
    """GPU-optimized prompt sampling."""
    H, W = shape
    coords_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    # Foreground classes
    for cls_id in range(1, num_classes):
        if cls_id in class_indices:
            indices = class_indices[cls_id]
            if indices.numel() > 0:
                if indices.shape[0] <= points_per_label:
                    sampled_idx = indices
                else:
                    perm = torch.randperm(indices.shape[0], device=device)[:points_per_label]
                    sampled_idx = indices[perm]

                y = torch.div(sampled_idx, W, rounding_mode="floor")
                x = sampled_idx % W
                coords = torch.stack([x, y], dim=1).float()
                coords_all.append(coords)
                labels_all.append(torch.ones(coords.shape[0], dtype=torch.long, device=device))

    # Background as negatives (class 0)
    if 0 in class_indices:
        indices = class_indices[0]
        if indices.numel() > 0:
            if indices.shape[0] <= points_per_label:
                sampled_idx = indices
            else:
                perm = torch.randperm(indices.shape[0], device=device)[:points_per_label]
                sampled_idx = indices[perm]

            y = torch.div(sampled_idx, W, rounding_mode="floor")
            x = sampled_idx % W
            coords = torch.stack([x, y], dim=1).float()
            coords_all.append(coords)
            labels_all.append(torch.zeros(coords.shape[0], dtype=torch.long, device=device))

    if coords_all:
        coords = torch.cat(coords_all, dim=0)
        labels = torch.cat(labels_all, dim=0)
    else:
        coords = torch.zeros((0, 2), dtype=torch.float32, device=device)
        labels = torch.zeros(0, dtype=torch.long, device=device)

    # Calculate box from all positive points
    pos_indices_list = [class_indices[c] for c in class_indices if c > 0]
    if pos_indices_list:
        all_pos_indices = torch.cat(pos_indices_list, dim=0)
        y = torch.div(all_pos_indices, W, rounding_mode="floor")
        x = all_pos_indices % W
        all_pos_coords = torch.stack([y, x], dim=1)
        box = _mask_to_box(all_pos_coords)
    else:
        box = None

    return PromptSample(coords=coords, labels=labels, box=box)


@torch.no_grad()
def sample_prompts_gpu(
    masks: torch.Tensor,
    num_classes: int,
    ignore_label: int,
    prompt_bins: Tuple[int, ...],
) -> List[Dict[int, PromptSample]]:
    """
    GPU-accelerated prompt sampling for a batch of masks.

    Args:
        masks: (B, H, W) tensor of class labels on GPU
        num_classes: number of semantic classes
        ignore_label: label to ignore
        prompt_bins: tuple of point counts to sample

    Returns:
        List of prompt_sets dictionaries, one per batch item
    """
    batch_size, H, W = masks.shape
    results: List[Dict[int, PromptSample]] = []

    for b in range(batch_size):
        mask = masks[b]
        class_indices = get_class_coordinates_gpu(mask, ignore_label)
        prompt_sets = {}
        for count in prompt_bins:
            prompt_sets[count] = sample_prompt_gpu(class_indices, count, num_classes, (H, W), mask.device)
        results.append(prompt_sets)

    return results


__all__ = [
    "PromptSample",
    "build_gt_points_from_mask",
    "get_class_coordinates",
    "sample_prompt",
    "build_prompt_sets",
    "sample_prompts_gpu",
    "get_class_coordinates_gpu",
    "sample_prompt_gpu",
]
