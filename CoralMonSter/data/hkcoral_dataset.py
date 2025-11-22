"""
PyTorch dataset + prompt sampling utilities for the HKCoral benchmark.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import time

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2 as T


@dataclass
class PromptSample:
    """
    Container holding the sampled positive/negative points and optional box prompt.
    """

    coords: torch.Tensor
    labels: torch.Tensor
    box: Optional[torch.Tensor]


def _list_image_label_pairs(root: Path, split: str) -> List[Tuple[Path, Path]]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing HKCoral images under {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing HKCoral labels under {label_dir}")

    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No HKCoral images found in {image_dir}")

    pairs: List[Tuple[Path, Path]] = []
    for img_path in image_paths:
        label_path = label_dir / f"{img_path.stem}_labelTrainIds.png"
        if not label_path.exists():
            raise FileNotFoundError(f"Expected label {label_path} for image {img_path.name}")
        pairs.append((img_path, label_path))
    return pairs


class HKCoralDataset(Dataset):
    """
    Cityscapes-like HKCoral dataset that emits tensors already normalized for SAM.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        image_size: int,
        num_classes: int,
        ignore_label: int = 255,
        prompt_points: int = 8,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        prompt_bins: Tuple[int, ...] = (1, 2, 4, 10),
    ) -> None:
        super().__init__()
        root = Path(root)
        self.samples = _list_image_label_pairs(root, split)
        self.split = split
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.prompt_points = prompt_points
        self.prompt_bins = prompt_bins
        self.image_size = image_size
        self.image_transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        timings: Dict[str, float] = {}
        img_path, label_path = self.samples[idx]

        start = time.perf_counter()
        image = read_image(str(img_path), mode=ImageReadMode.RGB)
        mask = read_image(str(label_path), mode=ImageReadMode.GRAY).squeeze(0)
        timings["load_io"] = time.perf_counter() - start

        original_size = torch.tensor(mask.shape[-2:], dtype=torch.long)

        start = time.perf_counter()
        mask = T.functional.resize(
            mask.unsqueeze(0),
            size=(self.image_size, self.image_size),
            interpolation=T.InterpolationMode.NEAREST,
        ).squeeze(0).to(torch.int64)
        timings["mask_resize"] = time.perf_counter() - start

        start = time.perf_counter()
        image = self.image_transform(image)
        timings["image_transform"] = time.perf_counter() - start

        start = time.perf_counter()
        start = time.perf_counter()
        class_indices = get_class_coordinates(mask, self.ignore_label)
        prompt = sample_prompt(class_indices, self.prompt_points, self.num_classes, mask.shape)
        prompt_sets = build_prompt_sets(class_indices, self.prompt_bins, self.num_classes, mask.shape)
        timings["prompt_sampling"] = time.perf_counter() - start
        timings["prompt_sampling"] = time.perf_counter() - start
        timings["total"] = sum(timings.values())

        return {
            "image": image,
            "mask": mask,
            "original_size": original_size,
            "file_name": img_path.name,
            "points": prompt.coords,
            "point_labels": prompt.labels,
            "box": prompt.box,
            "prompt_sets": prompt_sets,
            "timings": timings,
        }


def get_class_coordinates(mask: torch.Tensor, ignore_label: int) -> Dict[int, torch.Tensor]:
    """
    Pre-compute flat indices for each class in the mask.
    Returns a dictionary mapping class_id to a tensor of flat indices.
    """
    flat_mask = mask.flatten()
    valid_mask = flat_mask != ignore_label
    valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
    
    if valid_indices.numel() == 0:
        return {}
        
    valid_labels = flat_mask[valid_indices]
    
    # Sort by label to group indices
    sorted_labels, sort_idx = torch.sort(valid_labels)
    sorted_indices = valid_indices[sort_idx]
    
    unique_labels, counts = torch.unique_consecutive(sorted_labels, return_counts=True)
    
    class_indices = {}
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
    shape: Tuple[int, int]
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
            choice = indices
        else:
            idx = torch.randperm(indices.shape[0])[:count]
            choice = indices[idx]
        return choice

    # Foreground classes
    for cls_id in range(1, num_classes):
        if cls_id in class_indices:
            sampled_idx = _sample_indices(class_indices[cls_id], points_per_label)
            if sampled_idx.numel():
                y = torch.div(sampled_idx, W, rounding_mode='floor')
                x = sampled_idx % W
                coords = torch.stack([x, y], dim=1).float()
                coords_all.append(coords)
                labels_all.append(torch.ones(coords.shape[0], dtype=torch.long))

    # Background as negatives (class 0)
    if 0 in class_indices:
        sampled_idx_bg = _sample_indices(class_indices[0], points_per_label)
        if sampled_idx_bg.numel():
            y = torch.div(sampled_idx_bg, W, rounding_mode='floor')
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
        y = torch.div(all_pos_indices, W, rounding_mode='floor')
        x = all_pos_indices % W
        # _mask_to_box expects (y, x) if we follow previous logic, but let's check implementation
        # Previous implementation was: box = _mask_to_box(all_positives) where all_positives was (N, 2) [y, x] from nonzero
        # So we should pass stack([y, x])
        all_pos_coords = torch.stack([y, x], dim=1)
        box = _mask_to_box(all_pos_coords)
    else:
        box = None

    return PromptSample(coords=coords, labels=labels, box=box)


def build_prompt_sets(
    class_indices: Dict[int, torch.Tensor],
    counts: Tuple[int, ...],
    num_classes: int,
    shape: Tuple[int, int]
) -> Dict[int, PromptSample]:
    prompt_sets = {}
    for count in counts:
        prompt_sets[count] = sample_prompt(class_indices, count, num_classes, shape)
    return prompt_sets


def _mask_to_box(coords: torch.Tensor) -> Optional[torch.Tensor]:
    if coords.numel() == 0:
        return None
    y_min, x_min = coords.min(dim=0).values
    y_max, x_max = coords.max(dim=0).values
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
