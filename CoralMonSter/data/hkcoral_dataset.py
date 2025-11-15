"""
PyTorch dataset + prompt sampling utilities for the HKCoral benchmark.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.mask_resize = transforms.Resize((image_size, image_size), interpolation=Image.NEAREST)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path)
        original_size = torch.tensor(mask.size[::-1], dtype=torch.long)  # (H, W)

        mask = torch.from_numpy(np.array(self.mask_resize(mask), dtype=np.int64))
        image = self.transform(image)

        prompt = sample_prompt(mask, self.prompt_points, self.num_classes, self.ignore_label)
        prompt_sets = build_prompt_sets(mask, self.prompt_bins, self.num_classes, self.ignore_label)

        return {
            "image": image,
            "mask": mask,
            "original_size": original_size,
            "file_name": img_path.name,
            "points": prompt.coords,
            "point_labels": prompt.labels,
            "box": prompt.box,
            "prompt_sets": prompt_sets,
        }


def sample_prompt(mask: torch.Tensor, points_per_label: int, num_classes: int, ignore_label: int) -> PromptSample:
    """
    Sample an equal number of points per semantic label (including background).
    """

    coords_all = []
    labels_all = []

    def _sample_coords(coords: torch.Tensor, count: int) -> torch.Tensor:
        if coords.numel() == 0 or count == 0:
            return torch.zeros((0, 2), dtype=torch.float32)
        if coords.shape[0] <= count:
            choice = coords
        else:
            idx = torch.randperm(coords.shape[0])[:count]
            choice = coords[idx]
        return choice[:, [1, 0]].float()  # switch to (x, y)

    valid_mask = mask != ignore_label
    # Foreground classes
    for cls_id in range(1, num_classes):
        cls_coords = torch.nonzero((mask == cls_id) & valid_mask, as_tuple=False)
        sampled = _sample_coords(cls_coords, points_per_label)
        if sampled.numel():
            coords_all.append(sampled)
            labels_all.append(torch.ones(sampled.shape[0], dtype=torch.long))

    # Background as negatives
    background_coords = torch.nonzero((mask == 0) & valid_mask, as_tuple=False)
    sampled_bg = _sample_coords(background_coords, points_per_label)
    if sampled_bg.numel():
        coords_all.append(sampled_bg)
        labels_all.append(torch.zeros(sampled_bg.shape[0], dtype=torch.long))

    if coords_all:
        coords = torch.cat(coords_all, dim=0)
        labels = torch.cat(labels_all, dim=0)
    else:
        coords = torch.zeros((0, 2), dtype=torch.float32)
        labels = torch.zeros(0, dtype=torch.long)

    positives = torch.nonzero((mask > 0) & valid_mask, as_tuple=False)
    box = _mask_to_box(positives)

    return PromptSample(coords=coords, labels=labels, box=box)


def build_prompt_sets(
    mask: torch.Tensor,
    counts: Tuple[int, ...],
    num_classes: int,
    ignore_label: int,
) -> Dict[int, PromptSample]:
    prompt_sets = {}
    for count in counts:
        prompt_sets[count] = sample_prompt(mask, count, num_classes, ignore_label)
    return prompt_sets


def _mask_to_box(coords: torch.Tensor) -> Optional[torch.Tensor]:
    if coords.numel() == 0:
        return None
    y_min, x_min = coords.min(dim=0).values
    y_max, x_max = coords.max(dim=0).values
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
