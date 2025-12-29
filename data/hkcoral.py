"""
PyTorch dataset for the HKCoral benchmark.
Shared prompt utilities now live in ``data/prompt_utils.py``.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2 as T

from .base_dataset import BaseCoralDataset


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


class HKCoralDataset(BaseCoralDataset):
    """Cityscapes-like HKCoral dataset that emits tensors already normalized for SAM."""

    def __init__(
        self,
        root: str,
        split: str,
        image_size: int,
        num_classes: int,
        ignore_label: int = 255,
        prompt_points: int = 8,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        prompt_bins: Tuple[int, ...] = (1, 2, 4, 10),
        compute_prompts: bool = False,  # Set False to defer to GPU
    ) -> None:
        prompt_points = max(prompt_bins) if prompt_bins else prompt_points
        super().__init__(
            image_size=image_size,
            num_classes=num_classes,
            ignore_label=ignore_label,
            prompt_points=prompt_points,
            prompt_bins=prompt_bins,
            compute_prompts=compute_prompts,
        )
        self.root = Path(root)
        self.samples = _list_image_label_pairs(self.root, split)
        self.split = split
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
        img_path, label_path = self.samples[idx]

        image = read_image(str(img_path), mode=ImageReadMode.RGB)
        mask = read_image(str(label_path), mode=ImageReadMode.GRAY).squeeze(0)

        original_size = torch.tensor(mask.shape[-2:], dtype=torch.long)

        mask = T.functional.resize(
            mask.unsqueeze(0),
            size=(self.image_size, self.image_size),
            interpolation=T.InterpolationMode.NEAREST,
        ).squeeze(0).to(torch.int64)

        image = self.image_transform(image)

        return self._finalize_sample(
            image=image,
            mask=mask,
            original_size=original_size,
            file_name=img_path.name,
        )
