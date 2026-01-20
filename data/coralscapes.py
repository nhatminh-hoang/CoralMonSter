"""CoralScapes dataset wrapper built on Hugging Face datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as F

from .base_dataset import BaseCoralDataset


@dataclass
class CoralScapesRecord:
    """Typed container for a CoralScapes example."""

    image: Image.Image
    label: Image.Image
    file_name: str


class CoralScapesDataset(BaseCoralDataset):
    """Dataset that streams CoralScapes splits via the Hugging Face hub."""

    def __init__(
        self,
        split: str,
        image_size: int,
        num_classes: int,
        ignore_label: int = 255,
        prompt_points: int = 8,
        prompt_bins: Tuple[int, ...] = (1, 2, 4, 10),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        dataset_id: str = "EPFL-ECEO/coralscapes",
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
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
        self.split = split
        self.dataset_id = dataset_id
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.hf_token = hf_token
        self.dataset = load_dataset(
            self.dataset_id,
            split=split,
            cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
            token=self.hf_token,
        )
        self.image_size = image_size
        self.image_transform = T.Compose(
            [
                T.ToImage(),
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]

        image = sample["image"].convert("RGB")
        label_img: Image.Image = sample["label"]
        file_name = sample.get("file_name") or f"coralscapes_{self.split}_{idx:06d}.png"

        original_size = torch.tensor((label_img.height, label_img.width), dtype=torch.long)
        mask = F.resize(
            F.pil_to_tensor(label_img),
            size=(self.image_size, self.image_size),
            interpolation=T.InterpolationMode.NEAREST,
        )
        mask = mask.squeeze(0).to(torch.int64).clamp(min=0, max=self.num_classes - 1)

        image_tensor = self.image_transform(image)

        return self._finalize_sample(
            image=image_tensor,
            mask=mask,
            original_size=original_size,
            file_name=file_name,
        )
