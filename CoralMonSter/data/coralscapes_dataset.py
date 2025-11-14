"""CoralScapes dataset wrapper built on Hugging Face datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .hkcoral_dataset import PromptSample, build_prompt_sets, sample_prompt


@dataclass
class CoralScapesRecord:
    """Typed container for a CoralScapes example."""

    image: Image.Image
    label: Image.Image
    file_name: str


class CoralScapesDataset(Dataset):
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
        cache_dir: Optional[Path] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.prompt_points = prompt_points
        self.prompt_bins = prompt_bins
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.dataset_id = dataset_id
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.hf_token = hf_token
        self.dataset = load_dataset(
            self.dataset_id,
            split=split,
            cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
            token=self.hf_token,
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.mask_resize = transforms.Resize((image_size, image_size), interpolation=Image.NEAREST)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        label_img: Image.Image = sample["label"]
        file_name = sample.get("file_name") or f"coralscapes_{self.split}_{idx:06d}.png"

        original_size = torch.tensor((label_img.height, label_img.width), dtype=torch.long)
        mask = torch.from_numpy(np.array(self.mask_resize(label_img), dtype=np.int64))
        mask = mask.clamp(min=0, max=self.num_classes - 1)
        image_tensor = self.transform(image)

        prompt = sample_prompt(mask, self.prompt_points, self.num_classes, self.ignore_label)
        prompt_sets = build_prompt_sets(mask, self.prompt_bins, self.num_classes, self.ignore_label)

        return {
            "image": image_tensor,
            "mask": mask,
            "original_size": original_size,
            "file_name": file_name,
            "points": prompt.coords,
            "point_labels": prompt.labels,
            "box": prompt.box,
            "prompt_sets": prompt_sets,
        }


__all__ = ["CoralScapesDataset", "CoralScapesRecord"]
