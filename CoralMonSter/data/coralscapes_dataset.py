"""CoralScapes dataset wrapper built on Hugging Face datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import time

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as F

from .hkcoral_dataset import PromptSample, build_prompt_sets, sample_prompt, get_class_coordinates


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
        compute_prompts: bool = False,  # Set False to defer to GPU
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
        self.compute_prompts = compute_prompts
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
        timings: Dict[str, float] = {}
        sample = self.dataset[idx]

        start = time.perf_counter()
        image = sample["image"].convert("RGB")
        label_img: Image.Image = sample["label"]
        timings["load_io"] = time.perf_counter() - start
        file_name = sample.get("file_name") or f"coralscapes_{self.split}_{idx:06d}.png"

        original_size = torch.tensor((label_img.height, label_img.width), dtype=torch.long)
        start = time.perf_counter()
        mask = F.resize(
            F.pil_to_tensor(label_img),
            size=(self.image_size, self.image_size),
            interpolation=T.InterpolationMode.NEAREST,
        )
        mask = mask.squeeze(0).to(torch.int64).clamp(min=0, max=self.num_classes - 1)
        timings["mask_resize"] = time.perf_counter() - start

        start = time.perf_counter()
        image_tensor = self.image_transform(image)
        timings["image_transform"] = time.perf_counter() - start

        result = {
            "image": image_tensor,
            "mask": mask,
            "original_size": original_size,
            "file_name": file_name,
            "timings": timings,
        }

        # Only compute prompts on CPU if explicitly requested
        if self.compute_prompts:
            start = time.perf_counter()
            class_indices = get_class_coordinates(mask, self.ignore_label)
            prompt = sample_prompt(class_indices, self.prompt_points, self.num_classes, mask.shape)
            prompt_sets = build_prompt_sets(class_indices, self.prompt_bins, self.num_classes, mask.shape)
            timings["prompt_sampling"] = time.perf_counter() - start
            result.update({
                "points": prompt.coords,
                "point_labels": prompt.labels,
                "box": prompt.box,
                "prompt_sets": prompt_sets,
            })

        timings["total"] = sum(timings.values())
        return result


__all__ = ["CoralScapesDataset", "CoralScapesRecord"]
