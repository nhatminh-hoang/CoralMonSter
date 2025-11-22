"""
Utilities for inference and result formatting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pycocotools.mask as mask_util
import torch
from PIL import Image
from torchvision import transforms


def preprocess_image(path: Path, image_size: int, mean, std) -> Dict[str, torch.Tensor]:
    image = Image.open(path).convert("RGB")
    original_size = torch.tensor((image.height, image.width), dtype=torch.long)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return {
        "image": transform(image),
        "original_size": original_size,
    }


def masks_to_json(
    file_name: str,
    logits: torch.Tensor,
    threshold: float,
    class_names: List[str],
) -> Dict:
    probs = torch.softmax(logits, dim=0)
    height, width = probs.shape[1:]
    annotations = []
    anno_id = 0
    for class_id in range(1, probs.shape[0]):
        mask = probs[class_id] > threshold
        if mask.sum() == 0:
            continue
        mask_np = mask.cpu().numpy().astype(np.uint8)
        rle = mask_util.encode(np.asfortranarray(mask_np))
        rle["counts"] = rle["counts"].decode("utf-8")
        ys, xs = np.where(mask_np == 1)
        bbox = [
            float(xs.min()),
            float(ys.min()),
            float(xs.max() - xs.min() + 1),
            float(ys.max() - ys.min() + 1),
        ]
        annotations.append(
            {
                "id": anno_id,
                "category_id": class_id,
                "category_name": class_names[class_id],
                "segmentation": rle,
                "area": float(mask_np.sum()),
                "bbox": bbox,
                "predicted_iou": float(probs[class_id][mask].mean().item()),
            }
        )
        anno_id += 1

    return {
        "image": {
            "image_id": 0,
            "width": int(width),
            "height": int(height),
            "file_name": file_name,
        },
        "annotations": annotations,
    }
