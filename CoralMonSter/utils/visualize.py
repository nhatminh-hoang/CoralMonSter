"""
Visualization helpers for logging CoralMonSter training progress.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch


def save_training_curves(history: Iterable[Dict[str, float]], path: Path) -> None:
    history = list(history)
    if not history:
        return
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_miou = [item["train_miou"] for item in history]
    val_miou = [item["val_miou"] for item in history]
    train_pix = [item.get("train_pix_acc", 0.0) for item in history]
    val_pix = [item.get("val_pix_acc", 0.0) for item in history]

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_miou, label="Train mIoU")
    plt.plot(epochs, val_miou, label="Val mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.legend()
    plt.title("mIoU Curves")

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_pix, label="Train PixAcc")
    plt.plot(epochs, val_pix, label="Val PixAcc")
    plt.xlabel("Epoch")
    plt.ylabel("Pixel Accuracy")
    plt.legend()
    plt.title("Pixel Accuracy")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_segmentation_comparison(
    image: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    palette: Sequence[Tuple[int, int, int]],
    path: Path,
    title: str,
    class_names: Optional[Sequence[str]] = None,
    prompt_coords: Optional[torch.Tensor] = None,
    prompt_labels: Optional[torch.Tensor] = None,
) -> None:
    image_np = _tensor_to_image(image, mean, std)
    pred_color = _colorize_mask(prediction.cpu().numpy(), palette)
    target_color = _colorize_mask(target.cpu().numpy(), palette)

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    ax = plt.subplot(1, 3, 1)
    ax.imshow(image_np)
    ax.set_title(f"Image: {title}")
    ax.axis("off")

    ax = plt.subplot(1, 3, 2)
    ax.imshow(target_color)
    ax.set_title("Ground Truth")
    ax.axis("off")

    ax = plt.subplot(1, 3, 3)
    ax.imshow(pred_color)
    ax.set_title("Prediction")
    ax.axis("off")

    if prompt_coords is not None and prompt_coords.numel() > 0:
        coords = prompt_coords.cpu().numpy()
        labels = (
            prompt_labels.cpu().numpy() if prompt_labels is not None else np.ones(coords.shape[0])
        )
        color_map = {1: "lime", 0: "red", -1: "yellow"}
        colors = [color_map.get(int(lbl), "white") for lbl in labels]
        axes = [plt.subplot(1, 3, i + 1) for i in range(3)]
        for ax in axes:
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=colors,
                marker="o",
                s=25,
                edgecolors="black",
                linewidths=0.5,
            )

    if class_names:
        handles = []
        for idx, name in enumerate(class_names):
            if idx >= len(palette):
                break
            color = tuple(c / 255.0 for c in palette[idx])
            handles.append(Patch(facecolor=color, edgecolor="white", label=name))
        if handles:
            plt.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.1),
                ncol=min(len(handles), 4),
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _tensor_to_image(tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    image = tensor.detach().cpu().clone()
    for c in range(image.shape[0]):
        image[c] = image[c] * std[c] + mean[c]
    image = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def _colorize_mask(mask: np.ndarray, palette: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, rgb in enumerate(palette):
        color[mask == idx] = rgb
    return color
