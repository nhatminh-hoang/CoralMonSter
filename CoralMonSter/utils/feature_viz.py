"""Feature visualization helpers (PCA overlays) for CoralMonSter."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _denormalize_image(image: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    img = image.detach().cpu().clone().float()
    for idx, (m, s) in enumerate(zip(mean, std)):
        img[idx] = img[idx] * s + m
    img = img.clamp(0.0, 1.0)
    img = img.permute(1, 2, 0).numpy()
    return (img * 255.0).astype(np.uint8)


def _normalize_component(comp: torch.Tensor) -> torch.Tensor:
    comp_min = comp.amin(dim=(0, 1), keepdim=True)
    comp_max = comp.amax(dim=(0, 1), keepdim=True)
    denom = (comp_max - comp_min).clamp(min=1e-6)
    return (comp - comp_min) / denom


def _compute_pca_grid(embedding: torch.Tensor, components: int = 3) -> torch.Tensor:
    emb = embedding.detach().to("cpu")
    c, h, w = emb.shape
    tokens = emb.reshape(c, -1).permute(1, 0).float()  # [hw, c]
    tokens = tokens - tokens.mean(dim=0, keepdim=True)
    if tokens.shape[0] == 0 or tokens.shape[1] == 0:
        return torch.zeros((h, w, components), dtype=torch.float32)
    q = min(components, tokens.shape[0], tokens.shape[1])
    if q == 0:
        return torch.zeros((h, w, components), dtype=torch.float32)
    u, s, v = torch.pca_lowrank(tokens, q=q)
    proj = tokens @ v[:, :q]
    grid = proj.reshape(h, w, q)
    grid = _normalize_component(grid)
    if q < components:
        pad = torch.zeros((h, w, components - q), dtype=grid.dtype, device=grid.device)
        grid = torch.cat([grid, pad], dim=-1)
    return grid


def save_pca_overlay(
    image: torch.Tensor,
    embedding: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    path: Path,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_img = _denormalize_image(image, mean, std)
    pca_grid = _compute_pca_grid(embedding)
    pca_tensor = pca_grid.permute(2, 0, 1).unsqueeze(0)
    upsampled = F.interpolate(
        pca_tensor,
        size=base_img.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    overlay = (base_img.astype(np.float32) / 255.0) * 0.6 + upsampled * 0.4
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(base_img)
    axes[0].set_title("Image")
    axes[0].axis("off")
    axes[1].imshow(upsampled)
    axes[1].set_title("PCA Heatmap")
    axes[1].axis("off")
    axes[2].imshow(np.clip(overlay, 0.0, 1.0))
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)
