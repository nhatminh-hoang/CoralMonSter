"""
Visualization utilities for CoralMonSter training and evaluation.

Includes:
  - save_training_curves()        — loss, mIoU, PixAcc during training
  - save_segmentation_comparison() — side-by-side GT vs prediction with palette
  - save_confusion_matrix()        — confusion matrix heatmap
  - colorize_mask()                — apply class palette to segmentation mask

Color palettes for HKCoral and CoralScapes are defined at module level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =====================================================================
# Dataset color palettes and class names
# =====================================================================

# HKCoral — 7 classes (Table 4 in paper)
HKCORAL_CLASS_NAMES = [
    "Background",
    "Massive",
    "Encrusting",
    "Branching",
    "Laminar",
    "Foliaceous",
    "Columnar",
]

HKCORAL_PALETTE = [
    (0,   0,   0),     # Background  — black
    (220, 20,  60),    # Massive     — crimson
    (255, 165,  0),    # Encrusting  — orange
    (50,  205, 50),    # Branching   — lime green
    (30,  144, 255),   # Laminar     — dodger blue
    (186, 85, 211),    # Foliaceous  — medium orchid
    (255, 215,  0),    # Columnar    — gold
]

# CoralScapes — 39 classes (use a generated colormap)
# Full class names from the dataset paper
CORALSCAPES_CLASS_NAMES = [
    "Background", "Acropora_branching", "Acropora_digitifera", "Acropora_tabular",
    "Algae_coralline", "Algae_encrusting", "Algae_macro", "Algae_turf",
    "Coral_bleached", "Coral_dead", "Coral_encrusting", "Coral_foliose",
    "Coral_massive", "Coral_submassive", "Echinopora", "Favia",
    "Favites", "Fish", "Fungia", "Galaxea",
    "Goniastrea", "Goniopora", "Hydnophora", "Leptastrea",
    "Lobophyllia", "Merulina", "Millepora", "Montipora",
    "Pavona", "Platygyra", "Pocillopora", "Porites_branching",
    "Porites_massive", "Sand", "Soft_coral", "Sponge",
    "Stylophora", "Substrate", "Water",
]


def _generate_palette(n: int) -> List[Tuple[int, int, int]]:
    """Generate a visually distinct color palette for n classes."""
    cmap = plt.cm.get_cmap("tab20", max(n, 20))
    palette = []
    for i in range(n):
        rgba = cmap(i % 20)
        palette.append((int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)))
    palette[0] = (0, 0, 0)  # Background is always black
    return palette


CORALSCAPES_PALETTE = _generate_palette(len(CORALSCAPES_CLASS_NAMES))


def get_dataset_info(dataset_name: str) -> Tuple[List[str], List[Tuple[int, int, int]]]:
    """Return (class_names, palette) for a dataset."""
    if dataset_name.lower() in ("hkcoral", "hk_coral"):
        return HKCORAL_CLASS_NAMES, HKCORAL_PALETTE
    elif dataset_name.lower() in ("coralscapes", "coral_scapes"):
        return CORALSCAPES_CLASS_NAMES, CORALSCAPES_PALETTE
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# =====================================================================
# Mask colorization
# =====================================================================

def colorize_mask(
    mask: np.ndarray,
    palette: List[Tuple[int, int, int]],
    ignore_index: int = 255,
) -> np.ndarray:
    """
    Apply a color palette to a segmentation mask.

    Args:
        mask:    (H, W) integer class indices
        palette: list of (R, G, B) tuples, one per class
        ignore_index: pixels with this value → black

    Returns:
        (H, W, 3) uint8 RGB image
    """
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in enumerate(palette):
        color_img[mask == cls_id] = color

    # Ignored pixels → black
    color_img[mask == ignore_index] = (0, 0, 0)
    return color_img


# =====================================================================
# Training curves
# =====================================================================

def save_training_curves(
    history: List[Dict[str, float]],
    save_path: Path,
) -> None:
    """
    Plot training curves: losses, mIoU, and Pixel Accuracy.

    Args:
        history: list of dicts, one per epoch, with keys like:
            epoch, train_loss, val_loss, train_dice, val_dice,
            train_ce, val_ce, train_mask_kd, val_mask_kd,
            train_token_kd, val_token_kd,
            train_miou, val_miou, train_pixacc, val_pixacc
        save_path: output PNG path
    """
    if not history:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("CoralMonSter Training Curves", fontsize=14, fontweight="bold")

    # ── Panel 1: Loss curves ─────────────────────────────────────────
    ax = axes[0]
    loss_series = [
        ("Total Loss",  "train_loss",     "val_loss",     "#1f77b4"),
        ("Dice Loss",   "train_dice",     "val_dice",     "#ff7f0e"),
        ("CE Loss",     "train_ce",       "val_ce",       "#2ca02c"),
        ("Mask KD",     "train_mask_kd",  "val_mask_kd",  "#d62728"),
        ("Token KD",    "train_token_kd", "val_token_kd", "#9467bd"),
    ]

    for label, train_key, val_key, color in loss_series:
        train_vals = [h.get(train_key) for h in history]
        val_vals = [h.get(val_key) for h in history]

        # Only plot if data exists
        if any(v is not None for v in train_vals):
            train_clean = [v if v is not None else float("nan") for v in train_vals]
            ax.plot(epochs, train_clean, "--", color=color, alpha=0.7, label=f"{label} (train)")

        if any(v is not None for v in val_vals):
            val_clean = [v if v is not None else float("nan") for v in val_vals]
            ax.plot(epochs, val_clean, "-", color=color, label=f"{label} (val)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Components")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: mIoU ────────────────────────────────────────────────
    ax = axes[1]
    train_miou = [h.get("train_miou") for h in history]
    val_miou = [h.get("val_miou") for h in history]

    if any(v is not None for v in train_miou):
        clean = [v if v is not None else float("nan") for v in train_miou]
        ax.plot(epochs, clean, "--", color="#1f77b4", label="Train mIoU")
    if any(v is not None for v in val_miou):
        clean = [v if v is not None else float("nan") for v in val_miou]
        ax.plot(epochs, clean, "-", color="#1f77b4", label="Val mIoU")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    ax.set_title("Mean IoU")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Pixel Accuracy ──────────────────────────────────────
    ax = axes[2]
    train_pa = [h.get("train_pixacc") for h in history]
    val_pa = [h.get("val_pixacc") for h in history]

    if any(v is not None for v in train_pa):
        clean = [v if v is not None else float("nan") for v in train_pa]
        ax.plot(epochs, clean, "--", color="#2ca02c", label="Train PixAcc")
    if any(v is not None for v in val_pa):
        clean = [v if v is not None else float("nan") for v in val_pa]
        ax.plot(epochs, clean, "-", color="#2ca02c", label="Val PixAcc")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pixel Accuracy")
    ax.set_title("Pixel Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Segmentation comparison
# =====================================================================

def save_segmentation_comparison(
    image: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    palette: List[Tuple[int, int, int]],
    class_names: List[str],
    save_path: Path,
    teacher_pred: Optional[np.ndarray] = None,
    ignore_index: int = 255,
) -> None:
    """
    Save side-by-side comparison: Image | Ground Truth | Student | [Teacher].

    Args:
        image:        (H, W, 3) RGB image (uint8, 0-255)
        prediction:   (H, W) integer class indices (student)
        target:       (H, W) integer ground-truth labels
        palette:      color palette for classes
        class_names:  list of class name strings
        save_path:    output PNG path
        teacher_pred: (H, W) optional teacher prediction
        ignore_index: label to ignore
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_panels = 4 if teacher_pred is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    # Panel 1: Input image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Panel 2: Ground truth
    gt_colored = colorize_mask(target, palette, ignore_index)
    axes[1].imshow(gt_colored)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Panel 3: Student prediction
    pred_colored = colorize_mask(prediction, palette, ignore_index)
    axes[2].imshow(pred_colored)
    axes[2].set_title("Student Prediction")
    axes[2].axis("off")

    # Panel 4: Teacher prediction (optional)
    if teacher_pred is not None:
        teacher_colored = colorize_mask(teacher_pred, palette, ignore_index)
        axes[3].imshow(teacher_colored)
        axes[3].set_title("Teacher Prediction")
        axes[3].axis("off")

    # ── Legend ────────────────────────────────────────────────────────
    # Collect unique classes present in GT or prediction
    present_classes = set(np.unique(target)) | set(np.unique(prediction))
    if teacher_pred is not None:
        present_classes |= set(np.unique(teacher_pred))
    present_classes.discard(ignore_index)

    patches = []
    for cls_id in sorted(present_classes):
        if cls_id < len(class_names) and cls_id < len(palette):
            color = np.array(palette[cls_id]) / 255.0
            patches.append(
                mpatches.Patch(color=color, label=class_names[cls_id])
            )

    if patches:
        fig.legend(
            handles=patches,
            loc="lower center",
            ncol=min(len(patches), 7),
            fontsize=8,
            bbox_to_anchor=(0.5, -0.02),
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Confusion matrix
# =====================================================================

def save_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """
    Save a confusion matrix heatmap.

    Args:
        confusion_matrix: (Ncls, Ncls) integer array
        class_names:      list of class name strings
        save_path:        output PNG path
        title:            plot title
        normalize:        if True, normalize rows to percentages
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm = cm / row_sums * 100

    n = len(class_names)
    fig_size = max(8, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Add text annotations for small matrices
    if n <= 15:
        fmt = ".1f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                ax.text(
                    j, i, f"{val:{fmt}}",
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=7,
                )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def denormalize_image(
    image_tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """
    Convert a normalized image tensor back to uint8 RGB numpy array.

    Args:
        image_tensor: (C, H, W) float tensor, normalized with mean/std
        mean:         channel means used for normalization
        std:          channel stds used for normalization

    Returns:
        (H, W, 3) uint8 numpy array
    """
    img = image_tensor.cpu().float()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]

    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img
