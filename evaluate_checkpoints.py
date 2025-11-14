#!/usr/bin/env python3
"""
Utility script to evaluate every checkpoint under a root directory and dump metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from CoralMonSter import CoralMonSter as CoralModel
from CoralMonSter import CoralTrainer, HKCoralConfig
from CoralMonSter.data import HKCoralDataset, hkcoral_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in a directory.")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints", help="Root directory to scan for .pth files.")
    parser.add_argument("--dataset_root", type=str, default="datasets/HKCoral", help="HKCoral dataset root.")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Base SAM checkpoint path.")
    parser.add_argument("--model_type", type=str, default=None, choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=str, default="checkpoint_evaluations.json", help="Where to dump aggregated results.")
    parser.add_argument("--log_root", type=str, default="logs", help="Base log directory (used to satisfy config paths).")
    parser.add_argument("--fig_root", type=str, default="confusion_figs", help="Directory for confusion matrix plots.")
    return parser.parse_args()


def infer_model_type_from_checkpoint(path: Path) -> Optional[str]:
    name = path.name.lower()
    for candidate in ("vit_h", "vit_l", "vit_b"):
        if candidate in name:
            return candidate
    return None


def prepare_dataloader(cfg: HKCoralConfig, batch_size: int, num_workers: int) -> DataLoader:
    dataset = HKCoralDataset(
        cfg.dataset_root,
        cfg.split,
        cfg.image_size,
        cfg.num_classes,
        cfg.ignore_label,
        prompt_points=cfg.distillation.prompt_points,
        prompt_bins=cfg.prompt_bins,
        mean=cfg.image_mean,
        std=cfg.image_std,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=hkcoral_collate_fn)


def evaluate_checkpoint(
    path: Path,
    device: torch.device,
    loader: DataLoader,
    cfg: HKCoralConfig,
    visualize_dir: Optional[Path] = None,
) -> dict:
    model = CoralModel(cfg).to(device)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)
    trainer = CoralTrainer(model, cfg)
    metrics = trainer.evaluate(loader, device, visualize_dir=visualize_dir)
    return {
        "checkpoint": str(path),
        "scenario": cfg.scenario_name,
        "split": cfg.split,
        "loss": metrics["loss"],
        "miou": metrics["miou"],
        "pixel_accuracy": metrics["pix_acc"],
        "class_miou": metrics["class_iou"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def save_confusion_figure(confusion: List[List[float]], class_names: List[str], path: Path) -> None:
    matrix = np.array(confusion)
    if matrix.size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm_matrix = matrix / row_sums
    plt.figure(figsize=(6, 5))
    im = plt.imshow(norm_matrix, interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.title("Normalized Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    for i in range(norm_matrix.shape[0]):
        for j in range(norm_matrix.shape[1]):
            value = norm_matrix[i, j]
            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def main() -> None:
    args = parse_args()
    checkpoint_paths = [
        p for p in sorted(Path(args.checkpoint_root).rglob("*.pth")) if not p.name.endswith("_last.pth")
    ]
    if not checkpoint_paths:
        print(f"No eligible checkpoints found under '{args.checkpoint_root}'.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)

    results: List[dict] = []
    for ckpt_path in checkpoint_paths:
        scenario_name = ckpt_path.parent.name
        inferred = infer_model_type_from_checkpoint(ckpt_path)
        model_type = args.model_type or inferred or "vit_h"
        if args.model_type is None and inferred is None:
            print(f"[Info] Could not infer model type from '{ckpt_path.name}', defaulting to 'vit_h'.")
        elif args.model_type is None and inferred is not None:
            print(f"[Info] Detected backbone '{inferred}' from '{ckpt_path.name}'.")
        cfg = HKCoralConfig(
            dataset_root=Path(args.dataset_root),
            split=args.split,
            image_size=args.image_size,
            model_type=model_type,
            sam_checkpoint=Path(args.sam_checkpoint),
            scenario_name=scenario_name,
        )
        cfg.checkpoint_root = Path(args.checkpoint_root)
        cfg.log_root = Path(args.log_root)
        cfg.resolve_paths()
        loader = prepare_dataloader(cfg, args.batch_size, args.num_workers)
        print(f"Evaluating {ckpt_path} on split='{args.split}' ...")
        result = evaluate_checkpoint(ckpt_path, device, loader, cfg, visualize_dir=Path(args.fig_root) / scenario_name)
        fig_path = Path(args.fig_root) / scenario_name / f"{ckpt_path.stem}_{args.split}_confusion.png"
        save_confusion_figure(result["confusion_matrix"], cfg.class_names, fig_path)
        results.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"Wrote evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
