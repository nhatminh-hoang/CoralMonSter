#!/usr/bin/env python3
"""
Utility script to evaluate every checkpoint under a root directory and dump metrics.
"""

from __future__ import annotations

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from CoralMonSter import CoralMonSter as CoralModel
from CoralMonSter import CoralTrainer, HKCoralConfig, CoralScapesConfig
from CoralMonSter.data import CoralScapesDataset, HKCoralDataset, hkcoral_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in a directory.")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints", help="Root directory to scan for .pth files.")
    parser.add_argument("--dataset_root", type=str, default="datasets/HKCoral", help="Dataset root (HKCoral folder or CoralScapes cache).")
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
    parser.add_argument("--dataset", type=str, default=None, choices=["hkcoral", "coralscapes"], help="Explicit dataset override (otherwise inferred from checkpoint path/keyword).")
    parser.add_argument("--dataset_cache_dir", type=str, default=None, help="Cache directory for CoralScapes downloads.")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"), help="Optional Hugging Face token for CoralScapes.")
    parser.add_argument(
        "--dataset_keyword",
        type=str,
        default=None,
        help="Only evaluate checkpoints whose path contains this (case-insensitive) keyword."
    )
    return parser.parse_args()


def infer_model_type_from_checkpoint(path: Path) -> Optional[str]:
    name = path.name.lower()
    for candidate in ("vit_h", "vit_l", "vit_b"):
        if candidate in name:
            return candidate
    return None


ConfigType = Union[HKCoralConfig, CoralScapesConfig]


def determine_dataset(
    ckpt_path: Path,
    dataset_arg: Optional[str],
    keyword: Optional[str],
) -> str:
    if dataset_arg:
        return dataset_arg.lower()
    lower_path = str(ckpt_path).lower()
    keyword_lower = keyword.lower() if keyword else ""
    if "coralscapes" in lower_path or "coralscapes" in keyword_lower:
        return "coralscapes"
    return "hkcoral"


def build_config(
    args: argparse.Namespace,
    scenario_name: str,
    model_type: str,
    dataset_choice: str,
) -> ConfigType:
    base_kwargs = dict(
        dataset_root=Path(args.dataset_root),
        split=args.split,
        image_size=args.image_size,
        model_type=model_type,
        sam_checkpoint=Path(args.sam_checkpoint),
        scenario_name=scenario_name,
    )
    if dataset_choice == "coralscapes":
        cfg: ConfigType = CoralScapesConfig(**base_kwargs)
        if args.dataset_cache_dir:
            cfg.dataset_cache_dir = Path(args.dataset_cache_dir)
        if args.hf_token:
            cfg.hf_token = args.hf_token
    else:
        cfg = HKCoralConfig(**base_kwargs)
    cfg.checkpoint_root = Path(args.checkpoint_root)
    cfg.log_root = Path(args.log_root)
    cfg.resolve_paths()
    return cfg


def prepare_dataloader(cfg: ConfigType, dataset_choice: str, batch_size: int, num_workers: int) -> DataLoader:
    if dataset_choice == "coralscapes":
        hf_split = {"val": "validation"}.get(cfg.split, cfg.split)
        dataset = CoralScapesDataset(
            split=hf_split,
            image_size=cfg.image_size,
            num_classes=cfg.num_classes,
            ignore_label=cfg.ignore_label,
            prompt_points=cfg.distillation.prompt_points,
            prompt_bins=cfg.prompt_bins,
            mean=cfg.image_mean,
            std=cfg.image_std,
            dataset_id=getattr(cfg, "dataset_id", "EPFL-ECEO/coralscapes"),
            cache_dir=getattr(cfg, "dataset_cache_dir", None),
            hf_token=getattr(cfg, "hf_token", None),
        )
    else:
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
    cfg: ConfigType,
    visualize_dir: Optional[Path] = None,
) -> dict:
    model = CoralModel(cfg).to(device)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)
    trainer = CoralTrainer(model, cfg)
    metrics = trainer.evaluate(loader, device, stage=cfg.split, visualize_dir=visualize_dir)
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

    # Scale the canvas to keep dense label sets (e.g., 40 CoralScapes classes) readable.
    num_classes = len(class_names)
    fig_width = 20
    fig_height = 20
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(norm_matrix, interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Normalized Confusion Matrix")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    ticks = np.arange(num_classes)
    font_size = 8 if num_classes > 20 else 10
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=55, ha="right", fontsize=font_size)
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names, fontsize=font_size)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")

    for i in range(norm_matrix.shape[0]):
        for j in range(norm_matrix.shape[1]):
            value = norm_matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=font_size,
            )

    # Reserve extra padding for long rotated labels.
    fig.subplots_adjust(left=0.32, bottom=0.35, right=0.98, top=0.92)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_paths = [
        p for p in sorted(Path(args.checkpoint_root).rglob("*.pth")) if not p.name.endswith("_last.pth")
    ]
    if args.dataset_keyword:
        keyword = args.dataset_keyword.lower()
        checkpoint_paths = [p for p in checkpoint_paths if keyword in str(p).lower()]
    if not checkpoint_paths:
        if args.dataset_keyword:
            print(
                f"No eligible checkpoints under '{args.checkpoint_root}' containing keyword '{args.dataset_keyword}'."
            )
        else:
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
        dataset_choice = determine_dataset(ckpt_path, args.dataset, args.dataset_keyword)
        cfg = build_config(args, scenario_name, model_type, dataset_choice)
        loader = prepare_dataloader(cfg, dataset_choice, args.batch_size, args.num_workers)
        print(f"Evaluating {ckpt_path} on split='{args.split}' ...")
        result = evaluate_checkpoint(ckpt_path, device, loader, cfg, visualize_dir=Path(args.fig_root) / scenario_name / "visualizations" if args.fig_root else None)
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
