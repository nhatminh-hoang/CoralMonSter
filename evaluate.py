#!/usr/bin/env python3
"""
CoralMonSter Evaluation Script

Evaluates a trained checkpoint on the test/val split and outputs:
  - Per-class IoU, mIoU, Pixel Accuracy
  - Confusion matrix visualization
  - Segmentation comparison images (image | GT | prediction)
  - Metrics saved as JSON and text

Usage:
    python evaluate.py --config configs/hkcoral_vit_b.yaml \
                       --checkpoint logs/hkcoral/checkpoints/best.pth \
                       --split test
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import yaml

from coralmonster.arch import CoralMonSter
from coralmonster.encoder import build_sam_encoder

from data import HKCoralDataset, CoralScapesDataset

from utils.metrics import SegmentationMeter
from utils.visualization import (
    colorize_mask,
    denormalize_image,
    get_dataset_info,
    save_confusion_matrix,
    save_segmentation_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoralMonSter Evaluation")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory (default: from config)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_images", action="store_true", default=True,
                        help="Save segmentation comparison images")
    parser.add_argument("--max_vis", type=int, default=50,
                        help="Max number of visualization images to save")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load config ──────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    num_classes = data_cfg["num_classes"]
    image_size = data_cfg["image_size"]
    ignore_index = data_cfg.get("ignore_label", 255)

    dataset_name = data_cfg["dataset"]
    class_names, palette = get_dataset_info(dataset_name)

    # ── Output directory ─────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        scenario = cfg.get("scenario_name", dataset_name)
        output_dir = Path("logs") / scenario

    viz_dir = output_dir / "test_set_visualization"
    cm_dir = output_dir / "confusion_matrix"
    viz_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    # ── Build dataset ────────────────────────────────────────────────
    common_kwargs = {
        "image_size": image_size,
        "num_classes": num_classes,
        "ignore_label": ignore_index,
        "prompt_points": cfg.get("distillation", {}).get("gt_points_per_class", 10),
        "mean": tuple(data_cfg.get("image_mean", [0.485, 0.456, 0.406])),
        "std": tuple(data_cfg.get("image_std", [0.229, 0.224, 0.225])),
    }

    if dataset_name.lower() in ("hkcoral", "hk_coral"):
        DatasetClass = HKCoralDataset
    elif dataset_name.lower() in ("coralscapes", "coral_scapes"):
        DatasetClass = CoralScapesDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    eval_ds = DatasetClass(root=data_cfg["root"], split=args.split, **common_kwargs)
    print(f"[Eval] Dataset: {dataset_name} ({args.split}), {len(eval_ds)} samples")

    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.get("training", {}).get("num_workers", 4),
        pin_memory=True,
    )

    # ── Build and load model ─────────────────────────────────────────
    print(f"[Eval] Loading checkpoint: {args.checkpoint}")
    sam = build_sam_encoder(
        model_type=model_cfg["type"],
        checkpoint_path=model_cfg["checkpoint"],
        image_size=image_size,
    )

    model = CoralMonSter(
        sam_model=sam,
        num_classes=num_classes,
        image_size=image_size,
        freeze_image_encoder=model_cfg.get("freeze_image_encoder", False),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_miou = ckpt.get("val_miou", "?")
    print(f"  Checkpoint from epoch {ckpt_epoch}, val mIoU={ckpt_miou}")

    # ── Evaluate ─────────────────────────────────────────────────────
    meter = SegmentationMeter(num_classes, ignore_index=ignore_index)
    mean = data_cfg.get("image_mean", [0.485, 0.456, 0.406])
    std = data_cfg.get("image_std", [0.229, 0.224, 0.225])

    n_saved = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            outputs = model(images, gt_masks=masks)
            preds = outputs["student_logits"].argmax(dim=1)

            meter.update(preds, masks)

            # Save visualization images
            if args.save_images and n_saved < args.max_vis:
                for j in range(images.shape[0]):
                    if n_saved >= args.max_vis:
                        break

                    img_np = denormalize_image(images[j].cpu(), mean, std)
                    pred_np = preds[j].cpu().numpy()
                    gt_np = masks[j].cpu().numpy()
                    fname = batch["file_name"][j] if "file_name" in batch else f"sample_{n_saved}"

                    save_segmentation_comparison(
                        image=img_np,
                        prediction=pred_np,
                        target=gt_np,
                        palette=palette,
                        class_names=class_names,
                        save_path=viz_dir / f"{Path(fname).stem}.png",
                        ignore_index=ignore_index,
                    )
                    n_saved += 1

    # ── Print results ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Evaluation Results — {dataset_name} ({args.split})")
    print(f"{'='*60}")
    print(meter.summary_string(class_names))
    print(f"{'='*60}")

    # ── Save confusion matrix ────────────────────────────────────────
    cm = meter.confusion_matrix()
    save_confusion_matrix(
        cm, class_names,
        save_path=cm_dir / f"confusion_matrix_{args.split}.png",
        title=f"Confusion Matrix ({args.split}, mIoU={meter.mean_iou()*100:.2f}%)",
    )
    print(f"\nConfusion matrix saved to: {cm_dir}")

    # ── Save metrics JSON ────────────────────────────────────────────
    iou_per_class = meter.per_class_iou()
    metrics_dict = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "mean_iou": meter.mean_iou(),
        "pixel_accuracy": meter.pixel_accuracy(),
        "per_class_iou": {
            class_names[i]: float(iou_per_class[i])
            for i in range(num_classes)
        },
    }

    metrics_path = output_dir / "loggings" / f"eval_metrics_{args.split}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")
    if n_saved > 0:
        print(f"Visualizations saved to: {viz_dir} ({n_saved} images)")


if __name__ == "__main__":
    main()
