#!/usr/bin/env python3
"""Evaluate CoralScapes semantic segmentation checkpoints without the upstream repo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from CoralMonSter.data import CoralScapesDataset, hkcoral_collate_fn
from CoralMonSter.models.coralscapes_baselines import BASELINE_SPECS, load_coralscapes_baseline
from CoralMonSter.utils import SegmentationMeter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate released CoralScapes baseline checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth checkpoint file.")
    parser.add_argument(
        "--model_key",
        type=str,
        required=True,
        choices=sorted(BASELINE_SPECS.keys()),
        help="Which architecture the checkpoint belongs to (matches filename prefix).",
    )
    parser.add_argument("--dataset_split", type=str, default="test", choices=["train", "validation", "test"], help="CoralScapes split to evaluate.")
    parser.add_argument("--dataset_id", type=str, default="EPFL-ECEO/coralscapes", help="HF dataset identifier.")
    parser.add_argument("--dataset_cache_dir", type=str, default=None, help="Optional huggingface cache directory.")
    parser.add_argument("--hf_token", type=str, default=None, help="HF token for private cache, if required.")
    parser.add_argument("--image_size", type=int, default=1024, help="Resolution to feed into the models.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ignore_label", type=int, default=0, help="Label ID to ignore during metrics.")
    parser.add_argument("--num_classes", type=int, default=40, help="Number of semantic classes (including background).")
    parser.add_argument("--hf_offline", action="store_true", help="Force HF_DATASETS_OFFLINE=1 for fully offline reads.")
    return parser.parse_args()


def build_dataloader(args: argparse.Namespace) -> DataLoader:
    dataset = CoralScapesDataset(
        split=args.dataset_split,
        image_size=args.image_size,
        num_classes=args.num_classes,
        ignore_label=args.ignore_label,
        prompt_points=0,
        prompt_bins=(1,),
        dataset_id=args.dataset_id,
        cache_dir=Path(args.dataset_cache_dir) if args.dataset_cache_dir else None,
        hf_token=args.hf_token,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=hkcoral_collate_fn,
    )
    return loader


def forward_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    outputs = model(images)
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    if hasattr(outputs, "logits"):
        return outputs.logits  # huggingface SemanticSegmenterOutput
    return outputs


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int, ignore_label: int) -> dict:
    meter = SegmentationMeter(num_classes=num_classes, ignore_index=ignore_label, eval_ignore_classes=())
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            logits = forward_logits(model, images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds = logits.argmax(dim=1)
            meter.update(preds.cpu(), masks.cpu())
            loss = ce_loss(logits, masks.long())
            batch_size = images.shape[0]
            loss_sum += loss.item() * batch_size
            sample_count += batch_size
    metrics = {
        "loss": loss_sum / max(sample_count, 1),
        "miou": meter.mean_iou(),
        "pixel_accuracy": meter.pixel_accuracy(),
        "per_class_iou": meter.per_class_iou(),
    }
    return metrics


def main() -> None:
    args = parse_args()
    if args.hf_offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    device = torch.device(args.device)
    loader = build_dataloader(args)
    model = load_coralscapes_baseline(args.model_key, args.checkpoint, args.num_classes)
    model.to(device)
    metrics = evaluate(model, loader, device, args.num_classes, args.ignore_label)
    print("=== Evaluation Results ===")
    for key, value in metrics.items():
        if isinstance(value, list):
            print(f"{key}: {[round(v, 4) for v in value]}")
        else:
            print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
