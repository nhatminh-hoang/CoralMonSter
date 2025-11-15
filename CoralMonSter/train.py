"""
Command-line entry point for training CoralMonSter on HKCoral or CoralScapes.
"""

from __future__ import annotations

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader, Subset

from CoralMonSter import CoralMonSter as CoralModel
from CoralMonSter import CoralTrainer, CoralScapesConfig, HKCoralConfig
from CoralMonSter.configs.scenarios import SCENARIO_PRESETS, apply_scenario_preset
from CoralMonSter.data import CoralScapesDataset, HKCoralDataset, hkcoral_collate_fn
from CoralMonSter.utils import save_checkpoint

def infer_model_type_from_checkpoint(path: str) -> Optional[str]:
    name = Path(path).name.lower()
    for candidate in ["vit_h", "vit_l", "vit_b"]:
        if candidate in name:
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CoralMonSter on coral datasets")
    parser.add_argument("--dataset", type=str, choices=["hkcoral", "coralscapes"], default="hkcoral")
    parser.add_argument("--dataset_root", type=str, default="datasets/HKCoral")
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default=None, choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--scenario_name", type=str, default="default")
    parser.add_argument(
        "--scenario_preset",
        type=str,
        choices=sorted(SCENARIO_PRESETS.keys()),
        default=None,
        help="Apply a predefined training scenario.",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--ema_momentum_min", type=float, default=0.996)
    parser.add_argument("--ema_momentum_max", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--profile", action="store_true", help="Profile a single forward/backward pass.")
    parser.add_argument("--compile", action="store_true", help="Compile the model for optimized execution.")
    parser.add_argument(
        "--gpu_devices",
        type=str,
        default=None,
        help="Comma-separated list of GPU device indices to use (e.g., '5,6,7').",
    )
    parser.add_argument("--train_subset", type=int, default=None, help="Limit number of training samples per epoch.")
    parser.add_argument("--val_subset", type=int, default=None, help="Limit number of validation samples.")
    parser.add_argument("--test_subset", type=int, default=None, help="Limit number of test samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu_devices and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
        torch.set_float32_matmul_precision("high")

    inferred_model_type = infer_model_type_from_checkpoint(args.sam_checkpoint)
    model_type = args.model_type or inferred_model_type or "vit_h"
    if inferred_model_type and args.model_type and inferred_model_type != args.model_type:
        print(
            f"[Warning] Checkpoint '{args.sam_checkpoint}' looks like '{inferred_model_type}' "
            f"but '--model_type' was set to '{args.model_type}'. Using '{args.model_type}'."
        )
    elif args.model_type is None and inferred_model_type:
        print(f"[Info] Detected backbone '{inferred_model_type}' from checkpoint name.")
    elif args.model_type is None and inferred_model_type is None:
        print("[Info] Could not infer backbone from checkpoint name; defaulting to 'vit_h'.")

    scenario_label = args.scenario_name
    if args.scenario_preset and scenario_label == "default":
        scenario_label = args.scenario_preset

    dataset_choice = args.dataset.lower()
    dataset_root = Path(args.dataset_root)
    dataset_cache_dir = Path(args.dataset_cache_dir).expanduser() if args.dataset_cache_dir else None

    if dataset_choice == "coralscapes":
        cfg = CoralScapesConfig(
            dataset_root=dataset_root,
            split="train",
            image_size=args.image_size,
            model_type=model_type,
            sam_checkpoint=Path(args.sam_checkpoint),
            scenario_name=scenario_label or "default",
            dataset_cache_dir=dataset_cache_dir,
            hf_token=args.hf_token,
        )
    else:
        cfg = HKCoralConfig(
            dataset_root=dataset_root,
            split="train",
            image_size=args.image_size,
            model_type=model_type,
            sam_checkpoint=Path(args.sam_checkpoint),
            scenario_name=scenario_label or "default",
        )
    cfg.freeze_image_encoder = True
    if args.scenario_preset:
        cfg = apply_scenario_preset(cfg, args.scenario_preset)
    cfg.optimization.batch_size = args.batch_size
    cfg.optimization.num_workers = args.num_workers
    cfg.optimization.max_epochs = args.max_epochs
    cfg.optimization.learning_rate = args.learning_rate
    cfg.optimization.weight_decay = args.weight_decay
    cfg.optimization.ema_momentum_min = args.ema_momentum_min
    cfg.optimization.ema_momentum_max = args.ema_momentum_max
    cfg.checkpoint_root = Path(args.output_dir)

    model = CoralModel(cfg).to(device)
    if args.compile:
        print("[Info] Compiling the model for optimized execution...")
        model = torch.compile(model)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)

    def build_dataset(split: str):
        if dataset_choice == "coralscapes":
            hf_split = {"val": "validation"}.get(split, split)
            return CoralScapesDataset(
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
        return HKCoralDataset(
            cfg.dataset_root,
            split,
            cfg.image_size,
            cfg.num_classes,
            cfg.ignore_label,
            prompt_points=cfg.distillation.prompt_points,
            prompt_bins=cfg.prompt_bins,
            mean=cfg.image_mean,
            std=cfg.image_std,
        )

    train_dataset = build_dataset("train")
    val_dataset = build_dataset("val")
    test_dataset = build_dataset("test")

    def maybe_limit(ds, limit):
        if limit is None or limit <= 0:
            return ds
        count = min(limit, len(ds))
        return Subset(ds, list(range(count)))

    train_dataset = maybe_limit(train_dataset, args.train_subset)
    val_dataset = maybe_limit(val_dataset, args.val_subset)
    test_dataset = maybe_limit(test_dataset, args.test_subset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.optimization.batch_size,
        shuffle=True,
        num_workers=cfg.optimization.num_workers,
        collate_fn=hkcoral_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.optimization.batch_size,
        shuffle=False,
        num_workers=cfg.optimization.num_workers,
        collate_fn=hkcoral_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.optimization.batch_size,
        shuffle=False,
        num_workers=cfg.optimization.num_workers,
        collate_fn=hkcoral_collate_fn,
    )

    trainer = CoralTrainer(
        model,
        cfg,
        enable_profile=args.profile,
    )
    best_path = trainer.fit(
        train_loader,
        val_loader,
        test_loader,
    )

    final_path = cfg.save_dir / f"{cfg.model_type}_coralmonster_last.pth"
    save_checkpoint({"model": model.state_dict()}, final_path)
    print(f"Final checkpoint saved to {final_path}")
    if best_path:
        print(f"Best mIoU checkpoint saved to {best_path}")


if __name__ == "__main__":
    main()
