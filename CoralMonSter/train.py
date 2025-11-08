"""
Command-line entry point for training CoralMonSter on HKCoral.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from CoralMonSter import CoralMonSter as CoralModel
from CoralMonSter import CoralTrainer, HKCoralConfig
from CoralMonSter.data import HKCoralDataset, hkcoral_collate_fn
from CoralMonSter.utils import save_checkpoint


def infer_model_type_from_checkpoint(path: str) -> Optional[str]:
    name = Path(path).name.lower()
    for candidate in ["vit_h", "vit_l", "vit_b"]:
        if candidate in name:
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CoralMonSter on HKCoral")
    parser.add_argument("--dataset_root", type=str, default="datasets/HKCoral")
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default=None, choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--scenario_name", type=str, default="default")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu) if device.type == "cuda" else None
    torch.set_float32_matmul_precision("high") if device.type == "cuda" else None

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

    cfg = HKCoralConfig(
        dataset_root=Path(args.dataset_root),
        split="train",
        image_size=args.image_size,
        model_type=model_type,
        sam_checkpoint=Path(args.sam_checkpoint),
        scenario_name=args.scenario_name,
    )
    cfg.freeze_image_encoder = True
    cfg.optimization.batch_size = args.batch_size
    cfg.optimization.num_workers = args.num_workers
    cfg.optimization.max_epochs = args.max_epochs
    cfg.optimization.learning_rate = args.learning_rate
    cfg.optimization.weight_decay = args.weight_decay
    cfg.optimization.ema_momentum_min = args.ema_momentum_min
    cfg.optimization.ema_momentum_max = args.ema_momentum_max
    cfg.checkpoint_root = Path(args.output_dir)

    model = CoralModel(cfg).to(device)
    # model = torch.compile(model) if torch.__version__ >= "2.0.0" else model
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)

    train_dataset = HKCoralDataset(
        cfg.dataset_root,
        "train",
        cfg.image_size,
        cfg.num_classes,
        cfg.ignore_label,
        prompt_points=cfg.distillation.prompt_points,
        prompt_bins=cfg.prompt_bins,
        mean=cfg.image_mean,
        std=cfg.image_std,
    )
    val_dataset = HKCoralDataset(
        cfg.dataset_root,
        "val",
        cfg.image_size,
        cfg.num_classes,
        cfg.ignore_label,
        prompt_points=cfg.distillation.prompt_points,
        prompt_bins=cfg.prompt_bins,
        mean=cfg.image_mean,
        std=cfg.image_std,
    )
    test_dataset = HKCoralDataset(
        cfg.dataset_root,
        "test",
        cfg.image_size,
        cfg.num_classes,
        cfg.ignore_label,
        prompt_points=cfg.distillation.prompt_points,
        prompt_bins=cfg.prompt_bins,
        mean=cfg.image_mean,
        std=cfg.image_std,
    )
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

    trainer = CoralTrainer(model, cfg)
    best_path = trainer.fit(train_loader, val_loader, test_loader)

    final_path = cfg.save_dir / f"{cfg.model_type}_coralmonster_last.pth"
    save_checkpoint({"model": model.state_dict()}, final_path)
    print(f"Final checkpoint saved to {final_path}")
    if best_path:
        print(f"Best mIoU checkpoint saved to {best_path}")


if __name__ == "__main__":
    main()
