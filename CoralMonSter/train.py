"""
Command-line entry point for training CoralMonSter on HKCoral.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from CoralMonSter import CoralMonSter as CoralModel
from CoralMonSter import CoralTrainer, HKCoralConfig
from CoralMonSter.configs.scenarios import SCENARIO_PRESETS, apply_scenario_preset
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
    parser.add_argument(
        "--gpu_devices",
        type=str,
        default=None,
        help="Comma-separated list of GPU device indices to use (e.g., '5,6,7').",
    )
    parser.add_argument("--distributed", action="store_true", help="Enable DistributedDataParallel training.")
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--dist_url", type=str, default="env://")
    return parser.parse_args()


def init_distributed_mode(args: argparse.Namespace) -> bool:
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.gpu))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
    elif args.distributed:
        raise ValueError("Distributed training requested but RANK/WORLD_SIZE are not set.")
    else:
        args.rank = 0
        args.world_size = 1
        return False

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.cuda.set_device(args.local_rank)
    return True


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def get_model_state(model: torch.nn.Module) -> dict:
    if isinstance(model, DDP):
        return model.module.state_dict()
    return model.state_dict()


def main() -> None:
    args = parse_args()
    if args.gpu_devices and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    distributed = init_distributed_mode(args)
    if distributed:
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(args.gpu)
    if device.type == "cuda":
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

    cfg = HKCoralConfig(
        dataset_root=Path(args.dataset_root),
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
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.optimization.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.optimization.num_workers,
        collate_fn=hkcoral_collate_fn,
        pin_memory=True,
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

    if distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    trainer = CoralTrainer(
        model,
        cfg,
        enable_profile=args.profile,
        distributed=distributed,
        local_rank=getattr(args, "local_rank", 0),
    )
    best_path = trainer.fit(
        train_loader,
        val_loader,
        test_loader,
        train_sampler=train_sampler,
    )

    final_path = cfg.save_dir / f"{cfg.model_type}_coralmonster_last.pth"
    if is_main_process():
        save_checkpoint({"model": get_model_state(model)}, final_path)
        print(f"Final checkpoint saved to {final_path}")
        if best_path:
            print(f"Best mIoU checkpoint saved to {best_path}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
