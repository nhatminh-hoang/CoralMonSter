from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Union, List

from CoralMonSter import HKCoralConfig, CoralScapesConfig
from CoralMonSter.configs.scenarios import SCENARIO_PRESETS, apply_scenario_preset
from CoralMonSter.utils.common import parse_prompt_bins

ConfigType = Union[HKCoralConfig, CoralScapesConfig]

def add_common_args(parser: argparse.ArgumentParser):
    """Add arguments common to training and evaluation."""
    group = parser.add_argument_group("Common")
    group.add_argument("--dataset", type=str, choices=["hkcoral", "coralscapes"], default="hkcoral")
    group.add_argument("--dataset_root", type=str, default="data_storage/HKCoral", help="Dataset root directory.")
    group.add_argument("--dataset_cache_dir", type=str, default=None, help="Cache directory for CoralScapes.")
    group.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face token.")
    group.add_argument("--sam_checkpoint", type=str, required=True, help="Path to SAM checkpoint.")
    group.add_argument("--model_type", type=str, default=None, choices=["vit_h", "vit_l", "vit_b"])
    group.add_argument("--image_size", type=int, default=1024)
    group.add_argument("--gpu", type=int, default=0)
    group.add_argument("--num_workers", type=int, default=4)
    group.add_argument("--batch_size", type=int, default=2)
    group.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage (20%% slower per batch)",
    )
    group.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use Flash Attention for 8x memory reduction (requires flash-attn package)",
    )
    group.add_argument("--use_lora", action="store_true", help="Enable LoRA for image encoder")
    group.add_argument("--lora_r", type=int, default=4)
    group.add_argument("--lora_alpha", type=int, default=32)
    group.add_argument("--lora_dropout", type=float, default=0.05)
    group.add_argument("--lora_target_modules", type=str, nargs="+", default=["qkv"])

def add_training_args(parser: argparse.ArgumentParser):
    """Add arguments specific to training."""
    group = parser.add_argument_group("Training")
    group.add_argument("--scenario_name", type=str, default="default")
    group.add_argument("--scenario_preset", type=str, choices=sorted(SCENARIO_PRESETS.keys()), default=None)
    group.add_argument("--max_epochs", type=int, default=40)
    group.add_argument("--learning_rate", type=float, default=1e-4)
    group.add_argument("--weight_decay", type=float, default=0.01)
    group.add_argument("--ema_momentum_min", type=float, default=0.996)
    group.add_argument("--ema_momentum_max", type=float, default=1.0)
    group.add_argument("--output_dir", type=str, default="checkpoints")
    group.add_argument("--log_pca_features", action="store_true")
    group.add_argument("--pca_samples_per_epoch", type=int, default=2)
    group.add_argument("--resume", type=str, default=None)
    group.add_argument("--profile", action="store_true")
    group.add_argument("--gpu_devices", type=str, default=None)
    group.add_argument("--prompt_bins", type=parse_prompt_bins, default=None)
    group.add_argument("--train_subset", type=int, default=None)
    group.add_argument("--val_subset", type=int, default=None)
    group.add_argument("--test_subset", type=int, default=None)

def build_config_from_args(args: argparse.Namespace, mode: str = "train") -> ConfigType:
    """Build configuration object from parsed arguments."""
    dataset_choice = args.dataset.lower()
    dataset_root = Path(args.dataset_root)
    
    # Determine model type if not provided
    model_type = args.model_type or "vit_h" # Fallback, caller might refine this

    base_kwargs = dict(
        dataset_root=dataset_root,
        split="train" if mode == "train" else getattr(args, "split", "test"),
        image_size=args.image_size,
        model_type=model_type,
        sam_checkpoint=Path(args.sam_checkpoint),
        scenario_name=getattr(args, "scenario_name", "default"),
        use_gradient_checkpointing=getattr(args, "use_gradient_checkpointing", False),
        use_flash_attention=getattr(args, "use_flash_attention", False),
        use_lora=getattr(args, "use_lora", False),
        lora_r=getattr(args, "lora_r", 4),
        lora_alpha=getattr(args, "lora_alpha", 32),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        lora_target_modules=getattr(args, "lora_target_modules", ["qkv"]),
    )

    if dataset_choice == "coralscapes":
        cfg = CoralScapesConfig(
            **base_kwargs,
            dataset_cache_dir=Path(args.dataset_cache_dir).expanduser() if args.dataset_cache_dir else None,
            hf_token=args.hf_token,
        )
    else:
        cfg = HKCoralConfig(**base_kwargs)

    if mode == "train":
        cfg.freeze_image_encoder = True
        if args.scenario_preset:
            cfg = apply_scenario_preset(cfg, args.scenario_preset)
            if args.scenario_name == "default":
                 cfg.scenario_name = args.scenario_preset # Update name if default
        
        if args.prompt_bins:
            cfg.prompt_bins = args.prompt_bins
            
        cfg.optimization.batch_size = args.batch_size
        cfg.optimization.num_workers = args.num_workers
        cfg.optimization.max_epochs = args.max_epochs
        cfg.optimization.learning_rate = args.learning_rate
        cfg.optimization.weight_decay = args.weight_decay
        cfg.optimization.ema_momentum_min = args.ema_momentum_min
        cfg.optimization.ema_momentum_max = args.ema_momentum_max
        cfg.checkpoint_root = Path(args.output_dir)
        cfg.enable_pca_logging = bool(args.log_pca_features)
        cfg.pca_samples_per_epoch = max(0, args.pca_samples_per_epoch)

    return cfg
