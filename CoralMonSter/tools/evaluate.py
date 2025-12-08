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
import sys
from pathlib import Path
from typing import List

# Ensure we can import CoralMonSter if running from tools/
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import torch
from CoralMonSter.core.config import add_common_args, build_config_from_args
from CoralMonSter.engine.runner import CoralRunner
from CoralMonSter.utils.common import (
    collect_checkpoints,
    infer_dataset_from_path,
    infer_model_type_from_checkpoint,
)
from CoralMonSter.utils.visualize import save_confusion_figure


def _clone_args(args: argparse.Namespace, **overrides) -> argparse.Namespace:
    """Create a lightweight argparse.Namespace copy with overrides."""
    merged = vars(args).copy()
    merged.update(overrides)
    return argparse.Namespace(**merged)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in a directory.")
    add_common_args(parser)
    
    # Add eval-specific args
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints", help="Root directory to scan for .pth files.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--output", type=str, default="checkpoint_evaluations.json", help="Where to dump aggregated results.")
    parser.add_argument("--log_root", type=str, default="logs", help="Base log directory.")
    parser.add_argument("--fig_root", type=str, default="confusion_figs", help="Directory for confusion matrix plots.")
    parser.add_argument(
        "--dataset_keyword",
        type=str,
        default=None,
        help="Only evaluate checkpoints whose path contains this (case-insensitive) keyword."
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    runner = CoralRunner(args, mode="eval")
    
    checkpoint_paths = collect_checkpoints(Path(args.checkpoint_root), args.dataset_keyword)
    if not checkpoint_paths:
        message = "No eligible checkpoints found under" if not args.dataset_keyword else "No eligible checkpoints under"
        suffix = f" containing keyword '{args.dataset_keyword}'" if args.dataset_keyword else ""
        print(f"{message} '{args.checkpoint_root}'{suffix}.")
        return

    results: List[dict] = []
    for ckpt_path in checkpoint_paths:
        scenario_name = ckpt_path.parent.name
        inferred = infer_model_type_from_checkpoint(ckpt_path)
        model_type = args.model_type or inferred or "vit_h"
        
        if args.model_type is None and inferred is None:
            print(f"[Info] Could not infer model type from '{ckpt_path.name}', defaulting to 'vit_h'.")
        elif args.model_type is None and inferred is not None:
            print(f"[Info] Detected backbone '{inferred}' from '{ckpt_path.name}'.")
        dataset_choice = infer_dataset_from_path(ckpt_path, args.dataset_keyword, fallback=args.dataset)
        cfg_args = _clone_args(
            args,
            dataset=dataset_choice,
            model_type=model_type,
            scenario_name=scenario_name,
        )
        cfg = build_config_from_args(cfg_args, mode="eval")
        cfg.checkpoint_root = Path(args.checkpoint_root)
        cfg.log_root = Path(args.log_root)
        cfg.resolve_paths()

        print(f"Evaluating {ckpt_path} on split='{args.split}' ...")
        visualize_dir = Path(args.fig_root) / scenario_name / "visualizations" if args.fig_root else None
        
        result = runner.evaluate(cfg, ckpt_path, visualize_dir=visualize_dir)
        
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
