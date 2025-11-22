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
from CoralMonSter.utils.common import infer_model_type_from_checkpoint
from CoralMonSter.utils.visualize import save_confusion_figure

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

    results: List[dict] = []
    for ckpt_path in checkpoint_paths:
        scenario_name = ckpt_path.parent.name
        inferred = infer_model_type_from_checkpoint(ckpt_path)
        model_type = args.model_type or inferred or "vit_h"
        
        if args.model_type is None and inferred is None:
            print(f"[Info] Could not infer model type from '{ckpt_path.name}', defaulting to 'vit_h'.")
        elif args.model_type is None and inferred is not None:
            print(f"[Info] Detected backbone '{inferred}' from '{ckpt_path.name}'.")
            
        # Determine dataset choice logic (kept from original)
        dataset_choice = args.dataset
        if not dataset_choice: # If not explicitly set, try to infer
             lower_path = str(ckpt_path).lower()
             keyword_lower = args.dataset_keyword.lower() if args.dataset_keyword else ""
             if "coralscapes" in lower_path or "coralscapes" in keyword_lower:
                 dataset_choice = "coralscapes"
             else:
                 dataset_choice = "hkcoral"
        
        # Temporarily override args for build_config
        args.dataset = dataset_choice
        args.model_type = model_type
        args.scenario_name = scenario_name
        
        cfg = build_config_from_args(args, mode="eval")
        cfg.checkpoint_root = Path(args.checkpoint_root) # Restore original checkpoint root if needed
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
