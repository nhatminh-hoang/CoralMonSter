#!/usr/bin/env python3
"""
Command-line entry point for training CoralMonSter on HKCoral or CoralScapes.
"""

from __future__ import annotations

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure we can import CoralMonSter if running from tools/
ROOT_DIR = Path(__file__).resolve().parents[2] # tools -> CoralMonSter -> root
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from CoralMonSter.core.config import add_common_args, add_training_args, build_config_from_args
from CoralMonSter.engine.runner import CoralRunner
from CoralMonSter.utils import save_aggregated_training_curves


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible multi-run training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CoralMonSter on coral datasets")
    add_common_args(parser)
    add_training_args(parser)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    all_test_metrics = []
    run_log_dirs = []
    base_scenario_name = None
    base_log_root = None
    for run_idx in range(args.num_runs):
        print(f"Starting run {run_idx + 1}/{args.num_runs}")
        runner = CoralRunner(args, mode="train")
        
        # Build Config
        cfg = build_config_from_args(args, mode="train")
        cfg.seed = args.seed + run_idx
        base_name = cfg.scenario_name
        if base_scenario_name is None:
            base_scenario_name = base_name
            base_log_root = cfg.log_root
        cfg.scenario_name = f"{base_name}_run{run_idx}"
        cfg.resolve_paths()

        run_log_dirs.append(cfg.log_dir)

        set_seed(cfg.seed)
        
        # Run Training
        test_metrics = runner.train(cfg)
        if test_metrics is not None:
            all_test_metrics.append(test_metrics)

        # Free memory before the next run
        del runner
        del cfg
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    
    # Aggregate training curves across runs (if multiple runs)
    if len(run_log_dirs) > 1 and base_scenario_name and base_log_root:
        histories = []
        for log_dir in run_log_dirs:
            metrics_path = Path(log_dir) / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    histories.append(json.load(f))
        if histories:
            mean_log_dir = Path(base_log_root) / base_scenario_name
            mean_log_dir.mkdir(parents=True, exist_ok=True)
            save_aggregated_training_curves(histories, mean_log_dir / "training_curves_mean.png")

    if all_test_metrics:
        # Compute mean metrics
        mean_metrics = {}
        for key in all_test_metrics[0].keys():
            first_val = all_test_metrics[0][key]

            # Scalar metrics
            if isinstance(first_val, (int, float)):
                values = [m[key] for m in all_test_metrics]
                mean_metrics[key] = sum(values) / len(values)
                continue

            # List metrics (per-class or confusion matrices)
            if isinstance(first_val, list):
                if not first_val:
                    mean_metrics[key] = first_val
                    continue

                # List of scalars
                if isinstance(first_val[0], (int, float)):
                    mean_metrics[key] = []
                    for i in range(len(first_val)):
                        values = [m[key][i] for m in all_test_metrics]
                        mean_metrics[key].append(sum(values) / len(values))
                    continue

                # List of lists (e.g., confusion matrix)
                if isinstance(first_val[0], list):
                    rows = len(first_val)
                    cols = len(first_val[0])
                    matrix_mean = []
                    for r in range(rows):
                        row_mean = []
                        for c in range(cols):
                            values = [m[key][r][c] for m in all_test_metrics]
                            row_mean.append(sum(values) / len(values))
                        matrix_mean.append(row_mean)
                    mean_metrics[key] = matrix_mean
                    continue

            # Fallback: keep the last run's value for unsupported types
            mean_metrics[key] = all_test_metrics[-1][key]
        
        print(f"\nMean test results over {len(all_test_metrics)} runs:")
        print(f"Mean loss={mean_metrics['loss']:.4f}, mIoU={mean_metrics['miou']:.4f}, pixAcc={mean_metrics['pix_acc']:.4f}")
    else:
        print("No test metrics collected (test_loader was None)")

if __name__ == "__main__":
    main()
