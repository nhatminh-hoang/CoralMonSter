#!/usr/bin/env python3
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

# Ensure we can import CoralMonSter if running from tools/
ROOT_DIR = Path(__file__).resolve().parents[2] # tools -> CoralMonSter -> root
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from CoralMonSter.core.config import add_common_args, add_training_args, build_config_from_args
from CoralMonSter.engine.runner import CoralRunner

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CoralMonSter on coral datasets")
    add_common_args(parser)
    add_training_args(parser)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    runner = CoralRunner(args, mode="train")
    
    # Build Config
    cfg = build_config_from_args(args, mode="train")
    
    # Run Training
    runner.train(cfg)

if __name__ == "__main__":
    main()
