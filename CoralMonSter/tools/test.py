#!/usr/bin/env python3
"""
Prompt-free CoralMonSter inference
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure we can import CoralMonSter if running from tools/
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from CoralMonSter import HKCoralConfig
from CoralMonSter.engine.runner import CoralRunner

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-free CoralMonSter inference")
    parser.add_argument("--model_type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to the base SAM checkpoint")
    parser.add_argument("--student_weights", type=str, default=None, help="Fine-tuned CoralMonSter weights")
    parser.add_argument("--test_img_path", type=str, required=True, help="Directory with test images")
    parser.add_argument("--output_path", type=str, required=True, help="Directory where JSON predictions are stored")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--prob_threshold", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    runner = CoralRunner(args, mode="test")
    
    # Build model using factory or manual config since test.py has specific needs
    # test.py uses HKCoralConfig but with test_img_path as dataset_root which is a bit hacky in original code
    # We will replicate the original logic but using the config class directly as it was
    cfg = HKCoralConfig(
        dataset_root=Path(args.test_img_path),
        split="test",
        image_size=args.image_size,
        model_type=args.model_type,
        sam_checkpoint=Path(args.sam_checkpoint),
    )
    
    runner.predict(cfg, student_weights=args.student_weights)

if __name__ == "__main__":
    main()
