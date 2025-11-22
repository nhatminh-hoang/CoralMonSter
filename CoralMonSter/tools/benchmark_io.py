#!/usr/bin/env python3
"""
Benchmark script for CoralMonSter dataset I/O.
"""

import argparse
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Ensure we can import CoralMonSter
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from CoralMonSter.core.config import build_config_from_args, add_common_args, add_training_args
from CoralMonSter.core.factory import build_dataset, prepare_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CoralMonSter Dataset I/O")
    add_common_args(parser)
    add_training_args(parser)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    return parser.parse_args()

def benchmark(loader, device, desc="Benchmark"):
    # Warmup
    print(f"Warming up ({desc})...")
    iter_loader = iter(loader)
    for _ in range(5):
        try:
            batch = next(iter_loader)
            if device.type == 'cuda':
                if "images" in batch:
                    batch["images"] = batch["images"].to(device, non_blocking=True)
                if "masks" in batch:
                    batch["masks"] = batch["masks"].to(device, non_blocking=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        except StopIteration:
            break

    print(f"Running {desc}...")
    start_time = time.perf_counter()
    count = 0
    
    timings = {
        "load_io": 0.0,
        "mask_resize": 0.0,
        "image_transform": 0.0,
        "prompt_sampling": 0.0,
        "total_getitem": 0.0
    }

    for i, batch in enumerate(tqdm(loader, desc=desc)):
        # Simulate transfer to GPU
        if device.type == 'cuda':
            if "images" in batch:
                batch["images"] = batch["images"].to(device, non_blocking=True)
            if "masks" in batch:
                batch["masks"] = batch["masks"].to(device, non_blocking=True)
            torch.cuda.synchronize()
        
        count += batch["images"].shape[0]
        
        # Aggregate timings if available
        if "timings" in batch:
            # batch["timings"] is a dict of floats (sum of batch)
            for k, v in batch["timings"].items():
                if k in timings:
                     timings[k] += v

    total_time = time.perf_counter() - start_time
    throughput = count / total_time
    
    print(f"\nResults for {desc}:")
    print(f"  Total Time: {total_time:.4f}s")
    print(f"  Total Samples: {count}")
    print(f"  Throughput: {throughput:.2f} samples/s")
    
    if count > 0:
        print("\nAverage Internal Timings (per sample):")
        for k, v in timings.items():
            print(f"  {k}: {v/count*1000:.2f} ms")

def main():
    args = parse_args()
    cfg = build_config_from_args(args, mode="train")
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu_devices else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
    
    print(f"Benchmarking on {device}")
    
    # Build Dataset
    dataset = build_dataset(cfg, args.split, args.dataset)
    print(f"Dataset size: {len(dataset)}")

    # Build Loader
    loader = prepare_dataloader(
        dataset, 
        cfg.optimization.batch_size, 
        cfg.optimization.num_workers, 
        shuffle=False, 
        limit=args.limit
    )
    
    benchmark(loader, device, desc="Full Pipeline")

if __name__ == "__main__":
    main()
