
import sys
import os
from pathlib import Path
import torch

# Add root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from CoralMonSter.models.backbones import build_sam_backbone

def inspect():
    # Mock config or just build backbone directly
    # We need a checkpoint path. The user command has checkpoints/vit_b_coralscop.pth
    # But I might not have it. Let's check if it exists or use a default if possible.
    # The user command implies it exists.
    
    ckpt = Path("checkpoints/vit_b_coralscop.pth")
    if not ckpt.exists():
        print(f"Checkpoint {ckpt} not found. Trying to find any .pth in checkpoints/")
        ckpts = list(Path("checkpoints").glob("*.pth"))
        if ckpts:
            ckpt = ckpts[0]
            print(f"Using {ckpt}")
        else:
            print("No checkpoints found. Cannot load model to inspect.")
            # Try loading without checkpoint if possible (random init)
            ckpt = Path("dummy.pth") # build_sam_backbone handles missing ckpt by random init if we catch the error or if it allows None
            
    try:
        model = build_sam_backbone("vit_b", ckpt)
        print("Model built successfully.")
        
        print("\n--- Image Encoder Modules ---")
        for name, mod in model.image_encoder.named_modules():
            if "attn" in name or "qkv" in name:
                print(name)
                
    except Exception as e:
        print(f"Error building model: {e}")

if __name__ == "__main__":
    inspect()
