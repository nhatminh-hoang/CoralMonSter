#!/usr/bin/env python3
"""
Advanced Feature Visualization & Comparison Tool for CoralMonSter.
Generates DINO-style attention and cluster visualizations comparing multiple models.
"""

import os
import sys
import argparse
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm
# Optional HF DINOv3; guard import for older transformers
try:  # pragma: no cover
    from transformers import AutoImageProcessor, AutoModel
except Exception:  # pragma: no cover
    AutoImageProcessor = None
    AutoModel = None

try:
    from sklearn.cluster import KMeans
except ImportError:
    print("sklearn not found. Please install scikit-learn.")
    sys.exit(1)

# Add root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from CoralMonSter.core.config import build_config_from_args
from CoralMonSter.core.factory import build_model, build_dataset

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def get_device(gpu_id: int) -> torch.device:
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")

def denormalize_image(image_tensor: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> np.ndarray:
    """Convert normalized tensor (C, H, W) to numpy (H, W, C) in [0, 1]."""
    mean = torch.tensor(mean).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(image_tensor.device)
    img = image_tensor * std + mean
    img = img.permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1)

def robust_normalize(data: np.ndarray, clip_percentile: float = 99.5) -> np.ndarray:
    """Normalize data to [0, 1] with outlier clipping for better contrast."""
    v_min = data.min()
    v_max = np.percentile(data, clip_percentile)
    data = np.clip(data, v_min, v_max)
    return (data - v_min) / (v_max - v_min + 1e-8)

# -----------------------------------------------------------------------------
# Model Wrapper
# -----------------------------------------------------------------------------

class ModelWrapper:
    def __init__(self, name: str, checkpoint_path: Optional[Path], args, device: torch.device,
                 timm_model: Optional[str] = None, hf_dinov3: Optional[str] = None):
        self.name = name
        self.device = device
        self.timm_model = timm_model
        self.hf_dinov3 = hf_dinov3
        self.latest_attn = None
        
        # Build Config (still needed for dataset settings)
        self.cfg = build_config_from_args(args, mode="eval")
        self.cfg.use_flash_attention = False
        self.base_mean = self.cfg.image_mean
        self.base_std = self.cfg.image_std
        
        if hf_dinov3:
            if AutoImageProcessor is None or AutoModel is None:
                raise ImportError("Transformers AutoModel/AutoImageProcessor not available. Please upgrade transformers to a version that includes DINOv3 (e.g., >=4.46).")
            print(f"[{name}] Loading HuggingFace DINOv3 model: {hf_dinov3}")
            self.processor = AutoImageProcessor.from_pretrained(hf_dinov3, token=getattr(args, "hf_token", None))
            self.model = AutoModel.from_pretrained(hf_dinov3, token=getattr(args, "hf_token", None)).to(device)
            self.model.eval()
            if hasattr(self.model, "set_attn_implementation"):
                # Enable eager attention to allow output_attentions=True
                self.model.set_attn_implementation("eager")
            self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
            self.imnet_mean = tuple(self.processor.image_mean)
            self.imnet_std = tuple(self.processor.image_std)
        elif timm_model:
            # TIMM backbone (e.g., DINOv2/v3)
            print(f"[{name}] Building timm backbone: {timm_model}")
            self.model = timm.create_model(
                timm_model,
                pretrained=True,
                num_classes=0,
                global_pool="",
            ).to(device)
            self.model.eval()
            self._patch_timm_attention()
            # Use ImageNet stats for timm backbones
            self.imnet_mean = (0.485, 0.456, 0.406)
            self.imnet_std = (0.229, 0.224, 0.225)
        else:
            # SAM / CoralMonSter checkpoint
            self.cfg.sam_checkpoint = checkpoint_path
            self.is_baseline = ("coralscop" in str(checkpoint_path) or "sam" in str(checkpoint_path).lower())
            print(f"[{name}] Building model...")
            self.model = build_model(self.cfg, device=device)
            self.model.eval()
            if not self.is_baseline:
                print(f"[{name}] Loading weights from {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location=device)
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.model.load_state_dict(state_dict, strict=False)

    def _patch_timm_attention(self) -> None:
        """Patch last block attention to expose weights (similar to SAM patch)."""
        blk = self.model.blocks[-1]
        attn = blk.attn

        def patched_attn(x, attn_mask=None):
            B, N, C = x.shape
            qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn_map = (q @ k.transpose(-2, -1)) * attn.scale
            attn_map = attn_map.softmax(dim=-1)
            self.latest_attn = attn_map.detach().cpu()
            attn_map = attn.attn_drop(attn_map)
            x_out = (attn_map @ v).transpose(1, 2).reshape(B, N, C)
            x_out = attn.proj(x_out)
            x_out = attn.proj_drop(x_out)
            return x_out

        attn.forward = patched_attn  # type: ignore[method-assign]

    def extract_features(self, image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference and return (features_flat, attention_map).
        image: (1, 3, H, W)
        Returns:
            features_flat: (H_feat*W_feat, C) numpy
            attn_avg: (H_feat, W_feat) numpy (averaged over heads and reshaped)
        """
        if self.hf_dinov3:
            # Denormalize to RGB then let HF processor normalize
            mean_sam = torch.tensor(self.base_mean).view(1, 3, 1, 1).to(image.device)
            std_sam = torch.tensor(self.base_std).view(1, 3, 1, 1).to(image.device)
            img = (image * std_sam + mean_sam).squeeze(0).permute(1, 2, 0).cpu().numpy()
            inputs = self.processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                outputs = self.model(pixel_values, output_attentions=True)
            tokens = outputs.last_hidden_state  # (B, N, C)
            attns = outputs.attentions[-1] if outputs.attentions else None  # (B, heads, N, N)
            if attns is not None:
                attn_weights = attns.squeeze(0).cpu()
                start = 1 + getattr(self.model.config, "num_register_tokens", 0)
                attn_weights = attn_weights[:, start:, start:]
            else:
                attn_weights = None
            start = 1 + getattr(self.model.config, "num_register_tokens", 0)
            tokens = tokens[:, start:, :]
            N = tokens.shape[1]
            H_feat = W_feat = int(math.sqrt(N))
            features = tokens.transpose(1, 2).reshape(1, -1, H_feat, W_feat)
        elif self.timm_model:
            # Re-normalize from SAM stats to ImageNet for timm models
            mean_sam = torch.tensor(self.base_mean).view(1, 3, 1, 1).to(image.device)
            std_sam = torch.tensor(self.base_std).view(1, 3, 1, 1).to(image.device)
            mean_im = torch.tensor(self.imnet_mean).view(1, 3, 1, 1).to(image.device)
            std_im = torch.tensor(self.imnet_std).view(1, 3, 1, 1).to(image.device)
            img = image * std_sam + mean_sam
            img = (img - mean_im) / std_im

            # Resize to timm model's expected resolution
            H_t, W_t = self.model.patch_embed.img_size
            if img.shape[-2:] != (H_t, W_t):
                img = F.interpolate(img, size=(H_t, W_t), mode="bicubic", align_corners=False)

            with torch.no_grad():
                tokens = self.model(img)
            # Drop CLS if present
            if hasattr(self.model, "cls_token"):
                tokens = tokens[:, 1:]
            attn_weights = self.latest_attn
            H_feat, W_feat = self.model.patch_embed.grid_size
            features = tokens.transpose(1, 2).reshape(1, -1, H_feat, W_feat)
        else:
            with torch.no_grad():
                features = self.model.image_encoder(image)
            attn_weights = None
            if hasattr(self.model.image_encoder.blocks[-1].attn, "latest_attn"):
                attn_weights = self.model.image_encoder.blocks[-1].attn.latest_attn.squeeze(0).cpu()
            else:
                print(f"[{self.name}] Warning: No attention weights captured.")
            H_feat = features.shape[2]
            W_feat = features.shape[3]

        # Process Features
        features = features.squeeze(0) # (C, H, W)
        C, H, W = features.shape
        features_flat = features.permute(1, 2, 0).reshape(-1, C).cpu().numpy()
        if attn_weights is not None and attn_weights.dim() == 4:
            attn_weights = attn_weights.squeeze(0)
        if attn_weights is not None:
            N_expected = H * W
            if attn_weights.shape[-1] == N_expected + 1:
                attn_weights = attn_weights[:, 1:, 1:]
            # If still mismatched (e.g., different patch grid), reshape conservatively
            if attn_weights.shape[-1] != N_expected:
                # Try infer grid
                side = int(math.sqrt(attn_weights.shape[-1]))
                if side * side == attn_weights.shape[-1]:
                    H = W = side
        return features_flat, attn_weights, (H, W)

# -----------------------------------------------------------------------------
# Visualization Logic
# -----------------------------------------------------------------------------

def visualize_dataset(args):
    device = get_device(args.gpu)
    print(f"Using device: {device}")
    
    # 1. Define Models to Compare
    checkpoints = []
    if args.sam_checkpoint:
        checkpoints.append(("Baseline (SAM)", Path(args.sam_checkpoint)))
    if args.compare_checkpoints:
        for i, ckpt in enumerate(args.compare_checkpoints):
            path = Path(ckpt)
            name = path.parent.name if path.parent.name != "checkpoints" else path.stem
            if "coralmonster" in name: name = name.replace("coralmonster", "CM")
            if "coralscapes" in name: name = name.replace("coralscapes", "")
            name = name.strip("_") or f"Model {i+1}"
            checkpoints.append((name, path))
    
    # Add timm DINO variants
    timm_models = []
    if args.timm_models:
        for tm in args.timm_models:
            timm_models.append(tm)
    hf_dinov3_models = []
    if args.hf_dinov3_models:
        for hm in args.hf_dinov3_models:
            hf_dinov3_models.append(hm)
    
    if not checkpoints and not timm_models and not hf_dinov3_models:
        print("No models provided. Use --sam_checkpoint, --compare_checkpoints, --timm_models, or --hf_dinov3_models.")
        return

    # 2. Load Models
    models = []
    for name, path in checkpoints:
        models.append(ModelWrapper(name, path, args, device))
    for tm in timm_models:
        models.append(ModelWrapper(tm, None, args, device, timm_model=tm))
    for hm in hf_dinov3_models:
        models.append(ModelWrapper(hm, None, args, device, hf_dinov3=hm))
        
    # 3. Load Dataset (from first model cfg)
    print(f"Loading dataset: {args.dataset}")
    dataset = build_dataset(models[0].cfg, split=args.split, dataset_choice=args.dataset)
    dataset.compute_prompts = True
    print(f"Dataset size: {len(dataset)}")
    
    # 4. Select Images
    if args.process_all:
        indices = list(range(len(dataset)))
    elif args.image_index >= 0:
        indices = [args.image_index]
    else:
        if args.num_images > 0:
            indices = np.linspace(0, len(dataset)-1, args.num_images, dtype=int).tolist()
        else:
            indices = [0]
            
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. Process Loop
    for idx in tqdm(indices, desc="Processing Images"):
        process_single_image(idx, dataset, models, output_dir, args, device)

def process_single_image(idx, dataset, models, output_dir, args, device):
    sample = dataset[idx]
    image = sample["image"].unsqueeze(0).to(device)
    mask = sample["mask"]
    gt_points = sample.get("points")
    
    if gt_points is None:
        return

    # Select Query Points (One per class)
    mask_np = mask.cpu().numpy()
    H_img, W_img = mask_np.shape
    points_by_class = {}
    
    for i in range(len(gt_points)):
        pt = gt_points[i]
        x, y = int(pt[0]), int(pt[1])
        x = min(max(x, 0), W_img - 1)
        y = min(max(y, 0), H_img - 1)
        label = mask_np[y, x]
        if label == 0 or label == 255: continue
        if label not in points_by_class: points_by_class[label] = []
        points_by_class[label].append(pt)
        
    query_points = []
    query_labels = []
    for label in sorted(points_by_class.keys()):
        query_points.append(points_by_class[label][0])
        query_labels.append(label)
        
    if not query_points:
        print(f"No valid foreground points for image {idx}")
        return

    # Run Models
    model_results = []
    for model in models:
        feats, attn_weights, (H_feat, W_feat) = model.extract_features(image)
        
        # K-Means
        unique_labels = np.unique(mask_np)
        unique_labels = unique_labels[unique_labels != 255]
        K = max(len(unique_labels), 3)
        
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feats)
        cluster_map = labels.reshape(H_feat, W_feat)

        # Upsample maps to image resolution for equal-sized visualizations
        def _resize_map(arr, target_h, target_w, mode="bilinear"):
            t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
            if mode == "nearest":
                t = F.interpolate(t, size=(target_h, target_w), mode=mode)
            else:
                t = F.interpolate(t, size=(target_h, target_w), mode=mode, align_corners=False)
            return t.squeeze().cpu().numpy()

        upsampled_cluster = _resize_map(cluster_map.astype(np.float32), H_img, W_img, mode="nearest")
        
        model_results.append({
            "attn_weights": attn_weights,
            "cluster_map": cluster_map,
            "cluster_upsampled": upsampled_cluster,
            "feat_shape": (H_feat, W_feat)
        })

    # Plotting
    num_models = len(models)
    num_rows = len(query_points)
    # Cols: Image | GT | (Attn | Cluster) * Models
    num_cols = 2 + (2 * num_models)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 5 * num_rows))
    if num_rows == 1: axes = np.expand_dims(axes, 0)
    
    # Prepare base images
    orig_img_np = denormalize_image(image.squeeze(0), models[0].cfg.image_mean, models[0].cfg.image_std)
    
    # Headers
    cols_titles = ["Query Point", "GT Mask"]
    for m in models:
        cols_titles.extend([f"{m.name}\nAttention", f"{m.name}\nCluster"])
        
    for i in range(num_cols):
        axes[0, i].set_title(cols_titles[i], fontsize=14, pad=10)

    # Rows
    for i, (pt, label_id) in enumerate(zip(query_points, query_labels)):
        # Row Info
        label_name = f"Class {label_id}"
        if hasattr(models[0].cfg, "class_names") and label_id < len(models[0].cfg.class_names):
            label_name = models[0].cfg.class_names[label_id]
            
        # 1. Original Image
        ax = axes[i, 0]
        ax.imshow(orig_img_np)
        ax.plot(pt[0], pt[1], 'rx', markersize=12, markeredgewidth=3)
        ax.set_ylabel(f"{label_name}", fontsize=12, rotation=90, labelpad=5)
        ax.set_xticks([]); ax.set_yticks([])
        
        # 2. GT Mask
        ax = axes[i, 1]
        gt_mask = (mask_np == label_id)
        ax.imshow(gt_mask, cmap="gray", interpolation="nearest")
        ax.plot(pt[0], pt[1], 'rx', markersize=10, markeredgewidth=2)
        ax.axis("off")
        
        # Models
        for m_idx, res in enumerate(model_results):
            H_f, W_f = res["feat_shape"]
            scale_x = W_f / W_img
            scale_y = H_f / H_img
            qx = min(max(int(pt[0] * scale_x), 0), W_f - 1)
            qy = min(max(int(pt[1] * scale_y), 0), H_f - 1)
            
            # Attention
            ax_attn = axes[i, 2 + m_idx * 2]
            if res["attn_weights"] is not None:
                q_idx = qy * W_f + qx
                # (Heads, N) -> (N,)
                attn_map = res["attn_weights"][:, q_idx, :].mean(dim=0)
                attn_map_2d = attn_map.reshape(H_f, W_f).numpy()
                attn_vis = torch.from_numpy(attn_map_2d).float().unsqueeze(0).unsqueeze(0)
                attn_vis = F.interpolate(attn_vis, size=(H_img, W_img), mode="bilinear", align_corners=False).squeeze().numpy()
                attn_vis = robust_normalize(attn_vis)
                ax_attn.imshow(attn_vis, cmap="inferno", interpolation="bicubic")
                ax_attn.plot(pt[0], pt[1], 'gx', markersize=10, markeredgewidth=2)
            ax_attn.axis("off")
            
            # Cluster
            ax_clust = axes[i, 2 + m_idx * 2 + 1]
            c_label = res["cluster_map"][qy, qx]
            c_mask = (res.get("cluster_upsampled") if res.get("cluster_upsampled") is not None else res["cluster_map"]) == c_label
            ax_clust.imshow(c_mask, cmap="gray", interpolation="nearest")
            ax_clust.plot(pt[0], pt[1], 'rx', markersize=10, markeredgewidth=2)
            ax_clust.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    out_path = output_dir / f"compare_idx{idx:04d}.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    # print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoralMonSter Feature Comparison")
    # Standard args
    parser.add_argument("--dataset", type=str, default="hkcoral")
    parser.add_argument("--dataset_root", type=str, default="data_storage/HKCoral")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to SAM baseline")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--gpu", type=int, default=-1)
    
    # Comparison args
    parser.add_argument("--compare_checkpoints", nargs="+", help="List of other checkpoints to compare")
    parser.add_argument("--timm_models", nargs="+", help="List of timm model names (e.g., vit_base_patch14_dinov2.lvd142m)")
    parser.add_argument("--hf_dinov3_models", nargs="+", help="List of HuggingFace DINOv3 model ids (e.g., facebook/dinov3-base)" )
    parser.add_argument("--image_index", type=int, default=-1, help="Specific image index")
    parser.add_argument("--num_images", type=int, default=5, help="Number of random images to process if index is -1")
    parser.add_argument("--process_all", action="store_true", help="Process the entire split")
    parser.add_argument("--output_dir", type=str, default="vis_results")
    
    # Dummy args for config builder
    parser.add_argument("--scenario_name", type=str, default="vis")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=["qkv"])
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--scenario_preset", type=str, default=None)

    args = parser.parse_args()
    visualize_dataset(args)
