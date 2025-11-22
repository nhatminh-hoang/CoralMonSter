from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from typing import Optional, Union

from CoralMonSter import CoralMonSter as CoralModel
from CoralMonSter import HKCoralConfig, CoralScapesConfig
from CoralMonSter.data import CoralScapesDataset, HKCoralDataset, hkcoral_collate_fn

ConfigType = Union[HKCoralConfig, CoralScapesConfig]

def build_model(cfg: ConfigType, device: Optional[torch.device] = None) -> CoralModel:
    """Build and optionally move model to device."""
    model = CoralModel(cfg)
    if device:
        model = model.to(device)
    return model

def build_dataset(cfg: ConfigType, split: str, dataset_choice: str = "hkcoral"):
    """Build dataset based on config and split."""
    if dataset_choice == "coralscapes":
        hf_split = {"val": "validation"}.get(split, split)
        return CoralScapesDataset(
            split=hf_split,
            image_size=cfg.image_size,
            num_classes=cfg.num_classes,
            ignore_label=cfg.ignore_label,
            prompt_points=cfg.distillation.prompt_points,
            prompt_bins=cfg.prompt_bins,
            mean=cfg.image_mean,
            std=cfg.image_std,
            dataset_id=getattr(cfg, "dataset_id", "EPFL-ECEO/coralscapes"),
            cache_dir=getattr(cfg, "dataset_cache_dir", None) or (cfg.dataset_root / "cache" if cfg.dataset_root else None),
            hf_token=getattr(cfg, "hf_token", None),
        )
    
    return HKCoralDataset(
        cfg.dataset_root,
        split,
        cfg.image_size,
        cfg.num_classes,
        cfg.ignore_label,
        cfg.distillation.prompt_points, # Accessing directly from cfg structure
        cfg.prompt_bins,
        cfg.image_mean,
        cfg.image_std,
    )

def prepare_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    limit: Optional[int] = None
) -> DataLoader:
    """Prepare dataloader with optional subset limiting."""
    if limit is not None and limit > 0:
        count = min(limit, len(dataset))
        dataset = Subset(dataset, list(range(count)))
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=hkcoral_collate_fn,
    )
