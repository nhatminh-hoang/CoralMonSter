"""Configuration helpers for CoralScapes training."""

from __future__ import annotations

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from huggingface_hub import hf_hub_download

from .hkcoral_config import HKCoralConfig


@dataclass
class CoralScapesConfig(HKCoralConfig):
    """HKCoral configuration variant with CoralScapes defaults and metadata hooks."""

    dataset_root: Path = Path("datasets/CoralScapes")
    dataset_cache_dir: Optional[Path] = None
    dataset_id: str = "EPFL-ECEO/coralscapes"
    hf_token: Optional[str] = None
    num_classes: int = 40
    eval_ignore_classes: Tuple[int, ...] = (0,)
    _class_names: Optional[List[str]] = field(default=None, init=False, repr=False)
    _class_palette: Optional[List[Tuple[int, int, int]]] = field(default=None, init=False, repr=False)

    def resolve_paths(self) -> "CoralScapesConfig":  # type: ignore[override]
        super().resolve_paths()
        if self.dataset_cache_dir is None:
            self.dataset_cache_dir = self.dataset_root / "cache"
        self.dataset_cache_dir = Path(self.dataset_cache_dir).expanduser().resolve()
        self._ensure_metadata()
        return self

    def _ensure_metadata(self) -> None:
        if self._class_names is not None and self._class_palette is not None:
            return
        try:
            cache_dir = str(self.dataset_cache_dir)
            id2label_path = "/home/21013299/Minh/CoralMonSter/datasets/CoralScapes/cache/datasets--EPFL-ECEO--coralscapes/snapshots/acd72c757c59db055b800dc054a2291487e4566f/id2label.json"
            label2color_path = "/home/21013299/Minh/CoralMonSter/datasets/CoralScapes/cache/datasets--EPFL-ECEO--coralscapes/snapshots/acd72c757c59db055b800dc054a2291487e4566f/label2color.json"
            with open(id2label_path, "r", encoding="utf-8") as fp:
                id2label_raw = {int(k): v for k, v in json.load(fp).items()}
            id2label_raw.setdefault(0, "background")
            max_idx = max(id2label_raw.keys())
            with open(label2color_path, "r", encoding="utf-8") as fp:
                label2color = json.load(fp)
            label2color.setdefault("background", [0, 0, 0])
            class_names: List[str] = []
            palette: List[Tuple[int, int, int]] = []
            for idx in range(max_idx + 1):
                name = id2label_raw.get(idx, f"class_{idx}")
                class_names.append(name)
                rgb = label2color.get(name, [0, 0, 0])
                palette.append(tuple(int(c) for c in rgb))
            self._class_names = class_names
            self._class_palette = palette
            self.num_classes = len(class_names)
        except Exception as exc:  # pragma: no cover - fallback path
            print(
                "[Warning] Failed to download CoralScapes metadata. Using placeholder class names."
            )
            print(exc)
            self._class_names = ["background"] + [f"class_{i}" for i in range(1, self.num_classes)]
            self._class_palette = [(0, 0, 0)] + [(i * 5 % 255, i * 15 % 255, i * 25 % 255) for i in range(1, self.num_classes)]

    @property
    def class_names(self) -> List[str]:  # type: ignore[override]
        self._ensure_metadata()
        return self._class_names or []

    @property
    def class_palette(self) -> List[Tuple[int, int, int]]:  # type: ignore[override]
        self._ensure_metadata()
        return self._class_palette or []


__all__ = ["CoralScapesConfig"]
