"""
Simple checkpoint helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(model: torch.nn.Module, path: Path, strict: bool = True) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=strict)
    if missing or unexpected:
        print(f"Missing keys: {missing}, Unexpected keys: {unexpected}")
    return checkpoint
