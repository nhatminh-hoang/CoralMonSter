"""
CoralMonSter package
====================

This package hosts the prompt-free coral segmentation framework that builds on
top of SAM. All new logic (data loading, model definitions, training helpers,
etc.) lives under this namespace so the upstream `segment_anything` sources can
stay untouched while still being vendored for convenience.
"""

import sys

from .configs.hkcoral_config import HKCoralConfig, DistillationConfig, OptimizationConfig
from .configs.coralscapes_config import CoralScapesConfig
from .configs.base_config import BaseCoralConfig, LoraConfig
from .models.coral_monster import CoralMonSter
from .engine.trainer import CoralTrainer
from . import segment_anything as _segment_anything_module

# Keep backward compatibility with Meta's import path.
sys.modules.setdefault("segment_anything", _segment_anything_module)

__all__ = [
    "BaseCoralConfig",
    "HKCoralConfig",
    "CoralScapesConfig",
    "DistillationConfig",
    "OptimizationConfig",
    "CoralMonSter",
        "LoraConfig",
    "CoralTrainer",
]
