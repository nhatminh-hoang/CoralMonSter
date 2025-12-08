"""
HKCoral-specific configuration built on shared CoralMonSter primitives.
"""

from dataclasses import dataclass

from .base_config import BaseCoralConfig, OptimizationConfig, DistillationConfig


@dataclass
class HKCoralConfig(BaseCoralConfig):
    """End-to-end configuration for HKCoral semantic segmentation."""

    pass
