"""
Configuration dataclasses for the HKCoral-focused CoralMonSter pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class OptimizationConfig:
    """Training hyper-parameters shared by the supervised and distillation losses."""

    batch_size: int = 2
    num_workers: int = 4
    max_epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    ema_momentum_min: float = 0.996
    ema_momentum_max: float = 1.0
    grad_clip_norm: float = 1.0


@dataclass
class DistillationConfig:
    """
    Loss weights and sampling knobs for the momentum-based knowledge distillation
    between the teacher and the student mask decoders.
    """

    dice_weight: float = 1.0
    ce_weight: float = 0.5
    mask_kd_weight: float = 1.0
    token_kd_weight: float = 1.0
    temperature: float = 1.0
    student_temperature: float = 0.1
    teacher_temperature_start: float = 0.04
    teacher_temperature_end: float = 0.07
    teacher_temperature_warmup_epochs: int = 30
    prompt_points: int = 8
    prompt_use_box: bool = True


@dataclass
class HKCoralConfig:
    """
    End-to-end configuration for HKCoral semantic segmentation.

    Attributes:
        dataset_root: Root directory of the HKCoral dataset.
        split: Dataset split to operate on (train/val/test).
        image_size: Spatial resolution fed to SAM's image encoder.
        num_classes: Number of semantic classes (including background).
        model_type: SAM backbone identifier.
        sam_checkpoint: Path to the SAM checkpoint that bootstraps the teacher.
        freeze_image_encoder: Whether to keep SAM's image encoder frozen.
        freeze_prompt_encoder: Whether to keep SAM's prompt encoder frozen.
        save_dir: Directory where checkpoints, logs, etc. are stored.
    """

    dataset_root: Path = Path("datasets/HKCoral")
    split: str = "train"
    image_size: int = 1024
    num_classes: int = 7  # background + six growth forms
    ignore_label: int = 255
    model_type: str = "vit_b"
    sam_checkpoint: Path = Path("checkpoints/sam_vit_b_coralscop.pth")
    freeze_image_encoder: bool = True
    freeze_prompt_encoder: bool = True
    scenario_name: str = "default"
    save_dir: Path = Path("checkpoints")
    log_root: Path = Path("logs")
    checkpoint_root: Path = Path("checkpoints")
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    prompt_bins: Tuple[int, ...] = (1, 2, 4, 10)

    def resolve_paths(self) -> "HKCoralConfig":
        """Expand user paths to absolute ``Path`` objects for downstream code."""
        self.dataset_root = Path(self.dataset_root).expanduser().resolve()
        self.sam_checkpoint = Path(self.sam_checkpoint).expanduser().resolve()
        self.log_root = Path(self.log_root).expanduser().resolve()
        self.checkpoint_root = Path(self.checkpoint_root).expanduser().resolve()
        self.log_dir = self.log_root / self.scenario_name
        self.save_dir = self.checkpoint_root / self.scenario_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def image_mean(self) -> Tuple[float, float, float]:
        # SAM's normalization constants.
        return (123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0)

    @property
    def image_std(self) -> Tuple[float, float, float]:
        return (58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0)

    @property
    def class_palette(self) -> List[Tuple[int, int, int]]:
        return [
            (0, 0, 0),          # background
            (255, 0, 0),        # massive
            (255, 165, 0),      # encrusting
            (255, 255, 0),      # branching
            (0, 128, 0),        # laminar
            (0, 191, 255),      # folliaceous
            (138, 43, 226),     # columnar
        ]
