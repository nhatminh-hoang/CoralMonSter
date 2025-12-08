"""
Dataset-agnostic configuration primitives for CoralMonSter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class OptimizationConfig:
    """Training hyper-parameters shared by supervised and distillation losses."""

    batch_size: int = 2
    num_workers: int = 4
    max_epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    ema_momentum_min: float = 0.996
    ema_momentum_max: float = 1.0
    grad_clip_norm: float = 1.0
    use_lr_scheduler: bool = True
    lr_warmup_epochs: int = 10
    lr_min_factor: float = 0.1
    use_teacher_momentum: bool = True
    momentum_skip_epochs: int = 1
    # SAM optimizer settings
    use_sam_optimizer: bool = False
    sam_rho: float = 2.0
    sam_adaptive: bool = False


@dataclass
class DistillationConfig:
    """Loss weights and sampling knobs for teacher-student distillation."""

    dice_weight: float = 1.5
    ce_weight: float = 1.0
    mask_kd_weight: float = 0.0
    token_kd_weight: float = 1.0
    token_kd_metric: str = "ce"
    token_classification_weight: float = 1.0
    student_temperature: float = 0.1
    teacher_temperature_start: float = 0.04
    teacher_temperature_end: float = 0.07
    teacher_temperature_warmup_epochs: int = 30
    center_momentum: float = 0.99
    prompt_points: int = 8
    prompt_use_box: bool = True


@dataclass
class LoraConfig:
    """LoRA tuning knobs kept separate for clarity."""

    enabled: bool = False
    r: int = 4
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["qkv"])


@dataclass
class BaseCoralConfig:
    """
    Core configuration for CoralMonSter training/evaluation.
    Subclasses can override dataset defaults (root, num_classes, metadata).
    """

    dataset_root: Path = Path("data_storage/HKCoral")
    split: str = "train"
    image_size: int = 256
    num_classes: int = 7  # background + six growth forms
    ignore_label: int = 255
    model_type: str = "vit_b"
    sam_checkpoint: Path = Path("checkpoints/sam_vit_b_coralscop.pth")
    sam_random_init: bool = False
    freeze_image_encoder: bool = True
    freeze_prompt_encoder: bool = False
    use_gradient_checkpointing: bool = False
    use_flash_attention: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)
    scenario_name: str = "default"
    save_dir: Path = Path("checkpoints")
    log_root: Path = Path("logs")
    checkpoint_root: Path = Path("checkpoints")
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    prompt_bins: Tuple[int, ...] = (1, 2, 4, 10)
    eval_ignore_classes: Tuple[int, ...] = ()
    enable_pca_logging: bool = False
    pca_samples_per_epoch: int = 2
    pca_log_dirname: str = "encoder_pca"

    def resolve_paths(self) -> "BaseCoralConfig":
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
        # SAM normalization constants.
        return (123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0)

    @property
    def image_std(self) -> Tuple[float, float, float]:
        return (58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0)

    @property
    def class_palette(self) -> List[Tuple[int, int, int]]:
        return [
            (0, 0, 0),
            (255, 0, 0),
            (255, 165, 0),
            (255, 255, 0),
            (0, 128, 0),
            (0, 191, 255),
            (138, 43, 226),
        ]

    @property
    def class_names(self) -> List[str]:
        return [
            "background",
            "massive",
            "encrusting",
            "branching",
            "laminar",
            "folliaceous",
            "columnar",
        ]

    # Compatibility shims for legacy attributes
    @property
    def use_lora(self) -> bool:
        return bool(self.lora.enabled)

    @use_lora.setter
    def use_lora(self, value: bool) -> None:
        self.lora.enabled = bool(value)

    @property
    def lora_r(self) -> int:
        return self.lora.r

    @lora_r.setter
    def lora_r(self, value: int) -> None:
        self.lora.r = int(value)

    @property
    def lora_alpha(self) -> int:
        return self.lora.alpha

    @lora_alpha.setter
    def lora_alpha(self, value: int) -> None:
        self.lora.alpha = int(value)

    @property
    def lora_dropout(self) -> float:
        return self.lora.dropout

    @lora_dropout.setter
    def lora_dropout(self, value: float) -> None:
        self.lora.dropout = float(value)

    @property
    def lora_target_modules(self) -> List[str]:
        return self.lora.target_modules

    @lora_target_modules.setter
    def lora_target_modules(self, value: List[str]) -> None:
        self.lora.target_modules = list(value)


__all__ = [
    "BaseCoralConfig",
    "OptimizationConfig",
    "DistillationConfig",
    "LoraConfig",
]
