"""
Scenario presets for toggling major training options.
"""

from __future__ import annotations

from typing import Dict

from CoralMonSter.configs.base_config import BaseCoralConfig


SCENARIO_PRESETS: Dict[str, Dict[str, object]] = {
    "full": {
        "description": "Full distillation: scheduler, centering, frozen encoder, EMA teacher",
        "use_lr_scheduler": True,
        "freeze_image_encoder": False,
        "use_teacher_momentum": True,
        "center_momentum": 0.99,
    },
    "no_scheduler": {
        "description": "Disable LR scheduler",
        "use_lr_scheduler": False,
        "freeze_image_encoder": True,
        "use_teacher_momentum": True,
        "center_momentum": 0.99,
    },
    "no_centering": {
        "description": "Disable DINO-style centering",
        "center_momentum": 0.0,
        "freeze_image_encoder": False,
        "use_teacher_momentum": True,
    },
    "frozen_encoder": {
        "description": "Fine-tune image encoder (no freezing)",
        "freeze_image_encoder": True,
        "use_teacher_momentum": True,
        "center_momentum": 0.99,
    },
    "no_momentum": {
        "description": "Disable EMA teacher updates",
        "use_teacher_momentum": False,
        "freeze_image_encoder": True,
        "center_momentum": 0.99,
        "token_kd_weight": 0.0,
    },
    "momentum_no_skip": {
        "description": "Start EMA/distillation immediately (no warm-up)",
        "momentum_skip_epochs": 0,
        "freeze_image_encoder": False,
        "use_teacher_momentum": True,
        "center_momentum": 0.99,
    },
    "token_kd_kl": {
        "description": "Use KL divergence instead of cross-entropy for token distillation",
        "token_kd_metric": "kl",
        "freeze_image_encoder": False,
        "use_teacher_momentum": True,
        "center_momentum": 0.99,
    },
    "lora_encoder": {
        "description": "LoRA fine-tuning of image encoder",
        "freeze_image_encoder": False,
        "use_lora": True,
        "use_teacher_momentum": True,
        "center_momentum": 0.99,
    },
    "sam_optimizer": {
        "description": "SAM optimizer with AdamW as base optimizer for sharpness-aware minimization",
        "use_sam_optimizer": True,
        "sam_rho": 2.0,
        "sam_adaptive": False,
        "freeze_image_encoder": True,
        "use_teacher_momentum": True,
        "center_momentum": 0.99,
    },
    "sam_optimizer_adaptive": {
        "description": "Adaptive SAM optimizer with AdamW as base optimizer",
        "use_sam_optimizer": True,
        "sam_rho": 2.0,
        "sam_adaptive": True,
        "freeze_image_encoder": True,
        "use_teacher_momentum": True,
        "center_momentum": 0.99,
    },
}


def apply_scenario_preset(cfg: BaseCoralConfig, preset_name: str) -> BaseCoralConfig:
    preset = SCENARIO_PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(f"Unknown scenario preset '{preset_name}'")

    if "use_lr_scheduler" in preset:
        cfg.optimization.use_lr_scheduler = bool(preset["use_lr_scheduler"])
    if "freeze_image_encoder" in preset:
        cfg.freeze_image_encoder = bool(preset["freeze_image_encoder"])
    if "use_teacher_momentum" in preset:
        cfg.optimization.use_teacher_momentum = bool(preset["use_teacher_momentum"])
    if "center_momentum" in preset:
        cfg.distillation.center_momentum = float(preset["center_momentum"])
    if "momentum_skip_epochs" in preset:
        cfg.optimization.momentum_skip_epochs = int(preset["momentum_skip_epochs"])
    if "token_kd_metric" in preset:
        cfg.distillation.token_kd_metric = str(preset["token_kd_metric"])
    if "token_kd_weight" in preset:
        cfg.distillation.token_kd_weight = float(preset["token_kd_weight"])
        cfg.distillation.mask_kd_weight = float(preset["token_kd_weight"])
    if "use_lora" in preset:
        cfg.use_lora = bool(preset["use_lora"])
    if "use_sam_optimizer" in preset:
        cfg.optimization.use_sam_optimizer = bool(preset["use_sam_optimizer"])
    if "sam_rho" in preset:
        cfg.optimization.sam_rho = float(preset["sam_rho"])
    if "sam_adaptive" in preset:
        cfg.optimization.sam_adaptive = bool(preset["sam_adaptive"])

    return cfg


__all__ = ["SCENARIO_PRESETS", "apply_scenario_preset"]
