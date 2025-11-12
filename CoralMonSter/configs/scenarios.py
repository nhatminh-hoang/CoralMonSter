"""
Scenario presets for toggling major training options.
"""

from __future__ import annotations

from typing import Dict

from CoralMonSter.configs.hkcoral_config import HKCoralConfig


SCENARIO_PRESETS: Dict[str, Dict[str, object]] = {
    "full": {
        "description": "Full distillation: scheduler, centering, frozen encoder, EMA teacher",
        "use_lr_scheduler": True,
        "freeze_image_encoder": True,
        "use_teacher_momentum": True,
        "center_momentum": 0.99,
    },
    "no_scheduler": {
        "description": "Disable LR scheduler",
        "use_lr_scheduler": False,
    },
    "no_centering": {
        "description": "Disable DINO-style centering",
        "center_momentum": 0.0,
    },
    "unfrozen_encoder": {
        "description": "Fine-tune image encoder (no freezing)",
        "freeze_image_encoder": False,
    },
    "no_momentum": {
        "description": "Disable EMA teacher updates",
        "use_teacher_momentum": False,
    },
    "momentum_skip_one": {
        "description": "Delay EMA/distillation until after epoch 1",
        "momentum_skip_epochs": 1,
    },
}


def apply_scenario_preset(cfg: HKCoralConfig, preset_name: str) -> HKCoralConfig:
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

    return cfg


__all__ = ["SCENARIO_PRESETS", "apply_scenario_preset"]
