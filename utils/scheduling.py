"""
Schedule functions for CoralMonSter per-epoch training.

All schedules are per-epoch (not per-step), as requested.

  - sinusoidal_schedule()     — EMA momentum (α) from 0.996 → 1
  - teacher_temp_schedule()   — teacher temperature (τ_t) from 0.04 → 0.07
  - cosine_lr_lambda()        — LR: linear warmup → cosine decay to 0
"""

from __future__ import annotations

import math
from typing import Callable


def sinusoidal_schedule(
    epoch: int,
    total_epochs: int,
    start_value: float,
    end_value: float,
) -> float:
    """
    Sinusoidal schedule from start_value → end_value over total_epochs.

    Uses sin(π/2 · progress) to ramp from start to end.
    This matches the paper's "sinusoidal update" for EMA momentum (Table 1).
    Also used for teacher temperature (τ_t).

    Args:
        epoch:        current epoch (0-indexed)
        total_epochs: total number of training epochs
        start_value:  value at epoch 0  (e.g. 0.996 for momentum)
        end_value:    value at final epoch (e.g. 1.0 for momentum)

    Returns:
        Interpolated value for this epoch.
    """
    if total_epochs <= 1:
        return end_value

    progress = min(epoch / (total_epochs - 1), 1.0)
    # sin(0) = 0, sin(π/2) = 1 → smooth ramp from start to end
    t = math.sin(math.pi / 2.0 * progress)
    return start_value + (end_value - start_value) * t


def cosine_lr_lambda(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
) -> float:
    """
    LR multiplier for cosine annealing with linear warmup.

    Paper Table 1: "Linear warmup from 0 to 1e-4 then sinusoidal update to 0"

    Phase 1 (epoch < warmup_epochs):
        lr_mult = epoch / warmup_epochs  (linear ramp 0 → 1)

    Phase 2 (epoch >= warmup_epochs):
        lr_mult = 0.5 * (1 + cos(π · (epoch - warmup) / (total - warmup)))
        This decays from 1 → 0 following a cosine curve.

    Usage:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda e: cosine_lr_lambda(e, ...))

    Args:
        epoch:         current epoch (0-indexed)
        total_epochs:  total training epochs
        warmup_epochs: number of warmup epochs

    Returns:
        Multiplier in [0, 1] to apply to base LR.
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return epoch / max(warmup_epochs, 1)
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_lr_lambda(
    total_epochs: int,
    warmup_epochs: int,
) -> Callable[[int], float]:
    """
    Build a LR lambda function for use with torch.optim.lr_scheduler.LambdaLR.

    Returns:
        A callable that maps epoch → lr_multiplier.
    """
    def lr_lambda(epoch: int) -> float:
        return cosine_lr_lambda(epoch, total_epochs, warmup_epochs)
    return lr_lambda
