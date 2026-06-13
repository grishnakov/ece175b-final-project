"""Learning-rate schedule: linear warmup followed by cosine decay.

    lr(t) = lr_base * t / warmup_steps                           (t < warmup)
    lr(t) = lr_min + 0.5*(lr_base - lr_min)*(1 + cos(pi * prog))  (otherwise)
"""

from __future__ import annotations

import math

import torch


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 1000,
    lr_min: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    base_lr = optimizer.param_groups[0]["lr"]
    min_ratio = (lr_min / base_lr) if base_lr > 0 else 0.0
    decay_steps = max(total_steps - warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / float(max(warmup_steps, 1))
        progress = (step - warmup_steps) / float(decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
