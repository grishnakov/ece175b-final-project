"""Exponential moving average (EMA) of model parameters."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self._backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.mul_(d).add_(m_p.detach(), alpha=1.0 - d)
        for s_b, m_b in zip(self.shadow.buffers(), model.buffers()):
            s_b.copy_(m_b)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow.state_dict())

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self._backup is None:
            raise RuntimeError("restore() called without a matching apply_to()")
        model.load_state_dict(self._backup)
        self._backup = None

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow.state_dict()}

    def load_state_dict(self, state: dict):
        self.decay = state["decay"]
        self.shadow.load_state_dict(state["shadow"])
