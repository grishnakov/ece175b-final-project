"""Base kernel interface and distance utilities."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Kernel(ABC, nn.Module):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def log_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def log_sharp(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """nabla_x k#(x,y) = k(x,y)(y - x)."""
        ...


def pairwise_sq_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """||x_i - y_j||^2 = ||x_i||^2 - 2<x_i,y_j> + ||y_j||^2."""
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    dist_sq = x_sq - 2 * x @ y.T + y_sq.T
    return dist_sq.clamp(min=0)


def pairwise_distances(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """||x_i - y_j|| = sqrt(||x_i - y_j||^2 + eps)."""
    return torch.sqrt(pairwise_sq_distances(x, y) + eps)
