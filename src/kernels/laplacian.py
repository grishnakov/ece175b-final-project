"""Laplacian kernel and its sharp variant.

k(x,y)  = exp(-||x-y|| / sigma)
k#(x,y) = sigma * (||x-y|| + sigma) * exp(-||x-y|| / sigma)
"""

import math
import torch

from .base import Kernel, pairwise_distances


class LaplacianKernel(Kernel):
    """k(x,y)  = exp(-||x-y|| / sigma)
    k#(x,y) = sigma * (||x-y|| + sigma) * exp(-||x-y|| / sigma)
    log k#(x,y) = log(sigma) + log(||x-y|| + sigma) - ||x-y|| / sigma
    """

    def __init__(self, sigma: float = 0.1):
        super().__init__(sigma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = pairwise_distances(x, y)
        return torch.exp(-dist / self.sigma)

    def log_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """log k(x,y) = -||x-y|| / sigma."""
        dist = pairwise_distances(x, y)
        return -dist / self.sigma

    def log_sharp(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """log k#(x,y) = log(sigma) + log(||x-y|| + sigma) - ||x-y|| / sigma."""
        dist = pairwise_distances(x, y)
        log_sigma = math.log(self.sigma)
        return log_sigma + torch.log(dist + self.sigma) - dist / self.sigma


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(5, 3)
    y = torch.randn(4, 3)

    kernel = LaplacianKernel(sigma=0.5)

    K = kernel(x, y)
    log_K = kernel.log_kernel(x, y)
    log_Ks = kernel.log_sharp(x, y)

    print(f"k(x,y) shape: {K.shape}, range: [{K.min():.4f}, {K.max():.4f}]")
    print(f"log k(x,y) shape: {log_K.shape}")
    print(f"log k#(x,y) shape: {log_Ks.shape}")

    assert torch.allclose(K, torch.exp(log_K), atol=1e-5), "log_kernel inconsistent!"

    dist = pairwise_distances(x, y)
    Ks_from_log = torch.exp(log_Ks)
    Ks_expected = kernel.sigma * (dist + kernel.sigma) * K
    assert torch.allclose(Ks_from_log, Ks_expected, atol=1e-5), "sharp kernel inconsistent!"

    x_self = torch.randn(3, 3)
    log_Ks_self = kernel.log_sharp(x_self, x_self)
    diag_vals = torch.exp(torch.diagonal(log_Ks_self))
    expected = kernel.sigma**2
    print(f"k#(x,x) diagonal: {diag_vals}, expected: {expected:.4f}")
    assert torch.allclose(diag_vals, torch.full_like(diag_vals, expected), atol=1e-3), \
        "Self-kernel check failed!"

    print("All checks passed!")
