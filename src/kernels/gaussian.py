"""Gaussian kernel and its sharp variant.

k(x,y)  = exp(-||x-y||^2 / (2*sigma^2))
k#(x,y) = sigma^2 * k(x,y)
"""

import torch

from .base import Kernel, pairwise_sq_distances


class GaussianKernel(Kernel):
    """k(x,y)  = exp(-||x-y||^2 / (2*sigma^2))
    k#(x,y) = sigma^2 * exp(-||x-y||^2 / (2*sigma^2))
    """

    def __init__(self, sigma: float = 1.0):
        super().__init__(sigma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist_sq = pairwise_sq_distances(x, y)
        return torch.exp(-dist_sq / (2 * self.sigma**2))

    def log_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """log k(x,y) = -||x-y||^2 / (2*sigma^2)."""
        dist_sq = pairwise_sq_distances(x, y)
        return -dist_sq / (2 * self.sigma**2)

    def log_sharp(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """log k#(x,y) = log(sigma^2) - ||x-y||^2 / (2*sigma^2)."""
        return torch.log(torch.tensor(self.sigma**2, device=x.device)) + self.log_kernel(x, y)


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(5, 3)
    y = torch.randn(4, 3)

    kernel = GaussianKernel(sigma=1.0)

    K = kernel(x, y)
    log_K = kernel.log_kernel(x, y)
    log_Ks = kernel.log_sharp(x, y)

    print(f"k(x,y) shape: {K.shape}, range: [{K.min():.4f}, {K.max():.4f}]")
    print(f"log k(x,y) shape: {log_K.shape}")
    print(f"log k#(x,y) shape: {log_Ks.shape}")

    assert torch.allclose(K, torch.exp(log_K), atol=1e-6), "log_kernel inconsistent!"

    Ks_from_log = torch.exp(log_Ks)
    Ks_expected = kernel.sigma**2 * K
    assert torch.allclose(Ks_from_log, Ks_expected, atol=1e-6), "sharp kernel inconsistent!"

    print("All checks passed!")
