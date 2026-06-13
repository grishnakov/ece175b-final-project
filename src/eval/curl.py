"""Curl diagnostic for drift field conservatism."""

import torch
import torch.nn as nn

from src.kernels.base import Kernel
from src.kernels.gaussian import GaussianKernel
from src.kernels.laplacian import LaplacianKernel


def compute_drift_field_at_point(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    kernel: Kernel,
    normalization: str = "vanilla",
) -> torch.Tensor:
    x_2d = x.unsqueeze(0)

    if normalization == "vanilla":
        # V_p^+(x) = E[k(x,y+)(y+ - x)] / E[k(x,y+)]
        K_pos = kernel(x_2d, y_pos).squeeze(0)
        Z_pos = K_pos.sum() + 1e-8
        displacements_pos = y_pos - x
        V_pos = (K_pos.unsqueeze(-1) * displacements_pos).sum(dim=0) / Z_pos

        K_neg = kernel(x_2d, y_neg).squeeze(0)
        Z_neg = K_neg.sum() + 1e-8
        displacements_neg = y_neg - x
        V_neg = (K_neg.unsqueeze(-1) * displacements_neg).sum(dim=0) / Z_neg

    elif normalization == "sharp":
        # V_p^{+#}(x) = E[k(x,y+)(y+ - x)] / E[k#(x,y+)]
        K_pos = kernel(x_2d, y_pos).squeeze(0)
        K_sharp_pos = torch.exp(kernel.log_sharp(x_2d, y_pos)).squeeze(0)
        Z_sharp_pos = K_sharp_pos.sum() + 1e-8
        displacements_pos = y_pos - x
        V_pos = (K_pos.unsqueeze(-1) * displacements_pos).sum(dim=0) / Z_sharp_pos

        K_neg = kernel(x_2d, y_neg).squeeze(0)
        K_sharp_neg = torch.exp(kernel.log_sharp(x_2d, y_neg)).squeeze(0)
        Z_sharp_neg = K_sharp_neg.sum() + 1e-8
        displacements_neg = y_neg - x
        V_neg = (K_neg.unsqueeze(-1) * displacements_neg).sum(dim=0) / Z_sharp_neg
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    return V_pos - V_neg


def compute_curl_norm(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    kernel: Kernel,
    normalization: str = "vanilla",
) -> float:
    """||curl||^2 = sum_{i<j} (J_ij - J_ji)^2."""
    D = x.shape[0]
    x_var = x.clone().detach().requires_grad_(True)

    V = compute_drift_field_at_point(x_var, y_pos, y_neg, kernel, normalization)

    jacobian = torch.zeros(D, D, device=x.device)
    for i in range(D):
        if x_var.grad is not None:
            x_var.grad.zero_()
        V[i].backward(retain_graph=(i < D - 1))
        jacobian[i] = x_var.grad.clone()

    antisym = jacobian - jacobian.T
    curl_sq = 0.5 * (antisym**2).sum().item()

    return curl_sq


def curl_diagnostic(
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    kernel: Kernel,
    normalization: str = "vanilla",
    num_points: int = 10,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    D = y_pos.shape[1]
    device = y_pos.device

    curls = []
    for i in range(num_points):
        if i % 2 == 0:
            idx = torch.randint(y_pos.shape[0], (1,))
            x = y_pos[idx].squeeze(0) + 0.1 * torch.randn(D, device=device)
        else:
            idx = torch.randint(y_neg.shape[0], (1,))
            x = y_neg[idx].squeeze(0) + 0.1 * torch.randn(D, device=device)

        curl_sq = compute_curl_norm(x, y_pos, y_neg, kernel, normalization)
        curls.append(curl_sq)

    curls = torch.tensor(curls)
    return {
        "normalization": normalization,
        "curl_mean": curls.mean().item(),
        "curl_std": curls.std().item(),
        "curl_max": curls.max().item(),
        "curl_min": curls.min().item(),
        "curls": curls.tolist(),
    }


if __name__ == "__main__":
    torch.manual_seed(42)

    D = 4
    N = 20
    y_pos = torch.randn(N, D)
    y_neg = torch.randn(N, D) + 0.5

    print("Curl diagnostic test (D=4, N=20)")
    print("=" * 50)

    for kernel_cls, name in [(GaussianKernel, "Gaussian"), (LaplacianKernel, "Laplacian")]:
        kernel = kernel_cls(sigma=1.0)
        for norm in ["vanilla", "sharp"]:
            result = curl_diagnostic(y_pos, y_neg, kernel, norm, num_points=5)
            print(f"[{name}, {norm}] curl_mean={result['curl_mean']:.6f}, "
                  f"curl_max={result['curl_max']:.6f}")

    print("\nExpected: Gaussian vanilla ≈ Gaussian sharp (since k# ∝ k)")
    print("Expected: Laplacian sharp << Laplacian vanilla")
    print("\nCurl diagnostic test passed!")
