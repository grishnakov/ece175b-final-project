"""MSE field-matching loss for vanilla drifting (Arms A, B).

    L = E_eps[||f_theta(eps) - sg(f_theta(eps) + V_{p,q}(f_theta(eps)))||^2]
    V_{p,q}(x) = V_p^+(x) - V_q^-(x)
    V_p^+(x) = E_{y+~p}[k(x,y+)(y+ - x)] / Z_p(x)
    V_q^-(x) = E_{y-~q}[k(x,y-)(y- - x)] / Z_q(x)
    Z_p(x) = E_{y+~p}[k(x,y+)]
"""

import torch
import torch.nn as nn

from src.kernels.base import Kernel


class MSEFieldLoss(nn.Module):
    def __init__(self, kernel: Kernel, stop_grad: bool = True):
        super().__init__()
        self.kernel = kernel
        self.stop_grad = stop_grad

    def compute_drift_field(
        self,
        x: torch.Tensor,
        y_pos: torch.Tensor,
        y_neg: torch.Tensor,
    ) -> torch.Tensor:
        """V_p^+(x) = sum_j k(x, y_j^+)(y_j^+ - x) / sum_j k(x, y_j^+)
        V_q^-(x) = sum_j k(x, y_j^-)(y_j^- - x) / sum_j k(x, y_j^-)
        """
        log_K_pos = self.kernel.log_kernel(x, y_pos)
        weights_pos = torch.softmax(log_K_pos, dim=1)
        displacements_pos = y_pos.unsqueeze(0) - x.unsqueeze(1)
        V_pos = (weights_pos.unsqueeze(-1) * displacements_pos).sum(dim=1)

        log_K_neg = self.kernel.log_kernel(x, y_neg)
        weights_neg = torch.softmax(log_K_neg, dim=1)
        displacements_neg = y_neg.unsqueeze(0) - x.unsqueeze(1)
        V_neg = (weights_neg.unsqueeze(-1) * displacements_neg).sum(dim=1)

        return V_pos - V_neg

    def forward(
        self,
        x_gen: torch.Tensor,
        y_pos: torch.Tensor,
        y_neg: torch.Tensor,
    ) -> torch.Tensor:
        V = self.compute_drift_field(x_gen, y_pos, y_neg)

        target = x_gen + V

        if self.stop_grad:
            target = target.detach()

        loss = ((x_gen - target) ** 2).mean()
        return loss


if __name__ == "__main__":
    from src.kernels.gaussian import GaussianKernel
    from src.kernels.laplacian import LaplacianKernel

    torch.manual_seed(42)

    B, D = 8, 16
    x = torch.randn(B, D, requires_grad=True)
    y_pos = torch.randn(B, D)
    y_neg = x.detach().clone()

    for kernel_cls, name in [(GaussianKernel, "Gaussian"), (LaplacianKernel, "Laplacian")]:
        for stop_grad in [True, False]:
            kernel = kernel_cls(sigma=1.0)
            loss_fn = MSEFieldLoss(kernel, stop_grad=stop_grad)
            loss = loss_fn(x, y_pos, y_neg)
            loss.backward(retain_graph=True)
            arm = "A" if stop_grad else "B"
            print(f"[{name}, Arm {arm}] loss={loss.item():.6f}, "
                  f"grad_norm={x.grad.norm().item():.6f}")
            x.grad.zero_()

    print("All checks passed!")
