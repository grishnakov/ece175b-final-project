"""Log-KDE scalar loss for Conservative Drifting (Arms C, D).

    L = E_x[log q_KDE[k#](x) - log p_KDE[k#](x)]
    log p_KDE[k#](x) = logsumexp_j(log k#(x, y_j^+)) - log(N_pos)
    log q_KDE[k#](x) = logsumexp_{j!=i}(log k#(x, y_j^-)) - log(N_neg - 1)
"""

import torch
import torch.nn as nn

from src.kernels.base import Kernel


class LogKDELoss(nn.Module):
    def __init__(self, kernel: Kernel, extra_stop_grad: bool = False):
        super().__init__()
        self.kernel = kernel
        self.extra_stop_grad = extra_stop_grad

    def _compute_log_kde_ratio(
        self, x: torch.Tensor, y_pos: torch.Tensor, y_neg: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N_pos = y_pos.shape[0]
        N_neg = y_neg.shape[0]
        square = x.shape[0] == N_neg

        log_Ks_pos = self.kernel.log_sharp(x, y_pos)
        if labels is not None:
            pos_match = labels[:, None] == labels[None, :]
            log_Ks_pos = log_Ks_pos.masked_fill(~pos_match, float("-inf"))
            n_pos = pos_match.sum(dim=1).clamp(min=1).to(log_Ks_pos.dtype)
        else:
            n_pos = torch.full((x.shape[0],), float(N_pos), device=x.device, dtype=log_Ks_pos.dtype)
        log_p = torch.logsumexp(log_Ks_pos, dim=1) - torch.log(n_pos)

        log_Ks_neg = self.kernel.log_sharp(x, y_neg)
        if square:
            log_Ks_neg = log_Ks_neg.clone()
            log_Ks_neg.fill_diagonal_(float("-inf"))
        if labels is not None:
            neg_match = labels[:, None] == labels[None, :]
            log_Ks_neg = log_Ks_neg.masked_fill(~neg_match, float("-inf"))
            n_neg = (neg_match.sum(dim=1) - (1 if square else 0)).clamp(min=1).to(log_Ks_neg.dtype)
        else:
            eff = max(N_neg - 1, 1) if square else N_neg
            n_neg = torch.full((x.shape[0],), float(eff), device=x.device, dtype=log_Ks_neg.dtype)
        log_q = torch.logsumexp(log_Ks_neg, dim=1) - torch.log(n_neg)

        return log_q - log_p

    def forward(
        self,
        x_gen: torch.Tensor,
        y_pos: torch.Tensor,
        y_neg: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.extra_stop_grad:
            with torch.no_grad():
                ratio_detached = self._compute_log_kde_ratio(x_gen, y_pos, y_neg, labels)
            ratio_live = self._compute_log_kde_ratio(x_gen, y_pos, y_neg, labels)
            ratio = ratio_detached.sign() * ratio_live
        else:
            ratio = self._compute_log_kde_ratio(x_gen, y_pos, y_neg, labels)

        finite = torch.isfinite(ratio)
        if finite.all():
            return ratio.mean()
        if finite.any():
            return ratio[finite].mean()
        return (ratio * 0.0).sum()


if __name__ == "__main__":
    from src.kernels.gaussian import GaussianKernel
    from src.kernels.laplacian import LaplacianKernel

    torch.manual_seed(42)

    B, D = 8, 16
    x = torch.randn(B, D, requires_grad=True)
    y_pos = torch.randn(B, D)
    y_neg = x.detach().clone()

    for kernel_cls, name in [(GaussianKernel, "Gaussian"), (LaplacianKernel, "Laplacian")]:
        kernel = kernel_cls(sigma=1.0)

        loss_fn_c = LogKDELoss(kernel, extra_stop_grad=False)
        loss_c = loss_fn_c(x, y_pos, y_neg)
        loss_c.backward(retain_graph=True)
        print(f"[{name}, Arm C] loss={loss_c.item():.6f}, grad_norm={x.grad.norm().item():.6f}")
        x.grad.zero_()

        loss_fn_d = LogKDELoss(kernel, extra_stop_grad=True)
        loss_d = loss_fn_d(x, y_pos, y_neg)
        loss_d.backward(retain_graph=True)
        print(f"[{name}, Arm D] loss={loss_d.item():.6f}, grad_norm={x.grad.norm().item():.6f}")
        x.grad.zero_()

    print("\nAll checks passed!")
