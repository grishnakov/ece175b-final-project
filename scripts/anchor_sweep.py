"""Anchor-set scaling sweep for Conservative Drifting.

kernel_evals/sec = steps/sec * batch_size * N

`uv run python scripts/anchor_sweep.py --n 16000 --steps 20000`
`uv run python scripts/anchor_sweep.py --n 256000 --steps 5 --dry-run`
`uv run python scripts/anchor_sweep.py --config configs/anchor_sweep_16k.yaml`
`uv run python scripts/anchor_sweep.py`
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.models.unet import UNet
from src.kernels.laplacian import LaplacianKernel
from src.kernels.gaussian import GaussianKernel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import load_config


def build_kernel(config: dict):
    kcfg = config.get("kernel", {})
    ktype = kcfg.get("type", "laplacian")
    sigma = kcfg.get("sigma", 0.1)
    if ktype == "gaussian":
        return GaussianKernel(sigma=sigma)
    return LaplacianKernel(sigma=sigma)


def build_model(config: dict, device: str) -> UNet:
    mcfg = config.get("model", {})
    return UNet(
        image_size=config.get("data", {}).get("image_size", 32),
        base_channels=mcfg.get("channels", 96),
        channel_mults=tuple(mcfg.get("channel_mults", [1, 2, 2, 2])),
        num_res_blocks=mcfg.get("num_res_blocks", 2),
        attention_resolutions=tuple(mcfg.get("attention_resolutions", [16])),
        noise_dim=mcfg.get("noise_dim", 128),
    ).to(device)


def load_anchors(n: int, device: str, image_size: int = 32) -> torch.Tensor:
    d = 3 * image_size * image_size
    try:
        from torchvision.datasets import CIFAR10
        from torchvision import transforms

        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        ds = CIFAR10(root="./data", train=True, download=False, transform=tfm)
        pool = torch.stack([ds[i][0] for i in range(len(ds))]).view(len(ds), -1)
        g = torch.Generator().manual_seed(0)
        idx = torch.randint(0, pool.shape[0], (n,), generator=g)
        anchors = pool[idx].contiguous()
        print(f"  Loaded {n} CIFAR-10 anchors (pool={pool.shape[0]}, "
              f"replacement={'yes' if n > pool.shape[0] else 'no'})")
    except Exception as e:
        print(f"  Warning: CIFAR-10 unavailable ({e}); using synthetic anchors.")
        anchors = torch.randn(n, d).clamp(-1, 1)
    return anchors.to(device)


def logsumexp_sharp_chunked(kernel, x: torch.Tensor, anchors: torch.Tensor,
                            chunk: int) -> torch.Tensor:
    """logsumexp_j log_sharp(x, anchor_j) over all anchors."""
    B = x.shape[0]
    N = anchors.shape[0]
    m = torch.full((B,), float("-inf"), device=x.device)
    s = torch.zeros((B,), device=x.device)
    for start in range(0, N, chunk):
        lk = kernel.log_sharp(x, anchors[start:start + chunk])
        cm = lk.max(dim=1).values
        new_m = torch.maximum(m, cm)
        new_m = torch.nan_to_num(new_m, neginf=0.0)
        s = s * torch.exp(m - new_m) + torch.exp(lk - new_m.unsqueeze(1)).sum(dim=1)
        m = new_m
    return m + torch.log(s)


def run_one(n: int, steps: int, config: dict, device: str, chunk: int,
            dry_run: bool) -> dict:
    print(f"\n{'='*60}\nAnchor sweep: N={n}, steps={steps}"
          f"{' (dry run)' if dry_run else ''}\n{'='*60}")

    tcfg = config.get("training", {})
    batch_size = int(tcfg.get("batch_size", 128))
    lr = float(tcfg.get("lr", 1e-4))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    noise_dim = config.get("model", {}).get("noise_dim", 128)

    torch.manual_seed(config.get("seed", 42))
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_model(config, device)
    model.train()
    kernel = build_kernel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    anchors = load_anchors(n, device, config.get("data", {}).get("image_size", 32))
    log_N = torch.log(torch.tensor(float(n), device=device))

    completed = True
    final_loss = float("nan")
    start_time = time.time()
    actual_chunk = chunk if chunk > 0 else n

    try:
        for step in range(1, steps + 1):
            z = torch.randn(batch_size, noise_dim, device=device)
            x_fake = model(z).view(batch_size, -1)
            y_neg = x_fake.detach()

            log_p = logsumexp_sharp_chunked(kernel, x_fake, anchors, actual_chunk) - log_N

            log_Ks_neg = kernel.log_sharp(x_fake, y_neg).clone()
            log_Ks_neg.fill_diagonal_(float("-inf"))
            log_q = torch.logsumexp(log_Ks_neg, dim=1) - torch.log(
                torch.tensor(float(max(batch_size - 1, 1)), device=device))

            loss = (log_q - log_p).mean()

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            final_loss = loss.item()

            if step % max(steps // 5, 1) == 0 or step == 1:
                print(f"  step {step}/{steps}  loss={final_loss:.4f}")
    except RuntimeError as e:
        completed = False
        print(f"  RuntimeError (likely OOM): {e}")

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start_time

    completed_steps = step if completed else step - 1
    steps_per_sec = completed_steps / elapsed if elapsed > 0 else 0.0
    kernel_evals_per_sec = steps_per_sec * batch_size * n
    if device == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_mem_gb = float("nan")

    result = {
        "n": n,
        "steps_requested": steps,
        "steps_completed": completed_steps,
        "steps_per_sec": round(steps_per_sec, 4),
        "kernel_evals_per_sec": round(kernel_evals_per_sec, 1),
        "peak_memory_gb": round(peak_mem_gb, 4),
        "final_loss": round(final_loss, 6) if final_loss == final_loss else None,
        "batch_size": batch_size,
        "chunk": actual_chunk,
        "completed": completed,
        "dry_run": dry_run,
    }
    print(f"  -> {steps_per_sec:.3f} steps/s, "
          f"{kernel_evals_per_sec:.2e} kernel-evals/s, "
          f"{peak_mem_gb:.2f} GB peak, loss={final_loss:.4f}, "
          f"completed={completed}")

    del model, anchors, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Anchor-set scaling sweep")
    parser.add_argument("--n", type=int, default=None, help="Single anchor count")
    parser.add_argument("--steps", type=int, default=None, help="Steps for this run")
    parser.add_argument("--config", type=str, default=None,
                        help="Config yaml (provides anchor_n, training.steps, model, kernel)")
    parser.add_argument("--chunk", type=int, default=32768,
                        help="Anchor chunk size for the positive logsumexp (0 = full matrix)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Feasibility run: build everything, run --steps, no convergence claim")
    parser.add_argument("--out", type=str, default="experiments/anchor_sweep/results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    if args.config is not None:
        config = load_config(args.config)
        n = int(config.get("anchor_n", args.n or 1000))
        steps = int(args.steps or config.get("training", {}).get("steps", 20000))
        jobs = [(n, steps, config)]
    elif args.n is not None:
        config = load_config("configs/default.yaml")
        steps = int(args.steps or 20000)
        jobs = [(args.n, steps, config)]
    else:
        config = load_config("configs/default.yaml")
        steps = int(args.steps or 20000)
        jobs = [(n, steps, config) for n in (1000, 4000, 16000, 64000)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
        except Exception:
            results = []

    for n, steps, config in jobs:
        res = run_one(n, steps, config, device, args.chunk, args.dry_run)
        results = [r for r in results if not (r["n"] == n and r.get("dry_run") == res["dry_run"])]
        results.append(res)
        results.sort(key=lambda r: (r["n"], r.get("dry_run", False)))
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  Saved results -> {out_path}")

    print(f"\n{'='*60}\nAnchor sweep complete. Results in {out_path}\n{'='*60}")


if __name__ == "__main__":
    main()
