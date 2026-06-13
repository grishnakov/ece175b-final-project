"""Calibration probe for the CD headline run (throughput + memory + sanity).

`uv run python scripts/_calib_probe.py [--steps 600] [--warmup 200]`
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from train import load_config
from src.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cd_headline.yaml")
    ap.add_argument("--steps", type=int, default=600, help="total probe steps")
    ap.add_argument("--warmup", type=int, default=200,
                    help="steps to discard before timing (lazy CUDA init / cudnn)")
    ap.add_argument("--save-every", type=int, default=5000,
                    help="checkpoint cadence in the real run (for the ETA table)")
    args = ap.parse_args()

    config = load_config(args.config)
    horizon = int(config["training"]["steps"])
    batch = int(config["training"]["batch_size"])

    trainer = Trainer(config)

    loader = trainer._get_data_loader()
    n_data = len(loader.dataset)
    dataset_name = config.get("data", {}).get("dataset", "cifar10")
    is_real = n_data >= 50000
    print(f"\n[probe] dataset={dataset_name}  size={n_data}  "
          f"({'REAL' if is_real else 'SYNTHETIC FALLBACK (!!)'})  conditional={trainer.conditional}")
    if not is_real:
        print("[probe] FATAL: synthetic-data fallback is active — real data did not load.")
        sys.exit(2)

    data_iter = iter(loader)
    trainer.model.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    losses = []
    t_warmup_end = None
    t_start = time.time()
    print(f"[probe] running {args.steps} steps at batch={batch} "
          f"(discarding first {args.warmup} for timing)...")

    for step in range(1, args.steps + 1):
        try:
            x_real, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x_real, labels = next(data_iter)
        x_real = x_real.to(trainer.device)
        labels = labels.to(trainer.device) if trainer.conditional else None
        loss_val = trainer.train_step(x_real, labels)
        losses.append(loss_val)

        if step == args.warmup:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_warmup_end = time.time()
        if step % 100 == 0:
            print(f"[probe]   step {step:4d}  loss={loss_val:+.4f}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.time()

    timed_steps = args.steps - args.warmup
    timed_secs = t_end - (t_warmup_end if t_warmup_end is not None else t_start)
    sps = timed_steps / timed_secs if timed_secs > 0 else float("nan")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        reserved_gb = torch.cuda.max_memory_reserved() / 1e9
    else:
        peak_gb = reserved_gb = float("nan")

    finite = all(map(lambda v: v == v and abs(v) != float("inf"), losses))
    first_avg = sum(losses[:50]) / max(len(losses[:50]), 1)
    last_avg = sum(losses[-50:]) / max(len(losses[-50:]), 1)

    rec_raw = sps * 3600 * 100
    rec_rounded = int((rec_raw // args.save_every) * args.save_every)
    hours_for = lambda n: n / sps / 3600 if sps > 0 else float("nan")

    print("\n" + "=" * 64)
    print("CALIBRATION SUMMARY")
    print("=" * 64)
    print(f"  model params        : {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"  batch size          : {batch}")
    print(f"  data                : {dataset_name} ({n_data})"
          f"{', class-conditional' if trainer.conditional else ''}")
    print(f"  steps/s (post-warmup): {sps:.3f}   over {timed_steps} steps / {timed_secs:.1f}s")
    print(f"  peak mem allocated  : {peak_gb:.2f} GB   (reserved {reserved_gb:.2f} GB) of 128 GB")
    print(f"  loss finite         : {finite}")
    print(f"  loss first50 / last50: {first_avg:+.4f} -> {last_avg:+.4f}")
    print("-" * 64)
    print(f"  config horizon now  : {horizon:,} steps -> {hours_for(horizon):.1f} h")
    print(f"  ~100h would fit     : {rec_raw:,.0f} steps")
    print(f"  RECOMMEND steps     : {rec_rounded:,}  (rounded to save_every={args.save_every})")
    print(f"                        -> {hours_for(rec_rounded):.1f} h at measured rate")
    print(f"  samples seen @ rec  : {rec_rounded * batch / 1e6:.1f}M "
          f"(vs arm_c_long 300k*128 = 38.4M)")
    if sps < 0.45:
        print("-" * 64)
        print("  WARNING: steps/s < 0.45 -> horizon would drop below ~250k.")
        print("  Plan fallback: batch 256 or model R8 (num_res_blocks=2). RE-PROBE before launch.")
    print("=" * 64)


if __name__ == "__main__":
    main()
