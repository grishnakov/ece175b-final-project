#!/usr/bin/env bash
# Final-project run: 300k arm_c_long, then the anchor-set scaling sweep.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "===== [$(date)] arm_c_long (300k) starting ====="
uv run python train.py --config configs/arm_c_long.yaml \
    2>&1 | tee experiments/arm_c_long_stdout.log

echo "===== [$(date)] anchor sweep 1k/4k/16k/64k (20k steps each) ====="
uv run python scripts/anchor_sweep.py \
    2>&1 | tee experiments/anchor_sweep_stdout.log

echo "===== [$(date)] anchor sweep 256k feasibility (1k steps) ====="
uv run python scripts/anchor_sweep.py --config configs/anchor_sweep_256k.yaml \
    2>&1 | tee -a experiments/anchor_sweep_stdout.log

echo "===== [$(date)] ALL DONE ====="
