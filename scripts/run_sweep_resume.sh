#!/usr/bin/env bash
# Resume the anchor sweep: 16k, 64k (20k steps each), then 256k feasibility (1k steps).
set -euo pipefail
cd "$(dirname "$0")/.."

for cfg in 16k 64k; do
    echo "===== [$(date)] anchor sweep ${cfg} (20k steps) ====="
    uv run python scripts/anchor_sweep.py --config configs/anchor_sweep_${cfg}.yaml \
        2>&1 | tee -a experiments/anchor_sweep_stdout.log
done

echo "===== [$(date)] anchor sweep 256k feasibility (1k steps) ====="
uv run python scripts/anchor_sweep.py --config configs/anchor_sweep_256k.yaml \
    2>&1 | tee -a experiments/anchor_sweep_stdout.log

echo "===== [$(date)] ANCHOR SWEEP DONE ====="
