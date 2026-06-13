#!/usr/bin/env bash
# Reproduce Conservative Drifting: Arm C (sharp norm, log-KDE, no stop-grad)
# `bash scripts/reproduce_cd.sh`
set -euo pipefail

echo "=== Reproducing Arm C: Conservative Drifting ==="
echo "This trains with log-KDE scalar loss and sharp normalization."
echo "No stop-gradient wrapper needed (gradient flows through scalar potential)."
echo ""

uv run python train.py \
    --config configs/arm_c.yaml \
    --seed 42

echo ""
echo "=== Conservative Drifting training complete ==="
echo "Results in experiments/arm_c/"
