#!/usr/bin/env bash
# Reproduce baseline: Arm A (vanilla drifting with stop-gradient)
# `bash scripts/reproduce_baseline.sh`
set -euo pipefail

echo "=== Reproducing Arm A: Vanilla Drifting (baseline) ==="
echo "This trains with stop-gradient MSE loss and vanilla normalization."
echo ""

uv run python train.py \
    --config configs/arm_a.yaml \
    --seed 42

echo ""
echo "=== Baseline training complete ==="
echo "Results in experiments/arm_a/"
