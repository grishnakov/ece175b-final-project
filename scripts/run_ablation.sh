#!/usr/bin/env bash
# Run the full 4-arm ablation study
# `bash scripts/run_ablation.sh`
set -euo pipefail

SEED=${1:-42}

echo "=== Full Ablation Study (seed=$SEED) ==="
echo ""

for arm in arm_a arm_b arm_c arm_d; do
    echo "--- Training $arm ---"
    uv run python train.py \
        --config "configs/${arm}.yaml" \
        --seed "$SEED"
    echo "--- $arm complete ---"
    echo ""
done

echo "=== Ablation study complete ==="
echo "Results in experiments/arm_{a,b,c,d}/"
