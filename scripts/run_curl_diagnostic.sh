#!/usr/bin/env bash
# Run the curl diagnostic to verify conservatism of drift fields
# `bash scripts/run_curl_diagnostic.sh`
set -euo pipefail

echo "=== Curl Diagnostic ==="
echo "Testing whether drift fields are conservative under different normalizations."
echo "Expected: Laplacian vanilla has nonzero curl; Laplacian sharp has zero curl."
echo ""

uv run python -m src.eval.curl

echo ""
echo "=== Diagnostic complete ==="
