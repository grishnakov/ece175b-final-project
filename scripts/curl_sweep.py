"""Extended curl diagnostic sweep across dimensions and sigmas."""

import json
import sys
import torch

sys.path.insert(0, ".")
from src.kernels.laplacian import LaplacianKernel
from src.kernels.gaussian import GaussianKernel
from src.eval.curl import curl_diagnostic

results = {}

for D in [4, 8, 16, 32]:
    for sigma in [0.5, 1.0, 2.0]:
        torch.manual_seed(42)
        y_pos = torch.randn(50, D)
        y_neg = torch.randn(50, D) + 0.3

        for kernel_cls, kname in [(LaplacianKernel, "laplacian"), (GaussianKernel, "gaussian")]:
            kernel = kernel_cls(sigma=sigma)
            for norm in ["vanilla", "sharp"]:
                result = curl_diagnostic(y_pos, y_neg, kernel, norm, num_points=20)
                key = f"{kname}_D{D}_s{sigma}_{norm}"
                results[key] = {
                    "curl_mean": result["curl_mean"],
                    "curl_std": result["curl_std"],
                    "curl_max": result["curl_max"],
                    "curl_min": result["curl_min"],
                }
                print(f"{key}: curl_mean={result['curl_mean']:.8f}, curl_max={result['curl_max']:.8f}")

print("\n=== Summary: Laplacian vanilla vs sharp ===")
for D in [4, 8, 16, 32]:
    for sigma in [0.5, 1.0, 2.0]:
        v_key = f"laplacian_D{D}_s{sigma}_vanilla"
        s_key = f"laplacian_D{D}_s{sigma}_sharp"
        v_curl = results[v_key]["curl_mean"]
        s_curl = results[s_key]["curl_mean"]
        ratio = v_curl / s_curl if s_curl > 1e-15 else float("inf")
        print(f"  D={D}, sigma={sigma}: vanilla={v_curl:.8f}, sharp={s_curl:.8f}, ratio={ratio:.1f}x")

out_path = "experiments/curl_sweep_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
