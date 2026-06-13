"""Compute FID scores for trained arms (raw and EMA weights).

`uv run python scripts/compute_fid_all.py`
`uv run python scripts/compute_fid_all.py --ckpt PATH --ema --name foo`
"""

import os
import sys
import json
import argparse

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from src.models.unet import UNet
from src.eval.fid import generate_images, save_images_to_dir

device = "cuda" if torch.cuda.is_available() else "cpu"

ref_dir = "experiments/cifar10_ref"
if not os.path.exists(ref_dir) or len(os.listdir(ref_dir)) < 10000:
    print("Preparing CIFAR-10 reference images...")
    os.makedirs(ref_dir, exist_ok=True)
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    ds = CIFAR10(root="./data", train=True, download=False)
    for i in range(10000):
        img = ds[i][0]
        img.save(os.path.join(ref_dir, f"{i:06d}.png"))
    print(f"Saved {10000} reference images to {ref_dir}")
else:
    print(f"Using existing reference images in {ref_dir}")

parser = argparse.ArgumentParser(description="Compute FID for trained models")
parser.add_argument("--ckpt", type=str, default=None, help="Single checkpoint path")
parser.add_argument("--ema", action="store_true", help="Use EMA weights for --ckpt")
parser.add_argument("--name", type=str, default=None, help="Label for --ckpt result")
parser.add_argument("--num-samples", type=int, default=10000)
args = parser.parse_args()

if args.ckpt is not None:
    arms = [(args.name or "custom", args.ckpt, args.ema)]
else:
    arms = [
        ("arm_c", "experiments/arm_c/checkpoints/step_0100000.pt", False),
        ("arm_c_long_raw", "experiments/arm_c_long/checkpoints/step_0300000.pt", False),
        ("arm_c_long_ema", "experiments/arm_c_long/checkpoints/step_0300000.pt", True),
    ]

from cleanfid import fid as cleanfid

results = {}

for arm_name, ckpt_path, use_ema in arms:
    print(f"\n{'='*60}")
    print(f"Computing FID for {arm_name} ({'EMA' if use_ema else 'raw'})")
    print(f"{'='*60}")

    if not os.path.exists(ckpt_path):
        print(f"  Skipping: checkpoint not found ({ckpt_path})")
        continue

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model_cfg = config.get("model", {})

    model = UNet(
        image_size=32,
        base_channels=model_cfg.get("channels", 96),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 2, 2])),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", [16])),
        noise_dim=model_cfg.get("noise_dim", 128),
    ).to(device)

    if use_ema:
        ema_state = ckpt.get("ema_state_dict")
        if ema_state is None:
            print(f"  Skipping: --ema requested but no ema_state_dict in {ckpt_path}")
            del model
            continue
        model.load_state_dict(ema_state["shadow"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    print(f"Generating {args.num_samples} images...")
    noise_dim = model_cfg.get("noise_dim", 128)
    images = generate_images(model, num_samples=args.num_samples, noise_dim=noise_dim,
                             batch_size=64, device=device)

    gen_dir = f"experiments/{arm_name}/fid_samples"
    os.makedirs(gen_dir, exist_ok=True)
    save_images_to_dir(images, gen_dir)

    print(f"Computing FID ({gen_dir} vs {ref_dir})...")
    fid_score = cleanfid.compute_fid(gen_dir, ref_dir, mode="clean", device=torch.device(device))
    results[arm_name] = fid_score
    print(f">>> {arm_name} FID = {fid_score:.2f}")

    del model, images
    torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("FID Summary")
print(f"{'='*60}")
for arm, fid_score in results.items():
    print(f"  {arm}: {fid_score:.2f}")

out_file = "experiments/fid_results.json"
existing = {}
if os.path.exists(out_file):
    try:
        with open(out_file) as f:
            existing = json.load(f)
    except Exception:
        existing = {}
existing.update(results)
with open(out_file, "w") as f:
    json.dump(existing, f, indent=2)
print(f"\nSaved to {out_file}")
