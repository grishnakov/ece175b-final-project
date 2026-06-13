"""Evaluate a class-conditional WikiArt checkpoint: per-style grid + clean-FID.

`uv run python scripts/eval_wikiart.py --ckpt experiments/wikiart_v2/checkpoints/step_0090000.pt --ema --name wikiart_v2_ema`
`uv run python scripts/eval_wikiart.py --ckpt .../step_0090000.pt --name wikiart_v2_raw`
`uv run python scripts/eval_wikiart.py --ckpt .../step_0090000.pt --grid-only`
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, ".")

import torch
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from src.models.unet import UNet
from src.data.wikiart import _safe_loader
from src.eval.fid import save_images_to_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(config: dict) -> UNet:
    mc = config["model"]
    return UNet(
        image_size=config.get("data", {}).get("image_size", 64),
        base_channels=mc.get("channels", 128),
        channel_mults=tuple(mc.get("channel_mults", [1, 2, 2, 2])),
        num_res_blocks=mc.get("num_res_blocks", 2),
        attention_resolutions=tuple(mc.get("attention_resolutions", [16, 8])),
        noise_dim=mc.get("noise_dim", 128),
        num_classes=mc.get("num_classes"),
    ).to(DEVICE)


def load_weights(model: UNet, ckpt: dict, use_ema: bool) -> str:
    if use_ema:
        ema = ckpt.get("ema_state_dict")
        if ema is None:
            raise ValueError("--ema requested but checkpoint has no ema_state_dict")
        model.load_state_dict(ema["shadow"])
        return "ema"
    model.load_state_dict(ckpt["model_state_dict"])
    return "raw"


def get_dataset(res: int, data_dir: str = "./dataset"):
    tfm = T.Compose([T.Resize(res), T.CenterCrop(res)])
    return torchvision.datasets.ImageFolder(root=data_dir, loader=_safe_loader, transform=tfm)


def prepare_reference(ds, res: int, n: int) -> str:
    ref_dir = f"experiments/wikiart_ref{res}"
    if os.path.isdir(ref_dir) and len(os.listdir(ref_dir)) >= n:
        print(f"Using existing reference {ref_dir} ({len(os.listdir(ref_dir))} images)")
        return ref_dir
    os.makedirs(ref_dir, exist_ok=True)
    idxs = list(range(len(ds.samples)))
    random.Random(0).shuffle(idxs)
    for i, idx in enumerate(idxs[:n]):
        ds[idx][0].save(os.path.join(ref_dir, f"{i:06d}.png"))
        if (i + 1) % 2000 == 0:
            print(f"  reference {i + 1}/{n}", flush=True)
    print(f"Saved {min(n, len(idxs))} reference images to {ref_dir}")
    return ref_dir


@torch.no_grad()
def generate_conditional(model, noise_dim, n, label_pool, batch=256, seed=0):
    model.eval()
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    pool = torch.tensor(label_pool, device=DEVICE)
    sel = pool[torch.randint(len(pool), (n,), generator=g, device=DEVICE)]
    out = []
    for s in range(0, n, batch):
        y = sel[s:s + batch]
        z = torch.randn(y.numel(), noise_dim, device=DEVICE, generator=g)
        x = (model(z, y) * 0.5 + 0.5).clamp(0, 1)
        x = (x * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)
        out.extend(x[i] for i in range(x.shape[0]))
    return out


@torch.no_grad()
def per_style_grid(model, noise_dim, num_classes, out_path, per=8, seed=0):
    model.eval()
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    y = torch.arange(num_classes, device=DEVICE).repeat_interleave(per)
    z = torch.randn(y.numel(), noise_dim, device=DEVICE, generator=g)
    x = (model(z, y) * 0.5 + 0.5).clamp(0, 1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(x, out_path, nrow=per)
    print(f"Saved per-style grid ({num_classes} styles x {per}) -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--ref-samples", type=int, default=10000)
    ap.add_argument("--grid-only", action="store_true", help="per-style grid only, skip FID")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    res = config.get("data", {}).get("image_size", 64)
    nc = config["model"].get("num_classes")
    noise_dim = config["model"].get("noise_dim", 128)
    if nc is None:
        sys.exit("This checkpoint is unconditional; use scripts/sample.py / compute_fid_all.py.")
    run_dir = f"experiments/{config['name']}"
    model = build_model(config)
    label = load_weights(model, ckpt, args.ema)
    name = args.name or f"{config['name']}_{label}"
    print(f"[{name}] {config['name']} step {ckpt.get('step', '?')} ({label}), {nc} classes, {res}x{res}")

    per_style_grid(model, noise_dim, nc, f"{run_dir}/eval_grid_{name}.png", per=8)

    if args.grid_only:
        return

    ds = get_dataset(res)
    ref_dir = prepare_reference(ds, res, args.ref_samples)
    print(f"Generating {args.num_samples} class-conditional images...")
    gen = generate_conditional(model, noise_dim, args.num_samples, ds.targets)
    gen_dir = f"{run_dir}/fid_gen_{name}"
    save_images_to_dir(gen, gen_dir)

    from cleanfid import fid as cleanfid
    print(f"Computing clean-FID ({gen_dir} vs {ref_dir})...")
    score = cleanfid.compute_fid(gen_dir, ref_dir, mode="clean", device=torch.device(DEVICE))
    print(f">>> {name} FID = {score:.2f}")

    out_file = "experiments/fid_results.json"
    existing = {}
    if os.path.exists(out_file):
        try:
            with open(out_file) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing[name] = score
    with open(out_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
