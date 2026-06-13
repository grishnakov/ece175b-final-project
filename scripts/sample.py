"""Generate a sample grid from a trained checkpoint (raw or EMA weights).

`uv run python scripts/sample.py --ckpt experiments/arm_c_long/checkpoints/step_0300000.pt`
`uv run python scripts/sample.py --ckpt .../step_0300000.pt --ema`
`uv run python scripts/sample.py --ckpt .../step_0300000.pt --ema -n 64 -o out.png`
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torchvision.utils import save_image

from src.models.unet import UNet


def build_model_from_config(config: dict, device: str) -> UNet:
    model_cfg = config.get("model", {})
    return UNet(
        image_size=config.get("data", {}).get("image_size", 32),
        base_channels=model_cfg.get("channels", 96),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 2, 2])),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", [16])),
        noise_dim=model_cfg.get("noise_dim", 128),
    ).to(device)


def load_weights(model: UNet, ckpt: dict, use_ema: bool):
    if use_ema:
        ema_state = ckpt.get("ema_state_dict")
        if ema_state is None:
            raise ValueError(
                "--ema requested but checkpoint has no ema_state_dict "
                "(was EMA enabled during training?)"
            )
        model.load_state_dict(ema_state["shadow"])
        return "ema"
    model.load_state_dict(ckpt["model_state_dict"])
    return "raw"


def main():
    parser = argparse.ArgumentParser(description="Sample a grid from a checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--ema", action="store_true", help="Use EMA weights instead of raw")
    parser.add_argument("-n", "--num-samples", type=int, default=64, help="Number of samples")
    parser.add_argument("--nrow", type=int, default=8, help="Images per row in the grid")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = build_model_from_config(config, device)
    label = load_weights(model, ckpt, args.ema)
    model.eval()

    noise_dim = config.get("model", {}).get("noise_dim", 128)
    with torch.no_grad():
        z = torch.randn(args.num_samples, noise_dim, device=device)
        samples = model(z)

    if args.output is not None:
        out_path = Path(args.output)
    else:
        ckpt_path = Path(args.ckpt)
        stem = ckpt_path.stem
        out_path = ckpt_path.parent.parent / f"sample_{stem}_{label}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(samples * 0.5 + 0.5, out_path, nrow=args.nrow)
    print(f"[{label}] step={ckpt.get('step', '?')}  ->  {out_path}")


if __name__ == "__main__":
    main()
