"""FID evaluation using clean-fid."""

import os
import tempfile
from pathlib import Path

import torch
import numpy as np
from PIL import Image


def generate_images(
    model: torch.nn.Module,
    num_samples: int,
    noise_dim: int,
    batch_size: int = 64,
    device: str = "cuda",
) -> list[np.ndarray]:
    model.eval()
    images = []

    with torch.no_grad():
        remaining = num_samples
        while remaining > 0:
            B = min(batch_size, remaining)
            z = torch.randn(B, noise_dim, device=device)
            x = model(z)

            x = (x * 0.5 + 0.5).clamp(0, 1)
            x = (x * 255).byte().cpu().numpy()
            x = x.transpose(0, 2, 3, 1)

            for i in range(B):
                images.append(x[i])
            remaining -= B

    return images


def save_images_to_dir(images: list[np.ndarray], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        Image.fromarray(img).save(os.path.join(output_dir, f"{i:06d}.png"))


def compute_fid(
    model: torch.nn.Module,
    noise_dim: int = 128,
    num_samples: int = 10000,
    batch_size: int = 64,
    device: str = "cuda",
    dataset_name: str = "cifar10",
    dataset_split: str = "train",
) -> float:
    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        print("Warning: clean-fid not installed. Returning dummy FID.")
        return -1.0

    print(f"Generating {num_samples} images for FID evaluation...")
    images = generate_images(model, num_samples, noise_dim, batch_size, device)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_images_to_dir(images, tmpdir)
        print(f"Computing FID against {dataset_name} ({dataset_split})...")
        try:
            score = cleanfid.compute_fid(
                tmpdir,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                mode="clean",
            )
        except Exception as e:
            print(f"Warning: FID computation failed ({e}). Returning -1.")
            score = -1.0

    return score


if __name__ == "__main__":
    from src.models.unet import UNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(
        image_size=32,
        base_channels=96,
        channel_mults=(1, 2, 2, 2),
        noise_dim=128,
    ).to(device)

    print("Generating test images...")
    images = generate_images(model, num_samples=8, noise_dim=128, batch_size=4, device=device)
    print(f"Generated {len(images)} images, shape: {images[0].shape}, dtype: {images[0].dtype}")
    print(f"Value range: [{images[0].min()}, {images[0].max()}]")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_images_to_dir(images, tmpdir)
        files = os.listdir(tmpdir)
        print(f"Saved {len(files)} images to temp dir")

    print("FID module smoke test passed!")
