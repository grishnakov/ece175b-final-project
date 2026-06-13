"""Frozen VGG16 perceptual feature extractor for feature-space Conservative Drifting.

phi(x): images in [-1,1], [B,3,H,W]  ->  [B, C*pool*pool] feature vectors.
"""

import torch
import torch.nn as nn
import torchvision


class VGGFeatures(nn.Module):
    def __init__(self, layer: int = 16, pool: int = 4):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        self.slice = nn.Sequential(*list(vgg.features[:layer])).eval()
        for p in self.slice.parameters():
            p.requires_grad_(False)
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) * 0.5
        x = (x - self.mean) / self.std
        h = self.slice(x)
        h = self.pool(h)
        return h.flatten(1)

    @property
    def feature_dim_per_pool(self) -> int:
        return 256


def build_feature_extractor(name: str = "vgg16", layer: int = 16, pool: int = 4) -> nn.Module:
    if name == "vgg16":
        return VGGFeatures(layer=layer, pool=pool)
    raise ValueError(f"Unknown feature extractor: {name}")
