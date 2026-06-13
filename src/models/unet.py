"""UNet generator for Conservative Drifting.

Single-pass generator: x = f_theta(z), z ~ N(0, I).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int) -> GroupNorm32:
    return GroupNorm32(min(32, channels), channels)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0,
                 emb_dim: int | None = None):
        super().__init__()
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = normalization(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * out_channels) if emb_dim else None

        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.norm2(h)
        if self.emb_proj is not None and emb is not None:
            scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip_proj(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0

        self.norm = normalization(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        h = F.scaled_dot_product_attention(q, k, v)
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)

        return x + self.proj_out(h)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (16,),
        noise_dim: int = 128,
        dropout: float = 0.0,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.class_emb = nn.Embedding(num_classes, noise_dim) if num_classes else None
        emb_dim = noise_dim if num_classes else None

        channels_list = [base_channels * m for m in channel_mults]
        num_levels = len(channel_mults)

        spatial_total = base_channels * image_size * image_size
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, spatial_total),
            nn.SiLU(),
        )

        self.input_conv = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        skip_channels = [base_channels]
        ch = base_channels
        current_res = image_size

        for level in range(num_levels):
            out_ch = channels_list[level]
            for _ in range(num_res_blocks):
                self.encoder.append(ResBlock(ch, out_ch, dropout, emb_dim=emb_dim))
                ch = out_ch
                if current_res in attention_resolutions:
                    self.encoder.append(SelfAttention(ch))
                skip_channels.append(ch)

            if level < num_levels - 1:
                self.encoder.append(Downsample(ch))
                skip_channels.append(ch)
                current_res //= 2

        self.bottleneck = nn.ModuleList([
            ResBlock(ch, ch, dropout, emb_dim=emb_dim),
            SelfAttention(ch),
            ResBlock(ch, ch, dropout, emb_dim=emb_dim),
        ])

        for level in reversed(range(num_levels)):
            out_ch = channels_list[level]
            for i in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                self.decoder.append(ResBlock(ch + skip_ch, out_ch, dropout, emb_dim=emb_dim))
                ch = out_ch
                if current_res in attention_resolutions:
                    self.decoder.append(SelfAttention(ch))

            if level > 0:
                self.decoder.append(Upsample(ch))
                current_res *= 2

        self.output_norm = normalization(ch)
        self.output_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

        self._decoder_resblock_mask = [isinstance(m, ResBlock) for m in self.decoder]

    def forward(self, z: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        B = z.shape[0]

        emb = None
        if self.class_emb is not None:
            if y is None:
                raise ValueError("class-conditional UNet called without class labels y")
            emb = self.class_emb(y)
            z = z + emb

        h = self.noise_proj(z)
        h = h.view(B, -1, self.image_size, self.image_size)
        h = self.input_conv(h)

        skips = [h]
        for module in self.encoder:
            h = module(h, emb) if isinstance(module, ResBlock) else module(h)
            if isinstance(module, (ResBlock, Downsample)):
                skips.append(h)

        for module in self.bottleneck:
            h = module(h, emb) if isinstance(module, ResBlock) else module(h)

        for module, is_res in zip(self.decoder, self._decoder_resblock_mask):
            if is_res:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = module(h, emb)
            else:
                h = module(h)

        h = F.silu(self.output_norm(h))
        h = self.output_conv(h)
        return torch.tanh(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(
        image_size=32,
        base_channels=96,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(16,),
        noise_dim=128,
    ).to(device)

    z = torch.randn(4, 128, device=device)
    with torch.no_grad():
        out = model(z)

    print(f"Input:  z.shape = {z.shape}")
    print(f"Output: out.shape = {out.shape}")
    print(f"Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Device: {device}")
