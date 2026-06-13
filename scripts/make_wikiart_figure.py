"""Render the WikiArt per-style figure for the writeup (final_writeup/figures/fig_wikiart.pdf).

`uv run --with matplotlib python scripts/make_wikiart_figure.py`
"""
import sys
import os

sys.path.insert(0, ".")

import torch
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.unet import UNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "experiments/wikiart_v2/checkpoints/step_0090000.pt"
OUT = "../final_writeup/figures/fig_wikiart.pdf"
STYLES = ["Ukiyo_e", "Impressionism", "Baroque", "Cubism",
          "Abstract_Expressionism", "Color_Field_Painting", "Pop_Art", "Realism"]
PER = 8
SEED = 0

classes = torchvision.datasets.ImageFolder("./dataset").classes
name2idx = {c: i for i, c in enumerate(classes)}

ckpt = torch.load(CKPT, map_location=DEV, weights_only=False)
cfg = ckpt["config"]; mc = cfg["model"]
model = UNet(image_size=cfg["data"]["image_size"], base_channels=mc["channels"],
             channel_mults=tuple(mc["channel_mults"]), num_res_blocks=mc["num_res_blocks"],
             attention_resolutions=tuple(mc["attention_resolutions"]), noise_dim=mc["noise_dim"],
             num_classes=mc["num_classes"]).to(DEV)
model.load_state_dict(ckpt["ema_state_dict"]["shadow"])
model.eval()

g = torch.Generator(device=DEV).manual_seed(SEED)
rows = []
with torch.no_grad():
    for s in STYLES:
        y = torch.full((PER,), name2idx[s], device=DEV)
        z = torch.randn(PER, mc["noise_dim"], device=DEV, generator=g)
        x = (model(z, y) * 0.5 + 0.5).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
        rows.append(x)

nS = len(STYLES)
fig, axes = plt.subplots(nS, PER, figsize=(PER * 0.8, nS * 0.8))
for i in range(nS):
    for j in range(PER):
        ax = axes[i, j]
        ax.imshow(rows[i][j])
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(STYLES[i].replace("_", " "), rotation=0, ha="right", va="center", fontsize=8)
            for sp in ax.spines.values():
                sp.set_visible(False)
        else:
            ax.axis("off")
plt.subplots_adjust(wspace=0.04, hspace=0.04)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, bbox_inches="tight")
print("saved", OUT)
