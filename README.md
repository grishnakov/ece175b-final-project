# Conservative Drifting Final Project

This project contains the code for a Conservative Drifting study: a single-step image generator trained with a sharp-normalized log-KDE objective. The main experiment is on CIFAR-10 at 32x32, with supporting scripts for curl diagnostics, anchor-set scaling, sampling checkpoints, FID evaluation, and an optional class-conditional WikiArt run at 64x64.

The key idea is to replace the non-conservative drifting field used by vanilla drifting with a sharp-normalized log-KDE scalar objective. In the report, this is used to train without the original stop-gradient wrapper, test conservativity with curl diagnostics, and measure how large KDE anchor sets scale on unified memory hardware.

## Setup

`uv` and Python 3.13 or newer are required.

```bash
cd ece175b-final-project
uv sync
```

## Training

Run the main Conservative Drifting CIFAR configuration:

```bash
uv run python train.py --config configs/arm_c.yaml
```

Run a shorter test version:

```bash
uv run python train.py --config configs/arm_c.yaml --steps 100
```

Run the longer final-report CIFAR recipe with EMA and warmup plus cosine decay:

```bash
uv run python train.py --config configs/arm_c_long.yaml
```

Resume from a checkpoint:

```bash
uv run python train.py --config configs/arm_c_long.yaml --resume experiments/arm_c_long/checkpoints/step_0100000.pt
```

Checkpoints, training logs, and sample grids are saved under `experiments/<run_name>/`.

## Reproducing Experiments

Run the four-arm ablation:

```bash
bash scripts/run_ablation.sh
```

Run the curl diagnostic sweep:

```bash
uv run python scripts/curl_sweep.py
```

... There are more scripts in the `scripts/` directory.

## Sampling and Evaluation

Generate a grid from a trained checkpoint:

```bash
uv run python scripts/sample.py --ckpt experiments/arm_c_long/checkpoints/step_0300000.pt --ema -n 64 -o samples.png
```

Compute clean-FID for the default trained CIFAR checkpoints:

```bash
uv run python scripts/compute_fid_all.py
```

Compute FID for one checkpoint:

```bash
uv run python scripts/compute_fid_all.py --ckpt experiments/arm_c_long/checkpoints/step_0300000.pt --ema --name arm_c_long_ema
```

## WikiArt Run

The WikiArt config: same stop-gradient-free recipe to 64x64 class-conditional generation with a VGG feature-space (didn't work without it) kernel and FiLM conditioning:

```bash
uv run python train.py --config configs/wikiart_v2.yaml
```

The WikiArt scripts expect an ImageFolder-style dataset at `dataset/`, with one subdirectory per style class. After training, evaluate or make a per-style grid with:

```bash
uv run python scripts/eval_wikiart.py --ckpt experiments/wikiart_v2/checkpoints/step_0090000.pt --ema --name wikiart_v2_ema
uv run python scripts/eval_wikiart.py --ckpt experiments/wikiart_v2/checkpoints/step_0090000.pt --grid-only
```