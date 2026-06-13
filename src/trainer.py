"""Unified config-driven trainer for Conservative Drifting."""

import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.unet import UNet, count_parameters
from src.kernels.gaussian import GaussianKernel
from src.kernels.laplacian import LaplacianKernel
from src.losses.mse_field import MSEFieldLoss
from src.losses.log_kde import LogKDELoss
from src.train.ema import EMA
from src.train.lr_schedule import build_warmup_cosine_scheduler


class Trainer:
    def __init__(self, config: dict, resume_from: str | None = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_step = 0

        self._set_seed(config.get("seed", 42))
        self._build_model()
        self._build_kernel()
        self._build_feature_space()
        self._build_loss()
        self._build_optimizer()
        self._build_scheduler()
        self._build_ema()
        self._setup_output_dir()

        if resume_from is not None:
            self._resume_checkpoint(resume_from)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.seed = seed

    def _build_model(self):
        model_cfg = self.config.get("model", {})
        self.num_classes = model_cfg.get("num_classes", None)
        self.conditional = self.num_classes is not None
        self.model = UNet(
            image_size=self.config.get("data", {}).get("image_size", 32),
            base_channels=model_cfg.get("channels", 96),
            channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 2, 2])),
            num_res_blocks=model_cfg.get("num_res_blocks", 2),
            attention_resolutions=tuple(model_cfg.get("attention_resolutions", [16])),
            noise_dim=model_cfg.get("noise_dim", 128),
            num_classes=self.num_classes,
        ).to(self.device)
        cond = f", class-conditional ({self.num_classes} classes)" if self.conditional else ""
        print(f"Model: {count_parameters(self.model):,} parameters on {self.device}{cond}")

    def _build_kernel(self):
        kernel_cfg = self.config.get("kernel", {})
        kernel_type = kernel_cfg.get("type", "laplacian")
        sigma = kernel_cfg.get("sigma", 0.1)

        if kernel_type == "gaussian":
            self.kernel = GaussianKernel(sigma=sigma)
        elif kernel_type == "laplacian":
            self.kernel = LaplacianKernel(sigma=sigma)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        print(f"Kernel: {kernel_type} (sigma={sigma})")

    def _build_feature_space(self):
        kernel_cfg = self.config.get("kernel", {})
        self.feature_space = kernel_cfg.get("space", "pixel") == "feature"
        if self.feature_space:
            from src.models.perceptual import build_feature_extractor
            self.feature_extractor = build_feature_extractor(
                name=kernel_cfg.get("feature_net", "vgg16"),
                layer=kernel_cfg.get("feature_layer", 16),
                pool=kernel_cfg.get("feature_pool", 4),
            ).to(self.device).eval()
            print(f"Kernel space: FEATURE ({kernel_cfg.get('feature_net', 'vgg16')}, "
                  f"layer {kernel_cfg.get('feature_layer', 16)}, pool {kernel_cfg.get('feature_pool', 4)})")
        else:
            self.feature_extractor = None
            print("Kernel space: pixel")

    def _build_loss(self):
        loss_cfg = self.config.get("loss", {})
        loss_type = loss_cfg.get("type", "log_kde")
        stop_grad = loss_cfg.get("stop_grad", False)

        if loss_type == "mse_field":
            self.loss_fn = MSEFieldLoss(self.kernel, stop_grad=stop_grad)
            label = f"MSE field-matching (stop_grad={stop_grad})"
        elif loss_type == "log_kde":
            self.loss_fn = LogKDELoss(self.kernel, extra_stop_grad=stop_grad)
            label = f"Log-KDE scalar (extra_stop_grad={stop_grad})"
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        print(f"Loss: {label}")

    def _build_optimizer(self):
        train_cfg = self.config.get("training", {})
        lr = float(train_cfg.get("lr", 1e-4))
        betas = tuple(float(b) for b in train_cfg.get("adam_betas", [0.9, 0.999]))
        weight_decay = train_cfg.get("weight_decay", 0.0)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        print(f"Optimizer: Adam (lr={lr}, grad_clip={self.grad_clip})")

    def _build_scheduler(self):
        train_cfg = self.config.get("training", {})
        schedule = train_cfg.get("lr_schedule", "constant")

        if schedule == "cosine":
            self.scheduler = build_warmup_cosine_scheduler(
                self.optimizer,
                total_steps=int(train_cfg.get("steps", 100000)),
                warmup_steps=int(train_cfg.get("warmup_steps", 1000)),
                lr_min=float(train_cfg.get("lr_min", 1e-6)),
            )
            print(
                f"LR schedule: warmup+cosine "
                f"(warmup={train_cfg.get('warmup_steps', 1000)}, "
                f"lr_min={train_cfg.get('lr_min', 1e-6)})"
            )
        elif schedule == "constant":
            self.scheduler = None
            print("LR schedule: constant")
        else:
            raise ValueError(f"Unknown lr_schedule: {schedule}")

    def _build_ema(self):
        train_cfg = self.config.get("training", {})
        decay = train_cfg.get("ema_decay", None)

        if decay:
            self.ema = EMA(self.model, decay=float(decay))
            print(f"EMA: enabled (decay={decay})")
        else:
            self.ema = None
            print("EMA: disabled")

    def _setup_output_dir(self):
        base_dir = self.config.get("output_dir", "experiments/")
        name = self.config.get("name", "unnamed")
        self.output_dir = Path(base_dir) / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_path = self.output_dir / "train_log.jsonl"
        print(f"Output: {self.output_dir}")

    def _resume_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_step = ckpt["step"]
        if self.scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.ema is not None and ckpt.get("ema_state_dict") is not None:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        print(f"Resumed from {path} at step {self.start_step}")

    def _get_data_loader(self) -> DataLoader:
        data_cfg = self.config.get("data", {})
        train_cfg = self.config.get("training", {})
        dataset_name = data_cfg.get("dataset", "cifar10")

        if dataset_name == "cifar10":
            try:
                from src.data.cifar10 import get_cifar10_loader
                return get_cifar10_loader(
                    batch_size=train_cfg.get("batch_size", 128),
                    num_workers=data_cfg.get("num_workers", 4),
                    data_dir="./data",
                )
            except Exception as e:
                print(f"Warning: Could not load CIFAR-10 ({e}). Using synthetic data.")
                return self._get_synthetic_loader()
        elif dataset_name == "wikiart":
            image_size = data_cfg.get("image_size", 64)
            num_workers = data_cfg.get("num_workers", 8)
            if self.conditional:
                from src.data.wikiart import get_wikiart_conditional_loader
                loader, classes = get_wikiart_conditional_loader(
                    classes_per_batch=train_cfg.get("classes_per_batch", 8),
                    samples_per_class=train_cfg.get("samples_per_class", 16),
                    image_size=image_size,
                    num_workers=num_workers,
                    data_dir="./dataset",
                    seed=self.seed,
                )
            else:
                from src.data.wikiart import get_wikiart_loader
                loader, classes = get_wikiart_loader(
                    batch_size=train_cfg.get("batch_size", 128),
                    image_size=image_size,
                    num_workers=num_workers,
                    data_dir="./dataset",
                )
            self.class_names = classes
            print(f"WikiArt: {len(loader.dataset)} images, {len(classes)} classes, "
                  f"{image_size}x{image_size}"
                  + (f", balanced {train_cfg.get('classes_per_batch', 8)}x"
                     f"{train_cfg.get('samples_per_class', 16)}" if self.conditional else ""))
            return loader
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _get_synthetic_loader(self) -> DataLoader:
        train_cfg = self.config.get("training", {})
        batch_size = train_cfg.get("batch_size", 128)
        n_samples = max(batch_size * 10, 1000)

        images = torch.randn(n_samples, 3, 32, 32).clamp(-1, 1)
        labels = torch.randint(0, 10, (n_samples,))
        dataset = TensorDataset(images, labels)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

    def _log_step(self, step: int, loss_val: float, elapsed: float):
        record = {
            "step": step,
            "loss": loss_val,
            "elapsed_s": elapsed,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _save_checkpoint(self, step: int):
        path = self.checkpoint_dir / f"step_{step:07d}.pt"
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "ema_state_dict": self.ema.state_dict() if self.ema is not None else None,
            "config": self.config,
        }, path)
        print(f"  Saved checkpoint: {path}")

    def _save_samples(self, step: int, num_samples: int = 64):
        self._save_sample_grid(step, num_samples, suffix="")
        if self.ema is not None:
            self.ema.apply_to(self.model)
            self._save_sample_grid(step, num_samples, suffix="_ema")
            self.ema.restore(self.model)

    def _save_sample_grid(self, step: int, num_samples: int, suffix: str = ""):
        self.model.eval()
        noise_dim = self.config.get("model", {}).get("noise_dim", 128)
        with torch.no_grad():
            if self.conditional:
                ncls = min(8, self.num_classes)
                per = max(num_samples // ncls, 1)
                y = torch.arange(ncls, device=self.device).repeat_interleave(per)
                z = torch.randn(y.numel(), noise_dim, device=self.device)
                samples = self.model(z, y)
                nrow = per
            else:
                z = torch.randn(num_samples, noise_dim, device=self.device)
                samples = self.model(z)
                nrow = 8

        try:
            from torchvision.utils import save_image
            save_path = self.output_dir / f"samples_step_{step:07d}{suffix}.png"
            save_image(samples * 0.5 + 0.5, save_path, nrow=nrow)
            print(f"  Saved samples: {save_path}")
        except Exception as e:
            print(f"  Warning: Could not save samples: {e}")

        self.model.train()

    def train_step(
        self,
        x_real: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> float:
        B = x_real.shape[0]
        noise_dim = self.config.get("model", {}).get("noise_dim", 128)

        z = torch.randn(B, noise_dim, device=self.device)
        x_fake = self.model(z, labels) if self.conditional else self.model(z)

        if self.feature_space:
            x_fake_repr = self.feature_extractor(x_fake)
            with torch.no_grad():
                x_real_repr = self.feature_extractor(x_real)
        else:
            x_fake_repr = x_fake.view(B, -1)
            x_real_repr = x_real.view(B, -1)
        y_neg = x_fake_repr.detach()

        if self.conditional:
            loss = self.loss_fn(x_fake_repr, x_real_repr, y_neg, labels=labels)
        else:
            loss = self.loss_fn(x_fake_repr, x_real_repr, y_neg)

        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model)

        return loss.item()

    def run(self):
        train_cfg = self.config.get("training", {})
        total_steps = train_cfg.get("steps", 100000)
        log_every = train_cfg.get("log_every", 100)
        save_every = train_cfg.get("save_every", 5000)
        eval_every = train_cfg.get("eval_every", 10000)

        remaining = total_steps - self.start_step
        print(f"\nStarting training for {total_steps} steps (resuming from {self.start_step}, {remaining} remaining)...")
        loader = self._get_data_loader()
        data_iter = iter(loader)

        self.model.train()
        start_time = time.time()
        running_loss = 0.0

        pbar = tqdm(range(self.start_step + 1, total_steps + 1), desc="Training")
        for step in pbar:
            try:
                x_real, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x_real, labels = next(data_iter)

            x_real = x_real.to(self.device)
            labels = labels.to(self.device) if self.conditional else None
            loss_val = self.train_step(x_real, labels)
            running_loss += loss_val

            if step % log_every == 0:
                avg_loss = running_loss / log_every
                elapsed = time.time() - start_time
                pbar.set_postfix(loss=f"{avg_loss:.4f}", elapsed=f"{elapsed:.0f}s")
                self._log_step(step, avg_loss, elapsed)
                running_loss = 0.0

            if step % save_every == 0:
                self._save_checkpoint(step)

            if step % eval_every == 0:
                self._save_samples(step)

        self._save_checkpoint(total_steps)
        self._save_samples(total_steps)

        elapsed = time.time() - start_time
        print(f"\nTraining complete: {total_steps} steps in {elapsed:.1f}s")
        print(f"Logs: {self.log_path}")
        print(f"Checkpoints: {self.checkpoint_dir}")
