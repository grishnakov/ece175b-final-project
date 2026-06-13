"""End-to-end smoke test for the full Conservative Drifting pipeline."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import time

print("=" * 60)
print("Conservative Drifting — End-to-End Smoke Test")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

errors = []

print("[1/9] Testing UNet model...")
try:
    from src.models.unet import UNet, count_parameters

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

    assert out.shape == (4, 3, 32, 32), f"Expected (4,3,32,32), got {out.shape}"
    params = count_parameters(model)
    print(f"  OK: {params:,} params, output shape {out.shape}")

    z = torch.randn(2, 128, device=device)
    out = model(z)
    loss = out.mean()
    loss.backward()
    print(f"  OK: backward pass works")
    del model, z, out, loss
    torch.cuda.empty_cache()
except Exception as e:
    errors.append(f"Model: {e}")
    print(f"  FAIL: {e}")

print("\n[2/9] Testing kernels...")
try:
    from src.kernels.gaussian import GaussianKernel
    from src.kernels.laplacian import LaplacianKernel

    for kernel_cls, name in [(GaussianKernel, "Gaussian"), (LaplacianKernel, "Laplacian")]:
        kernel = kernel_cls(sigma=1.0)
        x = torch.randn(8, 16)
        y = torch.randn(8, 16)

        K = kernel(x, y)
        log_K = kernel.log_kernel(x, y)
        log_Ks = kernel.log_sharp(x, y)

        assert K.shape == (8, 8), f"Kernel shape wrong: {K.shape}"
        assert torch.allclose(K, torch.exp(log_K), atol=1e-5), "log_kernel inconsistent"

        print(f"  OK: {name} kernel, sharp kernel, log space")
except Exception as e:
    errors.append(f"Kernels: {e}")
    print(f"  FAIL: {e}")

print("\n[3/9] Testing losses...")
try:
    from src.losses.mse_field import MSEFieldLoss
    from src.losses.log_kde import LogKDELoss

    x = torch.randn(8, 16, requires_grad=True)
    y_pos = torch.randn(8, 16)
    y_neg = x.detach().clone()

    for kernel_cls, kname in [(GaussianKernel, "Gauss"), (LaplacianKernel, "Lapl")]:
        kernel = kernel_cls(sigma=1.0)

        for sg, arm in [(True, "A"), (False, "B")]:
            loss_fn = MSEFieldLoss(kernel, stop_grad=sg)
            loss = loss_fn(x, y_pos, y_neg)
            loss.backward(retain_graph=True)
            x.grad.zero_()
            print(f"  OK: {kname} MSE Arm {arm}, loss={loss.item():.4f}")

        for sg, arm in [(False, "C"), (True, "D")]:
            loss_fn = LogKDELoss(kernel, extra_stop_grad=sg)
            loss = loss_fn(x, y_pos, y_neg)
            loss.backward(retain_graph=True)
            x.grad.zero_()
            print(f"  OK: {kname} LogKDE Arm {arm}, loss={loss.item():.4f}")

except Exception as e:
    errors.append(f"Losses: {e}")
    print(f"  FAIL: {e}")

print("\n[4/9] Testing training loop (3 steps per arm, batch_size=8)...")
try:
    import yaml
    from pathlib import Path
    from src.trainer import Trainer

    config_dir = Path("configs")
    for arm in ["arm_a", "arm_b", "arm_c", "arm_d"]:
        with open(config_dir / "default.yaml") as f:
            config = yaml.safe_load(f)
        with open(config_dir / f"{arm}.yaml") as f:
            config.update(yaml.safe_load(f))

        config["training"]["steps"] = 3
        config["training"]["batch_size"] = 8
        config["training"]["log_every"] = 1
        config["training"]["save_every"] = 999999
        config["training"]["eval_every"] = 999999
        config["data"]["num_workers"] = 0
        config["output_dir"] = "experiments/_smoke"

        torch.cuda.empty_cache()
        trainer = Trainer(config)
        start = time.time()
        trainer.run()
        elapsed = time.time() - start
        print(f"  OK: {arm} — 3 steps in {elapsed:.1f}s")
        del trainer
        torch.cuda.empty_cache()

except Exception as e:
    errors.append(f"Training: {e}")
    print(f"  FAIL: {e}")

print("\n[5/9] Testing FID image generation...")
try:
    from src.eval.fid import generate_images

    model = UNet(image_size=32, base_channels=96, channel_mults=(1, 2, 2, 2), noise_dim=128)
    model = model.to(device)

    images = generate_images(model, num_samples=8, noise_dim=128, batch_size=4, device=device)
    assert len(images) == 8, f"Expected 8 images, got {len(images)}"
    assert images[0].shape == (32, 32, 3), f"Wrong shape: {images[0].shape}"
    assert images[0].dtype.name == "uint8", f"Wrong dtype: {images[0].dtype}"

    print(f"  OK: Generated {len(images)} images, shape={images[0].shape}")
    del model
    torch.cuda.empty_cache()
except Exception as e:
    errors.append(f"FID: {e}")
    print(f"  FAIL: {e}")

print("\n[6/9] Testing curl diagnostic...")
try:
    from src.eval.curl import curl_diagnostic

    y_pos = torch.randn(15, 4)
    y_neg = torch.randn(15, 4) + 0.5

    kernel = LaplacianKernel(sigma=1.0)

    result_v = curl_diagnostic(y_pos, y_neg, kernel, "vanilla", num_points=3)
    result_s = curl_diagnostic(y_pos, y_neg, kernel, "sharp", num_points=3)

    assert result_v["curl_mean"] > result_s["curl_mean"], \
        f"Expected vanilla curl > sharp curl, got {result_v['curl_mean']} vs {result_s['curl_mean']}"

    print(f"  OK: vanilla curl={result_v['curl_mean']:.6f}, sharp curl={result_s['curl_mean']:.6f}")
    print(f"  Confirmed: vanilla > sharp (theory validated)")
except Exception as e:
    errors.append(f"Curl: {e}")
    print(f"  FAIL: {e}")

print("\n[7/9] Testing EMA update/apply/restore round-trip...")
try:
    from src.train.ema import EMA

    model = UNet(image_size=32, base_channels=96, channel_mults=(1, 2, 2, 2),
                 num_res_blocks=2, attention_resolutions=(16,), noise_dim=128).to(device)
    ema = EMA(model, decay=0.99)

    ref_param = next(model.parameters())
    raw_before = ref_param.detach().clone()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    ema.update(model)

    raw_after = ref_param.detach().clone()
    ema.apply_to(model)
    applied = ref_param.detach().clone()
    ema.restore(model)
    restored = ref_param.detach().clone()

    assert torch.allclose(restored, raw_after), "restore() did not recover raw weights"
    assert not torch.allclose(applied, raw_after), "EMA weights identical to raw (load path broken)"

    ema2 = EMA(model, decay=0.99)
    ema2.load_state_dict(ema.state_dict())
    s1 = next(iter(ema.shadow.parameters()))
    s2 = next(iter(ema2.shadow.parameters()))
    assert torch.allclose(s1, s2), "EMA state_dict round-trip mismatch"

    print("  OK: EMA update/apply/restore + state_dict round-trip")
    del model, ema, ema2
    torch.cuda.empty_cache()
except Exception as e:
    errors.append(f"EMA: {e}")
    print(f"  FAIL: {e}")

print("\n[8/9] Testing warmup+cosine LR scheduler...")
try:
    from src.train.lr_schedule import build_warmup_cosine_scheduler

    m = UNet(image_size=32, base_channels=96, channel_mults=(1, 2, 2, 2), noise_dim=128).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=2e-4)
    sched = build_warmup_cosine_scheduler(opt, total_steps=100, warmup_steps=10, lr_min=1e-6)

    lrs = []
    for _ in range(100):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])

    assert lrs[0] < lrs[9], "warmup did not ramp LR up"
    assert max(lrs) <= 2e-4 + 1e-9, "LR exceeded base LR"
    assert lrs[-1] < lrs[9], "cosine decay did not reduce LR"
    print(f"  OK: scheduler ramps {lrs[0]:.2e} -> peak {max(lrs):.2e} -> {lrs[-1]:.2e}")
    del m, opt, sched
    torch.cuda.empty_cache()
except Exception as e:
    errors.append(f"Scheduler: {e}")
    print(f"  FAIL: {e}")

print("\n[9/9] Testing one anchor-sweep step (N=256, B=8)...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from anchor_sweep import logsumexp_sharp_chunked

    kernel = LaplacianKernel(sigma=0.1)
    x = torch.randn(8, 3072, device=device, requires_grad=True)
    anchors = torch.randn(256, 3072, device=device)

    lse_chunked = logsumexp_sharp_chunked(kernel, x, anchors, chunk=64)
    lse_full = torch.logsumexp(kernel.log_sharp(x, anchors), dim=1)
    assert torch.allclose(lse_chunked, lse_full, atol=1e-4), "chunked logsumexp mismatch"
    lse_chunked.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "anchor-sweep grad invalid"

    print("  OK: chunked anchor logsumexp matches full + gradient flows")
    torch.cuda.empty_cache()
except Exception as e:
    errors.append(f"AnchorSweep: {e}")
    print(f"  FAIL: {e}")

print("\n" + "=" * 60)
if errors:
    print(f"SMOKE TEST FAILED: {len(errors)} error(s)")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("SMOKE TEST PASSED: All components working correctly")
    print("Ready for training on DGX Spark with full batch sizes.")
    sys.exit(0)
