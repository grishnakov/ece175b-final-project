"""Conservative Drifting — training entry point.

`uv run python train.py --config configs/arm_c.yaml`
`uv run python train.py --config configs/arm_c.yaml --steps 100`
"""

import argparse
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    config_dir = Path(config_path).parent
    default_path = config_dir / "default.yaml"

    config = {}
    if default_path.exists():
        with open(default_path) as f:
            config = yaml.safe_load(f) or {}

    with open(config_path) as f:
        overrides = yaml.safe_load(f) or {}

    config.update(overrides)
    return config


def main():
    parser = argparse.ArgumentParser(description="Conservative Drifting Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.steps is not None:
        config["training"]["steps"] = args.steps
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.seed is not None:
        config["seed"] = args.seed

    print("=" * 60)
    print("Conservative Drifting — Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Loaded config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    from src.trainer import Trainer

    trainer = Trainer(config, resume_from=args.resume)
    trainer.run()


if __name__ == "__main__":
    main()
