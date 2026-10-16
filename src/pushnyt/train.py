"""Training entrypoint. Replace the body of `train` with your model."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(config: dict) -> None:
    set_seed(config["seed"])
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} on {device}")
    print(f"config: {config}")
    # TODO: load data, build model, run training loop, save checkpoint to output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == "__main__":
    main()
