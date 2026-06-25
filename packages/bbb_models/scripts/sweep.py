#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

GEO_TYPES = {"struct_egnn_geo"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment sweeps from configs/experiments.")
    parser.add_argument(
        "--mode",
        choices=("classifier", "geo"),
        default="classifier",
        help="Run tabular/ESM experiments (classifier) or structural EGNN (geo).",
    )
    parser.add_argument("--experiments-dir", default="configs/experiments")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--output-root", default="artifacts")
    args = parser.parse_args()

    train_script = Path(__file__).resolve().parent / "train.py"
    for exp in sorted(Path(args.experiments_dir).glob("*.yaml")):
        model_type = yaml.safe_load(exp.read_text(encoding="utf-8")).get("model_type", "")
        is_geo = model_type in GEO_TYPES
        if args.mode == "geo" and not is_geo:
            continue
        if args.mode == "classifier" and is_geo:
            continue
        subprocess.run(
            [
                sys.executable,
                str(train_script),
                "--exp",
                str(exp),
                "--data-config",
                args.data_config,
                "--train-config",
                args.train_config,
                "--output-root",
                args.output_root,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
