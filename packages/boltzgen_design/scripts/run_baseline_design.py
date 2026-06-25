#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline GSK3beta BoltzGen campaign.")
    p.add_argument("--config", required=True, help="Path to design_campaign.yaml")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    design_spec = (cfg_path.parent / cfg["inputs"]["design_spec"]).resolve()
    protocol = cfg["campaign"]["protocol"]
    num_designs = int(cfg["campaign"]["num_designs"])
    steps = cfg["campaign"]["steps"]

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    check_cmd = ["boltzgen", "check", str(design_spec), "--output", str(output_dir / "checked")]
    run_cmd = [
        "boltzgen",
        "run",
        str(design_spec),
        "--output",
        str(output_dir),
        "--protocol",
        protocol,
        "--num_designs",
        str(num_designs),
        "--steps",
        *steps,
    ]

    print(" ".join(check_cmd))
    print(" ".join(run_cmd))
    if not args.dry_run:
        subprocess.run(check_cmd, check=True)
        subprocess.run(run_cmd, check=True)


if __name__ == "__main__":
    main()

