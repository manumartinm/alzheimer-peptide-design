#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from boltzgen_design.guidance.feats_builder import guidance_config_to_feats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare guidance metadata for BoltzGen diffusion.")
    p.add_argument("--target-config", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument(
        "--output-feats-json",
        default=None,
        help="Optional path for diffusion feat keys (defaults to <output-json dir>/guidance_feats.json).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.target_config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    payload = {
        "hotspots_primary": cfg["regions"]["hotspots_primary"],
        "hotspots_secondary": cfg["regions"]["hotspots_secondary"],
        "atp_cleft": cfg["regions"]["atp_cleft"],
        "guidance": cfg["guidance"],
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved guidance config: {out}")

    feats_path = Path(args.output_feats_json) if args.output_feats_json else out.parent / "guidance_feats.json"
    feats = guidance_config_to_feats(cfg["guidance"], target_root=cfg_path.parent)
    feats_path.write_text(json.dumps(feats, indent=2), encoding="utf-8")
    print(f"Saved diffusion feat keys: {feats_path}")


if __name__ == "__main__":
    main()
