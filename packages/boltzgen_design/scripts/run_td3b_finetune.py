#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from boltzgen_design.td3b.reward import TD3BRewardConfig
from boltzgen_design.td3b.wdce_trainer import build_weighted_samples, dump_weighted_replay


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TD3B weighted replay (MVP).")
    p.add_argument("--candidates-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    candidates = json.loads(Path(args.candidates_json).read_text(encoding="utf-8"))
    cfg = TD3BRewardConfig(tau=args.tau, alpha=args.alpha)
    weighted = build_weighted_samples(candidates, cfg=cfg)
    dump_weighted_replay(weighted, Path(args.output_json))
    print(f"Saved {len(weighted)} TD3B weighted candidates to {args.output_json}")


if __name__ == "__main__":
    main()

