#!/usr/bin/env python3
"""Generate GSK3β guidance JSON configs from gsk3b_target.yaml."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

DEFAULT_TARGET_YAML = Path(__file__).resolve().parents[1] / "configs" / "gsk3b_target.yaml"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "targets" / "gsk3b"


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def build_guidance_json(cfg: dict, repo_root: Path, bbb_ckpt: Path | None) -> dict:
    regions = cfg["regions"]
    guidance = dict(cfg["guidance"])
    bbb = dict(guidance.pop("bbb", {}))
    if bbb_ckpt is not None:
        bbb["ckpt"] = _repo_relative(bbb_ckpt.resolve(), repo_root)
    guidance["bbb"] = bbb
    return {
        "hotspots_primary": regions["hotspots_primary"],
        "hotspots_secondary": regions.get("hotspots_secondary", []),
        "atp_cleft": regions["atp_cleft"],
        "guidance": guidance,
    }


def build_guidance_feats_json(cfg: dict, repo_root: Path, bbb_ckpt: Path) -> dict:
    g = cfg["guidance"]
    bbb = g["bbb"]
    ckpt = _repo_relative(bbb_ckpt.resolve(), repo_root)
    return {
        "guidance_hotspot_weight": g["hotspot_weight"],
        "guidance_atp_weight": g["atp_weight"],
        "guidance_alpha": g["alpha"],
        "guidance_cutoff_angstrom": g["cutoff_angstrom"],
        "guidance_lj_sigma": g["lj_sigma"],
        "guidance_max_force": g["max_force"],
        "guidance_bbb_weight": bbb["weight"],
        "guidance_membrane_weight": bbb["membrane_weight"],
        "guidance_bbb_ckpt": ckpt,
        "guidance_bbb_sigma_gate": bbb["sigma_gate"],
        "guidance_bbb_hidden": bbb["hidden"],
        "guidance_bbb_layers": bbb["layers"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--bbb-ckpt",
        type=Path,
        default=None,
        help="Path to exp09 geo checkpoint (default: packages/bbb_models/artifacts/.../best.ckpt)",
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()

    repo_root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    bbb_ckpt = args.bbb_ckpt or (
        repo_root
        / "packages/bbb_models/artifacts/models/exp09_struct_egnn_noise/checkpoints/best.ckpt"
    )

    cfg = yaml.safe_load(args.target_yaml.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    guidance_json = build_guidance_json(cfg, repo_root, bbb_ckpt if bbb_ckpt.exists() else None)
    feats_json = build_guidance_feats_json(cfg, repo_root, bbb_ckpt)

    (args.out_dir / "guidance.json").write_text(
        json.dumps(guidance_json, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "guidance_feats.json").write_text(
        json.dumps(feats_json, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out_dir / 'guidance.json'}")
    print(f"Wrote {args.out_dir / 'guidance_feats.json'}")


if __name__ == "__main__":
    main()
