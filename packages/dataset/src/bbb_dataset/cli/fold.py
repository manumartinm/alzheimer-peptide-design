#!/usr/bin/env python3
"""Fold peptides via the Boltz API SDK and build a structural manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from bbb_dataset.builder import BuildConfig, DatasetBuilder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fold peptides with the Boltz API SDK.")
    parser.add_argument("--base-dir", default=".", help="Dataset root (contains data/processed)")
    parser.add_argument(
        "--input",
        default=None,
        help="Override input parquet path (default: peptides_bbb_with_augmentation.parquet)",
    )
    parser.add_argument("--config", default="configs/folding.yaml")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Rebuild manifest from succeeded runs in boltz-experiments (no API calls)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-fold all sequences via API even if a succeeded run already exists",
    )
    args = parser.parse_args(argv)

    base_dir = Path(args.base_dir).resolve()
    load_dotenv(base_dir / ".env.local")

    if not args.manifest_only and not os.environ.get("BOLTZ_API_KEY"):
        raise RuntimeError("BOLTZ_API_KEY missing; set it in dataset/.env.local")

    cfg = BuildConfig.from_base_dir(base_dir)
    input_parquet = Path(args.input).resolve() if args.input else None
    _, stats = DatasetBuilder(cfg).build_manifest(
        fold_config_path=(base_dir / args.config).resolve(),
        input_parquet=input_parquet,
        manifest_only=args.manifest_only,
        resume=not args.no_resume,
    )
    if args.manifest_only:
        print(
            f"Imported {stats['n_folded']} / {stats['n_unique_sequences']} sequences "
            f"({stats['n_missing']} still missing succeeded runs)"
        )
    else:
        print(
            f"Manifest rows: {stats['n_folded']} / {stats['n_unique_sequences']} "
            f"(imported={stats.get('n_imported', 0)}, api={stats.get('n_api_folded', 0)}, "
            f"errors={stats.get('n_errors', 0)})"
        )
    if stats.get("errors"):
        print("First errors:")
        for err in stats["errors"][:5]:
            print(f"  - {err}")
    print(f"Saved structural manifest to {stats['manifest_path']}")


if __name__ == "__main__":
    main()
