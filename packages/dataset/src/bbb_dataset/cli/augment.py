#!/usr/bin/env python3
"""Generate augmented training rows from the gold BBB peptide dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from bbb_dataset.builder import BuildConfig, DatasetBuilder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build augmented BBB peptide dataset artifacts.")
    parser.add_argument("--base-dir", default=".", help="Dataset root (contains data/processed)")
    parser.add_argument(
        "--config",
        default="configs/augmentation.yaml",
        help="Augmentation yaml config",
    )
    args = parser.parse_args(argv)

    base_dir = Path(args.base_dir).resolve()
    cfg = BuildConfig.from_base_dir(base_dir)
    _, _, stats = DatasetBuilder(cfg).build_augmented(
        augment_config_path=(base_dir / args.config).resolve(),
    )

    print(f"Gold rows: {stats['gold_rows']}")
    print(f"Augmented rows: {stats['n_generated']}")
    print(f"Combined rows: {stats['combined_rows']}")
    print(f"Wrote {cfg.processed_dir / 'peptides_bbb_augmented_extra.parquet'}")
    print(f"Wrote {cfg.processed_dir / 'peptides_bbb_with_augmentation.parquet'}")
    print(f"Stats: {cfg.processed_dir / 'augmentation_stats.json'}")


if __name__ == "__main__":
    main()
