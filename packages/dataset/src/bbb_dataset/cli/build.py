#!/usr/bin/env python3
"""End-to-end BBB dataset pipeline: build, augmentation, optional folding."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from bbb_dataset.builder import BuildConfig, DatasetBuilder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the full BBB dataset pipeline.")
    parser.add_argument("--base-dir", default=".", help="Dataset root (contains data/processed)")
    parser.add_argument("--skip-augment", action="store_true", help="Skip sequence augmentation")
    parser.add_argument("--skip-fold", action="store_true", help="Skip Boltz structure folding")
    parser.add_argument(
        "--augment-config",
        default="configs/augmentation.yaml",
        help="Augmentation yaml config",
    )
    parser.add_argument(
        "--fold-config",
        default="configs/folding.yaml",
        help="Folding yaml config",
    )
    args = parser.parse_args(argv)

    base_dir = Path(args.base_dir).resolve()
    load_dotenv(base_dir / ".env.local")

    cfg = BuildConfig.from_base_dir(base_dir)
    builder = DatasetBuilder(cfg)
    augment_config = (base_dir / args.augment_config).resolve()
    fold_config = (base_dir / args.fold_config).resolve()

    print(f"Dataset root: {base_dir}")

    print("\n[1/3] Building gold dataset...")
    gold_df, gold_stats = builder.build_gold()
    print(f"Gold: {gold_df.shape}")
    print("Gold stats:", gold_stats)

    combined_df = gold_df
    if args.skip_augment:
        print("\n[2/3] Augmentation skipped.")
    else:
        print("\n[2/3] Data augmentation...")
        _, combined_df, aug_stats = builder.build_augmented(
            gold_df=gold_df,
            augment_config_path=augment_config,
        )
        print(f"Combined: {combined_df.shape}")
        print("Aug stats:", aug_stats)

    run_folding = not args.skip_fold and bool(os.environ.get("BOLTZ_API_KEY"))
    if not run_folding:
        if args.skip_fold:
            print("\n[3/3] Folding skipped (--skip-fold).")
        else:
            print("\n[3/3] Folding skipped (set BOLTZ_API_KEY in .env.local).")
    else:
        print("\n[3/3] Structure folding...")
        manifest_df, fold_stats = builder.build_manifest(
            input_df=combined_df,
            fold_config_path=fold_config,
        )
        print(f"Manifest: {manifest_df.shape}")
        print("Fold stats:", fold_stats)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
