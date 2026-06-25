#!/usr/bin/env python3
"""End-to-end BBB dataset pipeline: build, EDA, augmentation, optional folding."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from tfg_bbb.eda import run_augmentation_eda, run_gold_eda
from tfg_bbb.pipeline import (
    BuildConfig,
    build_augmented_gold_dataset,
    build_gold_dataset,
    build_peptide_struct_manifest,
)


def _print_eda_summary(result: dict[str, object], title: str) -> None:
    print(f"\n=== {title} ===")
    for key in ("overview", "fold_table", "cluster_leakage", "train_overview", "fold_compare"):
        value = result.get(key)
        if hasattr(value, "empty") and not value.empty:
            print(value.to_string(index=False))
    leakage = result.get("cluster_leakage")
    if hasattr(leakage, "empty") and leakage.empty:
        print("Cluster leakage across folds: none detected.")
    figures = result.get("figures") or []
    if figures:
        print("Figures:")
        for path in figures:
            print(f"  - {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the full BBB dataset pipeline.")
    parser.add_argument("--base-dir", default=".", help="Dataset root (contains data/processed)")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA figure generation")
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

    cfg = BuildConfig(
        base_dir=base_dir,
        min_length=6,
        max_length=30,
        identity_threshold=0.9,
        random_seed=42,
        use_b3pdb=True,
        use_brainpeps=True,
        b3pdb_path=base_dir / "data" / "raw" / "b3pdb.tsv",
        brainpeps_path=base_dir / "data" / "raw" / "brainpeps.tsv",
    )
    processed_dir = cfg.processed_dir
    eda_dir = processed_dir / "eda_figures"
    augment_config = (base_dir / args.augment_config).resolve()
    fold_config = (base_dir / args.fold_config).resolve()

    print(f"Dataset root: {base_dir}")

    print("\n[1/5] Building gold dataset...")
    gold_df, gold_stats = build_gold_dataset(cfg)
    print(f"Gold: {gold_df.shape}")
    print("Gold stats:", gold_stats)

    if not args.skip_eda:
        print("\n[2/5] EDA pre-augmentation...")
        gold_eda = run_gold_eda(gold_df, eda_dir / "gold")
        _print_eda_summary(gold_eda, "Gold dataset overview")
    else:
        print("\n[2/5] EDA pre-augmentation skipped.")

    combined_df = gold_df
    aug_df = gold_df.iloc[0:0].copy()
    if args.skip_augment:
        print("\n[3/5] Augmentation skipped.")
    else:
        print("\n[3/5] Data augmentation...")
        aug_df, combined_df, aug_stats = build_augmented_gold_dataset(
            cfg,
            gold_df=gold_df,
            augment_config_path=augment_config,
        )
        print(f"Augmented: {aug_df.shape} | Combined: {combined_df.shape}")
        print("Aug stats:", aug_stats)

    if not args.skip_eda and not args.skip_augment:
        print("\n[4/5] EDA post-augmentation...")
        aug_eda = run_augmentation_eda(gold_df, combined_df, aug_df, eda_dir / "augmentation")
        _print_eda_summary(aug_eda, "Pre / post augmentation comparison")
    else:
        print("\n[4/5] EDA post-augmentation skipped.")

    run_folding = not args.skip_fold and bool(os.environ.get("BOLTZ_API_KEY"))
    if not run_folding:
        if args.skip_fold:
            print("\n[5/5] Folding skipped (--skip-fold).")
        else:
            print("\n[5/5] Folding skipped (set BOLTZ_API_KEY in .env.local).")
    else:
        print("\n[5/5] Structure folding...")
        manifest_df, fold_stats = build_peptide_struct_manifest(
            cfg,
            input_df=combined_df,
            fold_config_path=fold_config,
        )
        print(f"Manifest: {manifest_df.shape}")
        print("Fold stats:", fold_stats)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
