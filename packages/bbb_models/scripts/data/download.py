#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from bbb_classifier.data.hf_peptides import (
    HF_DATASET_REPO,
    cache_dir_from_config,
    sync_hf_dataset,
)
from bbb_classifier.utils.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the BBB peptide dataset from Hugging Face."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Local cache directory (default: dataset_root from --data-config)",
    )
    parser.add_argument(
        "--repo",
        default=HF_DATASET_REPO,
        help="Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
        help="Data config used to resolve the default cache directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if peptides.parquet already exists",
    )
    args = parser.parse_args()

    output = args.output or cache_dir_from_config(read_yaml(args.data_config))
    cache = sync_hf_dataset(output, repo_id=args.repo, force=args.force)
    print(f"Dataset ready at {cache}")
    print(f"  parquet: {cache / 'peptides.parquet'}")
    print(f"  structures: {cache / 'structures'}")


if __name__ == "__main__":
    main()
