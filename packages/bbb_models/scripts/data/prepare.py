#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bbb_classifier.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy prepared BBB parquet for training.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    df = pd.read_parquet(args.input)
    ensure_dir(Path(args.output).parent)
    df.to_parquet(args.output, index=False)
    print(f"Prepared dataset saved to {args.output} with {len(df)} rows.")


if __name__ == "__main__":
    main()
