from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    data = base_dir / "data"
    paths = {
        "raw": data / "raw",
        "interim": data / "interim",
        "processed": data / "processed",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
