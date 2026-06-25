from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def infer_tabular_columns(df: pd.DataFrame, excluded: Iterable[str]) -> list[str]:
    excluded_set = set(excluded)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in numeric_cols if c not in excluded_set]


def tabular_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.zeros((len(df), 0), dtype=np.float32)
    return df[columns].fillna(0.0).to_numpy(dtype=np.float32)
