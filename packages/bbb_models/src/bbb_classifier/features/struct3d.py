from __future__ import annotations

import numpy as np
import pandas as pd


def feature_matrix_3d(df: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
    cols = columns or []
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    available = [c for c in cols if c in df.columns]
    if not available:
        return np.zeros((len(df), 0), dtype=np.float32)
    return df[available].fillna(0.0).to_numpy(dtype=np.float32)
