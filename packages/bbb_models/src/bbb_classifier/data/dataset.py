from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BBBSample:
    sequence: str
    y: int
    tab: np.ndarray | None
    esm: np.ndarray | None
    feat3d: np.ndarray | None
    graph: dict[str, Any] | None


class BBBDataFrameDataset:
    def __init__(self, df: pd.DataFrame, label_col: str, sequence_col: str) -> None:
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.sequence_col = sequence_col

    def __len__(self) -> int:
        return len(self.df)

    def labels(self) -> np.ndarray:
        return self.df[self.label_col].to_numpy(dtype=np.int64)

    def sequences(self) -> list[str]:
        return self.df[self.sequence_col].astype(str).tolist()
