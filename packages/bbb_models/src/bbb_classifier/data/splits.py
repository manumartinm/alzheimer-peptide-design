from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def train_val_split(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    random_state: int,
    fold_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fold_col and fold_col in df.columns:
        val_df = df[df[fold_col] == 0].copy()
        train_df = df[df[fold_col] != 0].copy()
        if len(train_df) > 0 and len(val_df) > 0:
            return train_df, val_df

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    y = df[label_col].to_numpy()
    train_idx, val_idx = next(splitter.split(df, y))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
