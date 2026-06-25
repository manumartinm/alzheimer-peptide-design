from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def mark_external_holdout(df: pd.DataFrame, split_col: str = "split", holdout_value: str = "val") -> pd.DataFrame:
    out = df.copy()
    out["external_test"] = out[split_col].astype(str).str.lower().eq(holdout_value).astype(int)
    return out


def assign_cluster_folds(
    df: pd.DataFrame,
    label_col: str = "bbb_label",
    cluster_col: str = "cluster_id",
    n_splits: int = 5,
    seed: int = 42,
    only_non_holdout: bool = True,
    holdout_col: str = "external_test",
) -> pd.DataFrame:
    out = df.copy()
    out["fold_id"] = -1

    if only_non_holdout and holdout_col in out.columns:
        mask = out[holdout_col] == 0
    else:
        mask = np.ones(len(out), dtype=bool)

    subset = out.loc[mask]
    if subset.empty:
        return out

    y = subset[label_col].to_numpy()
    groups = subset[cluster_col].to_numpy()
    unique_groups = len(np.unique(groups))
    class_counts = pd.Series(y).value_counts()
    max_allowed = int(min(unique_groups, class_counts.min())) if not class_counts.empty else 0
    n_eff = max(2, min(n_splits, max_allowed)) if max_allowed >= 2 else 1

    if n_eff < 2:
        out.loc[subset.index, "fold_id"] = 0
        out["fold_id"] = out["fold_id"].astype(int)
        return out

    splitter = StratifiedGroupKFold(n_splits=n_eff, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(splitter.split(subset, y=y, groups=groups)):
        fold_rows = subset.iloc[val_idx].index
        out.loc[fold_rows, "fold_id"] = fold

    out["fold_id"] = out["fold_id"].astype(int)
    return out
