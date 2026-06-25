from __future__ import annotations

import pandas as pd

from bbb_dataset.splits import FoldSplitter


def test_mark_external_holdout_flags_val_split() -> None:
    df = pd.DataFrame({"split": ["train", "val", "train"]})
    out = FoldSplitter().mark_external_holdout(df)
    assert out["external_test"].tolist() == [0, 1, 0]


def test_assign_cluster_folds_respects_groups() -> None:
    df = pd.DataFrame(
        {
            "bbb_label": [1, 0, 1, 0, 1, 0],
            "cluster_id": [0, 0, 1, 1, 2, 2],
            "external_test": [0, 0, 0, 0, 0, 0],
        }
    )
    out = FoldSplitter(n_splits=3, seed=0).assign_cluster_folds(df)
    assert set(out["fold_id"]) <= {-1, 0, 1, 2}
    assert (out["fold_id"] >= 0).all()
