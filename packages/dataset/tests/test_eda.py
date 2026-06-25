from __future__ import annotations

import pandas as pd

from tfg_bbb.eda import cluster_leakage_table, dataset_overview_table, fold_overview_table


def test_dataset_overview_table_counts() -> None:
    df = pd.DataFrame(
        {
            "sequence": ["AAAAAA", "BBBBBB", "CCCCCC"],
            "length": [6, 6, 8],
            "bbb_label": [1, 0, 1],
            "cluster_id": [0, 1, 1],
        }
    )
    out = dataset_overview_table(df, name="test")
    assert out.iloc[0]["rows"] == 3
    assert out.iloc[0]["bbb_positive"] == 2
    assert out.iloc[0]["unique_sequences"] == 3


def test_fold_overview_table() -> None:
    df = pd.DataFrame(
        {
            "fold_id": [0, 0, 1, 1],
            "bbb_label": [1, 0, 1, 1],
            "cluster_id": [10, 11, 12, 13],
            "external_test": [0, 0, 0, 0],
        }
    )
    out = fold_overview_table(df)
    assert len(out) == 2
    assert out.loc[out["fold_id"] == 0, "rows"].iloc[0] == 2


def test_cluster_leakage_table_empty_when_clean() -> None:
    df = pd.DataFrame(
        {
            "fold_id": [0, 1, 2],
            "bbb_label": [1, 0, 1],
            "cluster_id": [1, 2, 3],
            "external_test": [0, 0, 0],
        }
    )
    leaked = cluster_leakage_table(df)
    assert leaked.empty
