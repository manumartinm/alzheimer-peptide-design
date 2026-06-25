from __future__ import annotations

from typing import ClassVar

import pandas as pd


class DatasetSchema:
    REQUIRED_COLUMNS: ClassVar[list[str]] = [
        "peptide_id",
        "source_id",
        "sequence",
        "length",
        "bbb_label",
        "source_db",
        "split",
        "source_split",
        "label_tier",
        "is_gold",
        "cluster_id",
        "external_test",
        "fold_id",
    ]

    @classmethod
    def validate(cls, df: pd.DataFrame) -> None:
        missing = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if not pd.api.types.is_integer_dtype(df["bbb_label"]):
            raise ValueError("bbb_label must be integer-like")
        if not pd.api.types.is_integer_dtype(df["fold_id"]):
            raise ValueError("fold_id must be integer-like")


def validate_dataset_schema(df: pd.DataFrame) -> None:
    DatasetSchema.validate(df)
