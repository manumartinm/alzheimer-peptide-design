from __future__ import annotations

import pandas as pd
import pytest

from bbb_dataset.cleaning import SequenceCleaner
from bbb_dataset.schema import DatasetSchema


def test_filter_drops_invalid_length_and_aa() -> None:
    df = pd.DataFrame(
        {
            "sequence": ["SHORT", "YGGFLR", "INVALID1"],
            "bbb_label": [1, 1, 1],
        }
    )
    cleaner = SequenceCleaner(min_length=6, max_length=30)
    out, stats = cleaner.filter(df)
    assert len(out) == 1
    assert out.iloc[0]["sequence"] == "YGGFLR"
    assert stats["rows_drop_length"] == 1
    assert stats["rows_drop_noncanonical"] == 1


def test_resolve_conflicts_drops_ambiguous_sequences() -> None:
    df = pd.DataFrame(
        {
            "sequence": ["YGGFLR", "YGGFLR", "AAAAAA"],
            "bbb_label": [1, 0, 1],
        }
    )
    out, stats = SequenceCleaner().resolve_conflicts(df)
    assert len(out) == 1
    assert out.iloc[0]["sequence"] == "AAAAAA"
    assert stats["conflict_sequences_removed"] == 1


def test_run_chains_filter_conflicts_dedup() -> None:
    df = pd.DataFrame(
        {
            "sequence": ["YGGFLR", "YGGFLA", "AAAAAA"],
            "bbb_label": [1, 1, 0],
        }
    )
    out, stats = SequenceCleaner(identity_threshold=0.5).run(df)
    assert len(out) <= 3
    assert "rows_after_identity_dedup" in stats


def test_schema_validation_errors() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        DatasetSchema.validate(pd.DataFrame({"sequence": ["AAAAAA"]}))
