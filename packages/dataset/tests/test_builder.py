from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from bbb_dataset.builder import BuildConfig, DatasetBuilder
from bbb_dataset.enums import ProcessedArtifact


def _raw_row(**kwargs) -> dict:
    base = {
        "source_id": "S1",
        "sequence": "YGGFLR",
        "length": 6,
        "bbb_label": 1,
        "source_db": "B3Pred_D1",
        "split": "train",
        "source_split": "train",
        "label_tier": "gold",
    }
    base.update(kwargs)
    return base


def test_build_gold_writes_parquet(tmp_path: Path) -> None:
    cfg = BuildConfig.from_base_dir(tmp_path)
    builder = DatasetBuilder(cfg)
    raw = pd.DataFrame([_raw_row(), _raw_row(sequence="AAAAAA", bbb_label=0)])

    with patch.object(builder._sources, "load_all", return_value=raw):
        dataset, stats = builder.build_gold()

    assert len(dataset) == 2
    assert "peptide_id" in dataset.columns
    assert "cluster_id" in dataset.columns
    assert stats["rows_after_filter"] == 2
    assert cfg.layout.artifact(ProcessedArtifact.PEPTIDES_BBB).exists()


def test_build_augmented_requires_gold(tmp_path: Path) -> None:
    cfg = BuildConfig.from_base_dir(tmp_path)
    builder = DatasetBuilder(cfg)
    with pytest.raises(FileNotFoundError):
        builder.build_augmented()
