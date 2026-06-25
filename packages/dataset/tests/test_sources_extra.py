from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from bbb_dataset.enums import SourceDb
from bbb_dataset.sources import SourceRegistry


def test_ensure_b3pred_downloads_when_missing(tmp_path: Path) -> None:
    registry = SourceRegistry(raw_dir=tmp_path)
    with patch("bbb_dataset.sources.requests.get") as mock_get:
        mock_get.return_value.content = b">id\nYGGFLR\n"
        mock_get.return_value.raise_for_status = lambda: None
        files = registry._ensure_b3pred_files()
    assert len(files) == 4
    assert all(path.exists() for path in files.values())


def test_load_brainpeps_without_label_column(tmp_path: Path) -> None:
    path = tmp_path / "brain.tsv"
    path.write_text("sequence\textra\nYGGFLR\tmeta\n", encoding="utf-8")
    from bbb_dataset.sources import load_brainpeps

    df = load_brainpeps(path)
    assert len(df) == 1
    assert int(df.iloc[0]["bbb_label"]) == 0


def test_load_brainpeps_missing_sequence_column(tmp_path: Path) -> None:
    path = tmp_path / "brain.tsv"
    path.write_text("foo\nbar\n", encoding="utf-8")
    from bbb_dataset.sources import load_brainpeps

    assert load_brainpeps(path).empty


def test_load_b3pred_via_download(tmp_path: Path) -> None:
    registry = SourceRegistry(raw_dir=tmp_path)
    with patch.object(registry, "_ensure_b3pred_files") as mock_ensure:
        fa = tmp_path / "pos.fa"
        fa.write_text(">p\nYGGFLR\n", encoding="utf-8")
        mock_ensure.return_value = {
            "b3pred_pos_train.fa": fa,
            "b3pred_pos_val.fa": fa,
            "b3pred_neg_train.fa": fa,
            "b3pred_neg_val.fa": fa,
        }
        df = registry.load(SourceDb.B3PRED_D1)
    assert len(df) == 4
