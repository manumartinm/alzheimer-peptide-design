from __future__ import annotations

from pathlib import Path

from bbb_dataset.sources import SourceRegistry, load_brainpeps


def test_load_brainpeps_parses_labels(tmp_path: Path) -> None:
    path = tmp_path / "brain.tsv"
    path.write_text("sequence\tbbb_label\nYGGFLR\t1\nCCCCCC\t0\n", encoding="utf-8")
    df = load_brainpeps(path)
    assert len(df) == 2
    assert df.iloc[0]["label_tier"] == "silver"
    assert int(df.iloc[0]["bbb_label"]) == 1


def test_load_all_concatenates_sources(tmp_path: Path) -> None:
    b3pdb = tmp_path / "b3pdb.tsv"
    b3pdb.write_text("sequence\nYGGFLR\n", encoding="utf-8")
    registry = SourceRegistry(raw_dir=tmp_path, b3pdb_path=b3pdb)
    with __import__("unittest.mock", fromlist=["patch"]).patch.object(
        registry,
        "load",
        side_effect=lambda source: __import__("pandas").DataFrame(
            [{"sequence": "YGGFLR", "source_db": source.value}]
        ),
    ):
        from bbb_dataset.enums import SourceDb

        df = registry.load_all(frozenset({SourceDb.B3PDB, SourceDb.BRAINPEPS}))
    assert len(df) == 2
