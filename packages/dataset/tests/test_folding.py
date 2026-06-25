from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bbb_dataset.folding import (
    FoldConfig,
    StructureFolder,
    manifest_fields,
    parse_run_json,
    run_dir_for_sequence,
    run_status,
)


def test_parse_run_json_extracts_metrics(tmp_path: Path) -> None:
    run_json = tmp_path / "run.json"
    run_json.write_text(
        json.dumps(
            {
                "output": {
                    "best_sample": {
                        "metrics": {"ptm": 0.8, "complex_plddt": 0.75},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    metrics = parse_run_json(run_json)
    assert metrics["ptm"] == 0.8
    fields = manifest_fields(metrics)
    assert fields["plddt"] == 75.0


def test_run_status_reads_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text('{"status": "succeeded"}', encoding="utf-8")
    assert run_status(run_dir) == "succeeded"


def test_rebuild_manifest_from_experiments_empty_when_no_runs(tmp_path: Path) -> None:
    df = pd.DataFrame({"sequence": ["YGGFLR"], "peptide_id": ["p1"]})
    folder = StructureFolder(FoldConfig())
    manifest, stats = folder.rebuild_manifest_from_experiments(
        df,
        experiments_dir=tmp_path / "experiments",
        structures_dir=tmp_path / "structures",
    )
    assert manifest.empty
    assert stats["n_missing"] == 1


def test_structure_folder_manifest_only(tmp_path: Path) -> None:
    df = pd.DataFrame({"sequence": ["YGGFLR"], "peptide_id": ["p1"]})
    folder = StructureFolder(FoldConfig())
    manifest, stats = folder.build_manifest(
        df,
        structures_dir=tmp_path / "structures",
        experiments_dir=tmp_path / "experiments",
        manifest_only=True,
    )
    assert manifest.empty
    assert stats["n_unique_sequences"] == 1


def test_run_dir_for_sequence_is_stable() -> None:
    path = run_dir_for_sequence(Path("/exp"), "YGGFLR")
    assert path.name.startswith("bbb-fold-")
