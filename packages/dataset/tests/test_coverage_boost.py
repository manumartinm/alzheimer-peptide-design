from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from bbb_dataset.cleaning import _run_cdhit_or_mmseqs
from bbb_dataset.folding import (
    FoldConfig,
    StructureFolder,
    fold_sequence,
    import_from_run_dir,
    resolve_or_fold_sequence,
)
from bbb_dataset.struct_io import parse_cif_backbone


def test_run_cdhit_or_mmseqs_returns_none_without_tools() -> None:
    with patch("bbb_dataset.cleaning.shutil.which", return_value=None):
        assignments = _run_cdhit_or_mmseqs(["ACDEFG", "ACDEFA"], threshold=0.8)
    assert assignments is None


def test_run_mmseqs_when_cdhit_unavailable(tmp_path: Path) -> None:
    tsv_content = "rep\tseq_0\nrep\tseq_1\n"

    def which(cmd: str) -> str | None:
        if cmd == "cd-hit":
            return None
        if cmd == "mmseqs":
            return "/usr/bin/mmseqs"
        return None

    def fake_run(cmd, **kwargs):
        if "createtsv" in cmd:
            (tmp_path / "clu.tsv").write_text(tsv_content, encoding="utf-8")
        return MagicMock(returncode=0)

    with (
        patch("bbb_dataset.cleaning.shutil.which", side_effect=which),
        patch("bbb_dataset.cleaning.subprocess.run", side_effect=fake_run),
        patch("bbb_dataset.cleaning.tempfile.TemporaryDirectory") as mock_tmp,
    ):
        mock_tmp.return_value.__enter__.return_value = str(tmp_path)
        (tmp_path / "input.fa").write_text(">seq_0\nACDEFG\n>seq_1\nACDEFA\n", encoding="utf-8")
        assignments = _run_cdhit_or_mmseqs(["ACDEFG", "ACDEFA"], threshold=0.8)
    assert assignments == [0, 0]


def test_run_cdhit_uses_subprocess_when_available(tmp_path: Path) -> None:
    clstr_content = """>Cluster 0
0	12aa, >seq_0...
0	12aa, >seq_1..."""
    with (
        patch("bbb_dataset.cleaning.shutil.which", return_value="/usr/bin/cd-hit"),
        patch("bbb_dataset.cleaning.subprocess.run") as mock_run,
        patch("bbb_dataset.cleaning.tempfile.TemporaryDirectory") as mock_tmp,
    ):
        mock_tmp.return_value.__enter__.return_value = str(tmp_path)
        (tmp_path / "input.fa").write_text(">seq_0\nACDEFG\n>seq_1\nACDEFA\n", encoding="utf-8")
        (tmp_path / "out.fa").write_text("", encoding="utf-8")
        (tmp_path / "out.fa.clstr").write_text(clstr_content, encoding="utf-8")
        mock_run.return_value = MagicMock(returncode=0)
        assignments = _run_cdhit_or_mmseqs(["ACDEFG", "ACDEFA"], threshold=0.8)
    assert assignments == [0, 0]


def test_parse_cif_backbone_uses_mmcif_parser(tmp_path: Path) -> None:
    cif = tmp_path / "x.cif"
    cif.write_text("data_\n", encoding="utf-8")
    fake_atom = MagicMock()
    fake_atom.coord = np.array([1.0, 2.0, 3.0])
    fake_atom.bfactor = 90.0
    fake_residue = MagicMock()
    fake_residue.id = (" ", 1, " ")
    fake_residue.get_resname.return_value = "ALA"
    fake_residue.__contains__ = lambda self, key: key == "CA"
    fake_residue.__getitem__ = lambda self, key: fake_atom
    fake_chain = MagicMock()
    fake_chain.__iter__ = lambda self: iter([fake_residue])
    fake_model = MagicMock()
    fake_model.__iter__ = lambda self: iter([fake_chain])
    fake_structure = [fake_model]

    with patch("Bio.PDB.MMCIFParser.MMCIFParser") as mock_parser_cls:
        mock_parser_cls.return_value.get_structure.return_value = fake_structure
        parsed = parse_cif_backbone(cif)
    assert parsed["sequence"] == ["A"]


def test_resolve_or_fold_imports_existing_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "experiments" / "bbb-fold-abc"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text('{"status": "succeeded"}', encoding="utf-8")
    cif = run_dir / "x.cif"
    cif.write_text("ATOM 1 CA ALA A 1 1.0 0.0 0.0 1.0 90.0\n", encoding="utf-8")
    with (
        patch("bbb_dataset.folding.run_dir_for_sequence", return_value=run_dir),
        patch("bbb_dataset.folding.find_structure_cif", return_value=cif),
        patch(
            "bbb_dataset.folding.parse_run_json",
            return_value={"complex_plddt": 0.8, "ptm": 0.5},
        ),
    ):
        result, source = resolve_or_fold_sequence(
            "A",
            model="boltz-2.1",
            structures_dir=tmp_path / "structures",
            experiments_dir=tmp_path / "experiments",
            resume=True,
        )
    assert source == "imported"
    assert result["length"] == 1


def test_fold_sequence_calls_boltz_api(tmp_path: Path) -> None:
    mock_client = MagicMock()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    mock_client.predictions.structure_and_binding.run.return_value = str(run_dir)
    (run_dir / "run.json").write_text(
        '{"status": "succeeded", "output": {"best_sample": {"metrics": {"complex_plddt": 0.8}}}}',
        encoding="utf-8",
    )
    cif = run_dir / "x.cif"
    cif.write_text("ATOM 1 CA ALA A 1 1.0 0.0 0.0 1.0 90.0\n", encoding="utf-8")
    with (
        patch.dict("os.environ", {"BOLTZ_API_KEY": "test-key"}),
        patch("boltz_api.Boltz", return_value=mock_client),
        patch("bbb_dataset.folding.find_structure_cif", return_value=cif),
    ):
        result = fold_sequence("A", model="boltz-2.1", structures_dir=tmp_path / "structures")
    assert result["length"] == 1


def test_build_manifest_with_mocked_fold(tmp_path: Path) -> None:
    df = pd.DataFrame({"sequence": ["YGGFLR"], "peptide_id": ["p1"]})
    metrics = {
        "sequence_hash": "hash123",
        "coords_path": str(tmp_path / "coords.npz"),
        "plddt": 90.0,
        "ptm": 0.8,
        "structure_confidence": 0.9,
        "iptm": 0.0,
        "complex_iplddt": 0.0,
        "complex_pde": 0.0,
        "complex_ipde": 0.0,
        "length": 6,
    }
    with patch(
        "bbb_dataset.folding.resolve_or_fold_sequence",
        return_value=(metrics, "folded"),
    ):
        folder = StructureFolder(FoldConfig(max_workers=1))
        manifest, stats = folder.build_manifest(
            df,
            structures_dir=tmp_path / "structures",
            experiments_dir=tmp_path / "experiments",
        )
    assert len(manifest) == 1
    assert stats["n_api_folded"] == 1


def test_import_from_run_dir_with_mocked_cif(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        '{"status": "succeeded", "output": {"best_sample": {"metrics": {"complex_plddt": 0.8}}}}',
        encoding="utf-8",
    )
    cif = run_dir / "structure.cif"
    cif.write_text(
        "ATOM 1 N ALA A 1 0.0 0.0 0.0 1.0 90.0\nATOM 2 CA ALA A 1 1.0 0.0 0.0 1.0 90.0\n",
        encoding="utf-8",
    )
    with patch("bbb_dataset.folding.find_structure_cif", return_value=cif):
        result = import_from_run_dir(run_dir, "A", structures_dir=tmp_path / "structures")
    assert result["sequence_hash"]
    assert Path(str(result["coords_path"])).exists()


def test_schema_type_validation() -> None:
    import pytest

    from bbb_dataset.schema import DatasetSchema

    bad = pd.DataFrame(
        {
            "peptide_id": ["a"],
            "source_id": ["s"],
            "sequence": ["AAAAAA"],
            "length": [6],
            "bbb_label": [1.5],
            "source_db": ["x"],
            "split": ["train"],
            "source_split": ["train"],
            "label_tier": ["gold"],
            "is_gold": [1],
            "cluster_id": [0],
            "external_test": [0],
            "fold_id": [0],
        }
    )
    with pytest.raises(ValueError, match="bbb_label"):
        DatasetSchema.validate(bad)


def test_build_manifest_records_errors(tmp_path: Path) -> None:
    df = pd.DataFrame({"sequence": ["YGGFLR"], "peptide_id": ["p1"]})
    with patch(
        "bbb_dataset.folding.resolve_or_fold_sequence",
        side_effect=RuntimeError("api down"),
    ):
        folder = StructureFolder(FoldConfig(max_workers=1))
        manifest, stats = folder.build_manifest(
            df,
            structures_dir=tmp_path / "structures",
            experiments_dir=tmp_path / "experiments",
        )
    assert manifest.empty
    assert stats["n_errors"] == 1


def test_load_augment_config_from_yaml(tmp_path: Path) -> None:
    from bbb_dataset.augmentation import load_augment_config

    path = tmp_path / "aug.yaml"
    path.write_text("enabled: false\n", encoding="utf-8")
    cfg = load_augment_config(path, min_length=5, max_length=20, random_seed=1)
    assert cfg.enabled is False
    assert cfg.min_length == 5


def test_parse_cif_backbone_from_atom_lines(tmp_path: Path) -> None:
    cif = tmp_path / "x.cif"
    cif.write_text(
        "ATOM 1 CA ALA A 1 1.0 2.0 3.0 1.0 90.0\nATOM 2 CA GLY A 2 4.0 5.0 6.0 1.0 80.0\n",
        encoding="utf-8",
    )
    parsed = parse_cif_backbone(cif)
    assert parsed["sequence"] == ["A", "G"]
    assert len(parsed["coords"]) == 2
