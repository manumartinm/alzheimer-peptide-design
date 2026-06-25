from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bbb_dataset.builder import BuildConfig, DatasetBuilder
from bbb_dataset.cleaning import deduplicate_by_identity, filter_sequences, resolve_label_conflicts
from bbb_dataset.enums import ProcessedArtifact
from bbb_dataset.paths import PathLayout
from bbb_dataset.struct_io import load_coords_npz, write_coords_npz


def test_module_level_cleaning_wrappers() -> None:
    df = pd.DataFrame({"sequence": ["YGGFLR", "YGGFLR"], "bbb_label": [1, 0]})
    filtered, _ = filter_sequences(df, min_length=6, max_length=30)
    assert len(filtered) == 2
    resolved, _ = resolve_label_conflicts(filtered)
    assert len(resolved) == 0


def test_deduplicate_module_wrapper() -> None:
    df = pd.DataFrame({"sequence": ["ACDEFG", "ACDEFA", "TTTTTT"], "bbb_label": [0, 1, 0]})
    out, _ = deduplicate_by_identity(df, threshold=0.8)
    assert len(out) >= 1


def test_build_augmented_with_in_memory_gold(tmp_path: Path) -> None:
    cfg = BuildConfig.from_base_dir(tmp_path)
    builder = DatasetBuilder(cfg)
    gold = pd.DataFrame(
        [
            {
                "peptide_id": "abc123",
                "source_id": "P1",
                "sequence": "YGGFLR",
                "length": 6,
                "bbb_label": 1,
                "source_db": "B3Pred_D1",
                "split": "train",
                "source_split": "train",
                "label_tier": "gold",
                "is_gold": 1,
                "cluster_id": 0,
                "external_test": 0,
                "fold_id": 1,
                "mw": 1.0,
            }
        ]
    )
    from bbb_dataset.augmentation import AugmentConfig

    aug_cfg = AugmentConfig(
        n_augmented_per_sample=1,
        random_state=0,
        seq_substitution_prob=1.0,
        seq_truncation_prob=0.0,
    )
    _, combined, stats = builder.build_augmented(gold_df=gold, aug_cfg=aug_cfg)
    assert stats["gold_rows"] == 1
    assert cfg.layout.artifact(ProcessedArtifact.COMBINED).exists()


def test_build_manifest_manifest_only(tmp_path: Path) -> None:
    cfg = BuildConfig.from_base_dir(tmp_path)
    builder = DatasetBuilder(cfg)
    df = pd.DataFrame({"sequence": ["YGGFLR"], "peptide_id": ["p1"]})
    manifest, stats = builder.build_manifest(input_df=df, manifest_only=True)
    assert manifest.empty
    assert "manifest_path" in stats


def test_coords_npz_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "coords.npz"
    parsed = {
        "coords": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "sequence": ["A"],
        "plddt_per_residue": np.array([90.0], dtype=np.float32),
    }
    write_coords_npz(path, parsed)
    loaded = load_coords_npz(path)
    assert loaded["sequence"] == "A"
    assert loaded["coords"].shape == (1, 3)


def test_path_layout_read_write(tmp_path: Path) -> None:
    layout = PathLayout(base_dir=tmp_path)
    layout.ensure_dirs()
    df = pd.DataFrame({"x": [1]})
    out = layout.processed_dir / "test.parquet"
    PathLayout.write_parquet(df, out)
    assert PathLayout.read_parquet(out).iloc[0]["x"] == 1
