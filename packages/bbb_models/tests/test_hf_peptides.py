from __future__ import annotations

from pathlib import Path

from bbb_classifier.dataset import (
    DEFAULT_CACHE_DIR,
    HF_DATASET_REPO,
    cache_dir_from_config,
    resolve_data_paths,
)


def test_default_cache_dir_under_package() -> None:
    assert DEFAULT_CACHE_DIR.name == "bbb-peptides"
    assert DEFAULT_CACHE_DIR.parent.name == "data"


def test_resolve_data_paths_relative() -> None:
    cfg = {
        "dataset_path": "data/bbb-peptides/peptides.parquet",
        "dataset_root": "data/bbb-peptides",
    }
    resolved = resolve_data_paths(cfg)
    assert resolved["dataset_path"].endswith("data/bbb-peptides/peptides.parquet")
    assert resolved["dataset_root"].endswith("data/bbb-peptides")
    assert Path(resolved["dataset_path"]).is_absolute()


def test_cache_dir_from_config() -> None:
    cfg = {"dataset_root": "data/bbb-peptides"}
    assert cache_dir_from_config(cfg) == DEFAULT_CACHE_DIR.resolve()


def test_hf_repo_default() -> None:
    assert HF_DATASET_REPO == "manumartinm/bbb-peptides"
