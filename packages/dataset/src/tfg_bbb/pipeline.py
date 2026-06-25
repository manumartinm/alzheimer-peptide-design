from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

import pandas as pd
import yaml

from .augment import AugmentConfig, augment_gold_dataframe, load_augment_config
from .clean import deduplicate_by_identity, filter_sequences, resolve_label_conflicts
from .eda import run_eda, run_augmentation_eda, run_gold_eda
from .features import add_feature_columns
from .folding import FoldConfig, build_struct_manifest
from .io import ensure_dirs, write_parquet
from .schema import validate_dataset_schema
from .sources import load_b3pdb, load_b3pred_d1, load_brainpeps
from .splits import assign_cluster_folds, mark_external_holdout


@dataclass
class BuildConfig:
    base_dir: Path
    min_length: int = 6
    max_length: int = 30
    identity_threshold: float = 0.9
    random_seed: int = 42
    use_b3pdb: bool = True
    use_brainpeps: bool = True
    b3pdb_path: Path | None = None
    brainpeps_path: Path | None = None

    @property
    def raw_dir(self) -> Path:
        return self.base_dir / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.base_dir / "data" / "processed"

    @property
    def interim_dir(self) -> Path:
        return self.base_dir / "data" / "interim"


def _attach_peptide_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["peptide_id"] = out["sequence"].map(lambda s: sha1(s.encode()).hexdigest()[:12])
    return out


def _clean_and_featurize(df: pd.DataFrame, cfg: BuildConfig, is_gold: int) -> tuple[pd.DataFrame, dict[str, int]]:
    cleaned, stats_filter = filter_sequences(df, min_length=cfg.min_length, max_length=cfg.max_length)
    cleaned, stats_conflicts = resolve_label_conflicts(cleaned)
    cleaned, stats_dedup = deduplicate_by_identity(
        cleaned,
        threshold=cfg.identity_threshold,
        keep_cluster_id=True,
    )
    cleaned = _attach_peptide_id(cleaned)
    cleaned = add_feature_columns(cleaned, sequence_col="sequence")
    cleaned["is_gold"] = int(is_gold)
    cleaned = mark_external_holdout(cleaned, split_col="split", holdout_value="val")
    cleaned = assign_cluster_folds(
        cleaned,
        label_col="bbb_label",
        cluster_col="cluster_id",
        n_splits=5,
        seed=cfg.random_seed,
        only_non_holdout=True,
    )
    cleaned["bbb_label"] = cleaned["bbb_label"].astype(int)
    cleaned["external_test"] = cleaned["external_test"].astype(int)
    validate_dataset_schema(cleaned)
    stats = {}
    stats.update(stats_filter)
    stats.update(stats_conflicts)
    stats.update(stats_dedup)
    return cleaned, stats


def build_gold_dataset(cfg: BuildConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    ensure_dirs(cfg.base_dir)
    tables = [load_b3pred_d1(cfg.raw_dir)]
    if cfg.use_b3pdb and cfg.b3pdb_path:
        b3pdb_df = load_b3pdb(cfg.b3pdb_path)
        if not b3pdb_df.empty:
            tables.append(b3pdb_df)
    if cfg.use_brainpeps and cfg.brainpeps_path:
        brain_df = load_brainpeps(cfg.brainpeps_path)
        if not brain_df.empty:
            tables.append(brain_df)

    merged = pd.concat(tables, ignore_index=True).fillna("")
    dataset, stats = _clean_and_featurize(merged, cfg, is_gold=1)
    write_parquet(dataset, cfg.processed_dir / "peptides_bbb.parquet")
    dataset.head(1000).to_csv(cfg.processed_dir / "peptides_bbb_preview.csv", index=False)
    return dataset, stats


def build_augmented_gold_dataset(
    cfg: BuildConfig,
    gold_df: pd.DataFrame | None = None,
    aug_cfg: AugmentConfig | None = None,
    *,
    augment_config_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Generate augmented training rows and write processed artifacts."""
    ensure_dirs(cfg.base_dir)
    if gold_df is None:
        gold_path = cfg.processed_dir / "peptides_bbb.parquet"
        if not gold_path.exists():
            raise FileNotFoundError(f"Gold dataset not found: {gold_path}")
        gold_df = pd.read_parquet(gold_path)

    if aug_cfg is None:
        config_path = augment_config_path or (cfg.base_dir / "configs" / "augmentation.yaml")
        aug_cfg = load_augment_config(
            config_path,
            min_length=cfg.min_length,
            max_length=cfg.max_length,
            random_seed=cfg.random_seed,
        )

    augmented_df, combined_df, stats = augment_gold_dataframe(gold_df, aug_cfg)

    extra_path = cfg.processed_dir / "peptides_bbb_augmented_extra.parquet"
    combined_path = cfg.processed_dir / "peptides_bbb_with_augmentation.parquet"
    stats_path = cfg.processed_dir / "augmentation_stats.json"

    if not augmented_df.empty:
        validate_dataset_schema(augmented_df.drop(columns=["is_augmented", "parent_peptide_id"], errors="ignore"))
    validate_dataset_schema(combined_df.drop(columns=["is_augmented", "parent_peptide_id"], errors="ignore"))

    write_parquet(augmented_df, extra_path)
    write_parquet(combined_df, combined_path)
    stats["gold_rows"] = int(len(gold_df))
    stats["combined_rows"] = int(len(combined_df))
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return augmented_df, combined_df, stats


def load_fold_config(path: Path) -> FoldConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload = payload or {}
    return FoldConfig(**payload)


def build_peptide_struct_manifest(
    cfg: BuildConfig,
    input_df: pd.DataFrame | None = None,
    *,
    fold_config_path: Path | None = None,
    input_parquet: Path | None = None,
    manifest_only: bool = False,
    resume: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fold unique sequences via Boltz API and write the structural manifest."""
    ensure_dirs(cfg.base_dir)
    fold_cfg = load_fold_config(fold_config_path or (cfg.base_dir / "configs" / "folding.yaml"))
    if resume is not None:
        fold_cfg.resume = resume

    if input_df is None:
        source = input_parquet or (cfg.processed_dir / "peptides_bbb_with_augmentation.parquet")
        if not Path(source).exists():
            source = cfg.processed_dir / "peptides_bbb.parquet"
        input_df = pd.read_parquet(source)

    structures_dir = cfg.base_dir / fold_cfg.structures_dir
    experiments_dir = cfg.base_dir / fold_cfg.experiments_dir
    manifest_df, stats = build_struct_manifest(
        input_df,
        structures_dir=structures_dir,
        experiments_dir=experiments_dir,
        fold_cfg=fold_cfg,
        manifest_only=manifest_only,
    )
    output_path = cfg.processed_dir / fold_cfg.output_name
    write_parquet(manifest_df, output_path)
    stats["manifest_path"] = str(output_path.resolve())
    return manifest_df, stats
