from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path

import pandas as pd
import yaml

from .augmentation import AugmentConfig, Augmenter, load_augment_config
from .cleaning import SequenceCleaner
from .enums import ProcessedArtifact, SourceDb
from .features import FeatureComputer
from .folding import FoldConfig, StructureFolder
from .paths import PathLayout
from .schema import DatasetSchema
from .sources import SourceRegistry
from .splits import FoldSplitter


@dataclass
class BuildConfig:
    layout: PathLayout
    min_length: int = 6
    max_length: int = 30
    identity_threshold: float = 0.9
    random_seed: int = 42
    sources: frozenset[SourceDb] = field(
        default_factory=lambda: frozenset({SourceDb.B3PRED_D1, SourceDb.B3PDB, SourceDb.BRAINPEPS})
    )
    b3pdb_path: Path | None = None
    brainpeps_path: Path | None = None

    @classmethod
    def from_base_dir(cls, base_dir: Path, **kwargs: object) -> BuildConfig:
        layout = PathLayout(base_dir=base_dir)
        defaults = {
            "b3pdb_path": base_dir / "data" / "raw" / "b3pdb.tsv",
            "brainpeps_path": base_dir / "data" / "raw" / "brainpeps.tsv",
        }
        defaults.update(kwargs)
        return cls(layout=layout, **defaults)  # type: ignore[arg-type]

    @property
    def base_dir(self) -> Path:
        return self.layout.base_dir

    @property
    def processed_dir(self) -> Path:
        return self.layout.processed_dir


class DatasetBuilder:
    def __init__(self, config: BuildConfig) -> None:
        self.config = config
        self._cleaner = SequenceCleaner(
            min_length=config.min_length,
            max_length=config.max_length,
            identity_threshold=config.identity_threshold,
        )
        self._features = FeatureComputer()
        self._splitter = FoldSplitter(n_splits=5, seed=config.random_seed)
        self._sources = SourceRegistry(
            raw_dir=config.layout.raw_dir,
            b3pdb_path=config.b3pdb_path,
            brainpeps_path=config.brainpeps_path,
        )

    def build_gold(self) -> tuple[pd.DataFrame, dict[str, int]]:
        self.config.layout.ensure_dirs()
        merged = self._sources.load_all(self.config.sources)
        dataset, stats = self._clean_and_featurize(merged, is_gold=1)
        out_path = self.config.layout.artifact(ProcessedArtifact.PEPTIDES_BBB)
        PathLayout.write_parquet(dataset, out_path)
        dataset.head(1000).to_csv(
            self.config.layout.artifact(ProcessedArtifact.PREVIEW_CSV), index=False
        )
        return dataset, stats

    def build_augmented(
        self,
        gold_df: pd.DataFrame | None = None,
        aug_cfg: AugmentConfig | None = None,
        *,
        augment_config_path: Path | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
        self.config.layout.ensure_dirs()
        if gold_df is None:
            gold_path = self.config.layout.artifact(ProcessedArtifact.PEPTIDES_BBB)
            if not gold_path.exists():
                raise FileNotFoundError(f"Gold dataset not found: {gold_path}")
            gold_df = PathLayout.read_parquet(gold_path)

        if aug_cfg is None:
            config_path = augment_config_path or (
                self.config.base_dir / "configs" / "augmentation.yaml"
            )
            aug_cfg = load_augment_config(
                config_path,
                min_length=self.config.min_length,
                max_length=self.config.max_length,
                random_seed=self.config.random_seed,
            )

        augmented_df, combined_df, stats = Augmenter(config=aug_cfg).run(gold_df)

        extra_path = self.config.layout.artifact(ProcessedArtifact.AUGMENTED_EXTRA)
        combined_path = self.config.layout.artifact(ProcessedArtifact.COMBINED)
        stats_path = self.config.layout.artifact(ProcessedArtifact.AUGMENTATION_STATS)

        if not augmented_df.empty:
            DatasetSchema.validate(
                augmented_df.drop(columns=["is_augmented", "parent_peptide_id"], errors="ignore")
            )
        DatasetSchema.validate(
            combined_df.drop(columns=["is_augmented", "parent_peptide_id"], errors="ignore")
        )

        PathLayout.write_parquet(augmented_df, extra_path)
        PathLayout.write_parquet(combined_df, combined_path)
        stats["gold_rows"] = len(gold_df)
        stats["combined_rows"] = len(combined_df)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return augmented_df, combined_df, stats

    def build_manifest(
        self,
        input_df: pd.DataFrame | None = None,
        fold_cfg: FoldConfig | None = None,
        *,
        fold_config_path: Path | None = None,
        input_parquet: Path | None = None,
        manifest_only: bool = False,
        resume: bool | None = None,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        self.config.layout.ensure_dirs()
        if fold_cfg is None:
            fold_cfg = self._load_fold_config(
                fold_config_path or (self.config.base_dir / "configs" / "folding.yaml")
            )
        if resume is not None:
            fold_cfg.resume = resume

        if input_df is None:
            source = input_parquet or self.config.layout.artifact(ProcessedArtifact.COMBINED)
            if not Path(source).exists():
                source = self.config.layout.artifact(ProcessedArtifact.PEPTIDES_BBB)
            input_df = PathLayout.read_parquet(source)

        structures_dir = self.config.base_dir / fold_cfg.structures_dir
        experiments_dir = self.config.base_dir / fold_cfg.experiments_dir
        folder = StructureFolder(fold_cfg)
        manifest_df, stats = folder.build_manifest(
            input_df,
            structures_dir=structures_dir,
            experiments_dir=experiments_dir,
            manifest_only=manifest_only,
        )
        output_path = self.config.layout.processed_dir / fold_cfg.output_name
        PathLayout.write_parquet(manifest_df, output_path)
        stats["manifest_path"] = str(output_path.resolve())
        return manifest_df, stats

    def _clean_and_featurize(
        self, df: pd.DataFrame, is_gold: int
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        cleaned, stats = self._cleaner.run(df)
        cleaned = self._attach_peptide_id(cleaned)
        cleaned = self._features.add_columns(cleaned, sequence_col="sequence")
        cleaned["is_gold"] = int(is_gold)
        cleaned = self._splitter.mark_external_holdout(cleaned, split_col="split")
        cleaned = self._splitter.assign_cluster_folds(
            cleaned,
            label_col="bbb_label",
            cluster_col="cluster_id",
            only_non_holdout=True,
        )
        cleaned["bbb_label"] = cleaned["bbb_label"].astype(int)
        cleaned["external_test"] = cleaned["external_test"].astype(int)
        DatasetSchema.validate(cleaned)
        return cleaned, stats

    @staticmethod
    def _attach_peptide_id(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["peptide_id"] = out["sequence"].map(lambda s: sha1(s.encode()).hexdigest()[:12])
        return out

    @staticmethod
    def _load_fold_config(path: Path) -> FoldConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        payload = payload or {}
        return FoldConfig(**payload)


def build_gold_dataset(cfg: BuildConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    return DatasetBuilder(cfg).build_gold()


def build_augmented_gold_dataset(
    cfg: BuildConfig,
    gold_df: pd.DataFrame | None = None,
    aug_cfg: AugmentConfig | None = None,
    *,
    augment_config_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    return DatasetBuilder(cfg).build_augmented(
        gold_df=gold_df,
        aug_cfg=aug_cfg,
        augment_config_path=augment_config_path,
    )


def build_peptide_struct_manifest(
    cfg: BuildConfig,
    input_df: pd.DataFrame | None = None,
    *,
    fold_config_path: Path | None = None,
    input_parquet: Path | None = None,
    manifest_only: bool = False,
    resume: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    return DatasetBuilder(cfg).build_manifest(
        input_df=input_df,
        fold_config_path=fold_config_path,
        input_parquet=input_parquet,
        manifest_only=manifest_only,
        resume=resume,
    )
