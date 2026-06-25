from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bbb_classifier.dataset import load_data_config
from bbb_classifier.enums import CLASSIFIER_MODEL_TYPES, ModelType
from bbb_classifier.io import ensure_dir, get_logger, read_yaml, set_seed


@dataclass
class DataConfig:
    dataset_path: str
    dataset_root: str
    id_col: str = "peptide_id"
    label_col: str = "bbb_label"
    sequence_col: str = "sequence"
    fold_col: str | None = "fold_id"
    test_size: float = 0.2
    random_state: int = 42
    tabular_exclude: list[str] = field(default_factory=list)
    three_d_columns: list[str] = field(default_factory=list)
    struct_manifest_path: str = ""
    dataset_repo: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path, *, ensure_hf: bool = False) -> DataConfig:
        raw = load_data_config(path, ensure=ensure_hf)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DataConfig:
        return cls(
            dataset_path=str(raw["dataset_path"]),
            dataset_root=str(raw.get("dataset_root", "")),
            id_col=str(raw.get("id_col", "peptide_id")),
            label_col=str(raw.get("label_col", "bbb_label")),
            sequence_col=str(raw.get("sequence_col", "sequence")),
            fold_col=raw.get("fold_col"),
            test_size=float(raw.get("test_size", 0.2)),
            random_state=int(raw.get("random_state", 42)),
            tabular_exclude=list(raw.get("tabular_exclude", [])),
            three_d_columns=list(raw.get("three_d_columns", [])),
            struct_manifest_path=str(raw.get("struct_manifest_path", "") or ""),
            dataset_repo=raw.get("dataset_repo"),
            raw=dict(raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return (
            dict(self.raw)
            if self.raw
            else {
                "dataset_path": self.dataset_path,
                "dataset_root": self.dataset_root,
                "id_col": self.id_col,
                "label_col": self.label_col,
                "sequence_col": self.sequence_col,
                "fold_col": self.fold_col,
                "test_size": self.test_size,
                "random_state": self.random_state,
                "tabular_exclude": self.tabular_exclude,
                "three_d_columns": self.three_d_columns,
                "struct_manifest_path": self.struct_manifest_path,
                **({"dataset_repo": self.dataset_repo} if self.dataset_repo else {}),
            }
        )


@dataclass
class ExperimentConfig:
    model_type: ModelType
    name: str
    features: dict[str, bool]
    model: dict[str, Any]
    esm: dict[str, Any]
    mixup: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        return cls.from_dict(read_yaml(path))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        model_type = ModelType(raw["model_type"])
        return cls(
            model_type=model_type,
            name=str(raw.get("name", model_type.value)),
            features=dict(raw.get("features", {})),
            model=dict(raw.get("model", {})),
            esm=dict(raw.get("esm", {})),
            mixup=dict(raw.get("mixup", {})),
            raw=dict(raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return (
            dict(self.raw)
            if self.raw
            else {
                "model_type": self.model_type.value,
                "name": self.name,
                "features": self.features,
                "model": self.model,
                "esm": self.esm,
                "mixup": self.mixup,
            }
        )


@dataclass
class TrainConfig:
    seed: int
    training: dict[str, Any]
    calibration: dict[str, Any]
    tracking: dict[str, Any]
    output: dict[str, Any]
    primary_metric: str
    maximize_metric: bool
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        return cls.from_dict(read_yaml(path))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        return cls(
            seed=int(raw.get("seed", 42)),
            training=dict(raw.get("training", {})),
            calibration=dict(raw.get("calibration", {})),
            tracking=dict(raw.get("tracking", {})),
            output=dict(raw.get("output", {})),
            primary_metric=str(raw.get("primary_metric", "pr_auc")),
            maximize_metric=bool(raw.get("maximize_metric", True)),
            raw=dict(raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return (
            dict(self.raw)
            if self.raw
            else {
                "seed": self.seed,
                "training": self.training,
                "calibration": self.calibration,
                "tracking": self.tracking,
                "output": self.output,
                "primary_metric": self.primary_metric,
                "maximize_metric": self.maximize_metric,
            }
        )


@dataclass
class RunContext:
    exp: ExperimentConfig
    data: DataConfig
    train: TrainConfig
    run_dir: Path
    logger: logging.Logger

    @classmethod
    def from_train_args(cls, args: argparse.Namespace) -> RunContext:
        exp = ExperimentConfig.from_yaml(args.exp)
        data = DataConfig.from_yaml(args.data_config, ensure_hf=True)
        train = TrainConfig.from_yaml(args.train_config)
        set_seed(train.seed)
        run_dir = ensure_dir(Path(args.output_root) / "models" / exp.name)
        logger = get_logger("bbb-classifier", run_dir / "train.log")
        if exp.model_type not in CLASSIFIER_MODEL_TYPES:
            raise ValueError(f"{exp.model_type.value} is not a classifier model; use bbb-geo train")
        return cls(exp=exp, data=data, train=train, run_dir=run_dir, logger=logger)

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> tuple[ExperimentConfig, DataConfig, list[str]]:
        import json

        meta = json.loads((run_dir / "train_metadata.json").read_text(encoding="utf-8"))
        exp = ExperimentConfig.from_dict(meta["exp_cfg"])
        data = DataConfig.from_dict(meta["data_cfg"])
        tab_cols = list(meta.get("tab_cols", []))
        return exp, data, tab_cols
