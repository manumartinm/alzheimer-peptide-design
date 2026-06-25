from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bbb_classifier.config import DataConfig, TrainConfig
from bbb_classifier.io import ensure_dir, get_logger, read_yaml, set_seed
from bbb_geo.enums import GEO_MODEL_TYPES, GeoModelType


@dataclass
class GeoExperimentConfig:
    model_type: GeoModelType
    name: str
    features: dict[str, bool]
    model: dict[str, Any]
    esm: dict[str, Any]
    mixup: dict[str, Any]
    struct: dict[str, Any]
    validation: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GeoExperimentConfig:
        return cls.from_dict(read_yaml(path))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GeoExperimentConfig:
        model_type = GeoModelType(raw["model_type"])
        return cls(
            model_type=model_type,
            name=str(raw.get("name", model_type.value)),
            features=dict(raw.get("features", {})),
            model=dict(raw.get("model", {})),
            esm=dict(raw.get("esm", {})),
            mixup=dict(raw.get("mixup", {})),
            struct=dict(raw.get("struct", {})),
            validation=dict(raw.get("validation", {})),
            raw=dict(raw),
        )

    def to_dict(self) -> dict[str, Any]:
        if self.raw:
            return dict(self.raw)
        return {
            "model_type": self.model_type.value,
            "name": self.name,
            "features": self.features,
            "model": self.model,
            "esm": self.esm,
            "mixup": self.mixup,
            "struct": self.struct,
            "validation": self.validation,
        }


@dataclass
class GeoRunContext:
    exp: GeoExperimentConfig
    data: DataConfig
    train: TrainConfig
    run_dir: Path
    logger: logging.Logger

    @classmethod
    def from_train_args(cls, args: argparse.Namespace) -> GeoRunContext:
        exp = GeoExperimentConfig.from_yaml(args.exp)
        data = DataConfig.from_yaml(args.data_config, ensure_hf=True)
        train = TrainConfig.from_yaml(args.train_config)
        if exp.model_type not in GEO_MODEL_TYPES:
            raise ValueError(f"{exp.model_type.value} is not a geo model; use bbb-classifier train")
        set_seed(train.seed)
        run_dir = ensure_dir(Path(args.output_root) / "models" / exp.name)
        logger = get_logger("bbb-geo", run_dir / "train.log")
        return cls(exp=exp, data=data, train=train, run_dir=run_dir, logger=logger)

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> tuple[GeoExperimentConfig, DataConfig, list[str]]:
        meta = json.loads((run_dir / "train_metadata.json").read_text(encoding="utf-8"))
        exp = GeoExperimentConfig.from_dict(meta["exp_cfg"])
        data = DataConfig.from_dict(meta["data_cfg"])
        tab_cols = list(meta.get("tab_cols", []))
        return exp, data, tab_cols
