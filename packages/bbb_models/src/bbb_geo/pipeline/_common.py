from __future__ import annotations

import argparse

from bbb_classifier.pipeline._common import (
    add_train_args,
    finalize_run,
    load_context,
    prepare_dataframes,
    sample_weights,
)
from bbb_classifier.utils.logging import get_logger
from bbb_classifier.utils.seed import set_seed

GEO_MODEL_TYPES = frozenset({"struct_egnn_geo"})


def setup_run(args: argparse.Namespace):
    exp_cfg, data_cfg, train_cfg, run_dir = load_context(args)
    set_seed(int(train_cfg.get("seed", 42)))
    logger = get_logger("bbb-geo", run_dir / "train.log")
    if exp_cfg["model_type"] not in GEO_MODEL_TYPES:
        raise ValueError(f"{exp_cfg['model_type']} is not a geo model; use scripts/classifier/train.py")
    return exp_cfg, data_cfg, train_cfg, run_dir, logger
