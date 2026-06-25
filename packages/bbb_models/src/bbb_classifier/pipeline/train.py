from __future__ import annotations

import argparse
import os

import mlflow

from bbb_classifier.pipeline._common import (
    add_train_args,
    finalize_run,
    prepare_dataframes,
    sample_weights,
    setup_run,
)
from bbb_classifier.pipeline.features import build_features
from bbb_classifier.pipeline.models import fit_and_predict


def run(args: argparse.Namespace) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return

    exp_cfg, data_cfg, train_cfg, run_dir, logger = setup_run(args)
    train_df, val_df = prepare_dataframes(args, exp_cfg, data_cfg)
    logger.info("Dataset loaded: %d train / %d val", len(train_df), len(val_df))

    tr_feat = build_features(train_df, data_cfg, exp_cfg)
    va_feat = build_features(val_df, data_cfg, exp_cfg)
    y_train = train_df[data_cfg["label_col"]].to_numpy(dtype=int)
    y_val = val_df[data_cfg["label_col"]].to_numpy(dtype=int)
    weights = sample_weights(train_df)

    tracking_cfg = train_cfg.get("tracking", {})
    if tracking_cfg.get("mlflow", False):
        mlflow.set_experiment(tracking_cfg.get("mlflow_experiment", "bbb_classifier"))

    val_prob = fit_and_predict(
        exp_cfg["model_type"],
        tr_feat,
        va_feat,
        y_train,
        y_val,
        weights,
        run_dir,
        train_cfg,
        exp_cfg,
    )
    finalize_run(
        run_dir=run_dir,
        exp_cfg=exp_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        tab_cols=tr_feat["tab_cols"],
        val_df=val_df,
        y_val=y_val,
        val_prob=val_prob,
        logger=logger,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a tabular/ESM BBB classifier.")
    add_train_args(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
