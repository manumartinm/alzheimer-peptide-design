from __future__ import annotations

import argparse
import os

import mlflow

from bbb_classifier.pipeline._common import (
    add_train_args,
    finalize_run,
    prepare_dataframes,
    sample_weights,
)
from bbb_geo.pipeline._common import setup_run
from bbb_geo.pipeline.features import apply_plddt_weights, build_features
from bbb_geo.pipeline.models import fit_and_predict


def run(args: argparse.Namespace) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return

    exp_cfg, data_cfg, train_cfg, run_dir, logger = setup_run(args)
    train_df, val_df = prepare_dataframes(args, exp_cfg, data_cfg)
    tr_feat = build_features(train_df, data_cfg, exp_cfg)
    va_feat = build_features(val_df, data_cfg, exp_cfg)
    train_df = apply_plddt_weights(tr_feat["struct_df"], exp_cfg)
    val_df = va_feat["struct_df"]
    logger.info("Dataset loaded: %d train / %d val (with structures)", len(train_df), len(val_df))

    y_train = train_df[data_cfg["label_col"]].to_numpy(dtype=int)
    y_val = val_df[data_cfg["label_col"]].to_numpy(dtype=int)
    weights = sample_weights(train_df)

    if train_cfg.get("tracking", {}).get("mlflow", False):
        mlflow.set_experiment(
            train_cfg.get("tracking", {}).get("mlflow_experiment", "bbb_classifier")
        )

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
        resume=not args.no_resume,
    )
    finalize_run(
        run_dir=run_dir,
        exp_cfg=exp_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        tab_cols=tr_feat.get("tab_cols", []),
        val_df=val_df,
        y_val=y_val,
        val_prob=val_prob,
        logger=logger,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a geometry EGNN BBB model.")
    add_train_args(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
