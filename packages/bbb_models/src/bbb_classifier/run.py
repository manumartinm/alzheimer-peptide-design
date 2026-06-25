from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml

from bbb_classifier.config import (
    RunContext,
    TrainConfig,
)
from bbb_classifier.dataset import load_data_config, train_val_split
from bbb_classifier.enums import CLASSIFIER_MODEL_TYPES, ModelType
from bbb_classifier.features import FeatureBuilder, rank_candidates, tabular_matrix
from bbb_classifier.io import ensure_dir, get_logger, read_yaml, write_json
from bbb_classifier.models import (
    ESMLGBMModel,
    ModelRegistry,
    TabularLGBMModel,
)
from bbb_classifier.training import (
    ProbabilityCalibrator,
    TorchData,
    classification_metrics,
    train_torch_model,
)


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exp", required=True, help="Experiment yaml path")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument(
        "--no-resume", action="store_true", help="Start from scratch even if checkpoints exist"
    )


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def _reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs = [], []
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (y_prob >= left) & (y_prob < right if i < n_bins - 1 else y_prob <= right)
        if not np.any(mask):
            continue
        centers.append(float(np.mean(y_prob[mask])))
        accs.append(float(np.mean(y_true[mask])))
    return np.array(centers), np.array(accs)


class ClassifierRun:
    def __init__(
        self, ctx: RunContext, *, dataset_path: str | None = None, no_resume: bool = False
    ):
        self.ctx = ctx
        self.dataset_path = dataset_path
        self.no_resume = no_resume

    @classmethod
    def from_train_args(cls, args: argparse.Namespace) -> ClassifierRun:
        return cls(
            RunContext.from_train_args(args),
            dataset_path=args.dataset_path,
            no_resume=args.no_resume,
        )

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> ClassifierRun:
        exp, data, _ = RunContext.from_run_dir(run_dir)
        ctx = RunContext(
            exp=exp,
            data=data,
            train=TrainConfig.from_dict({}),
            run_dir=run_dir,
            logger=get_logger("bbb-classifier"),
        )
        return cls(ctx)

    def _load_dataframe(self) -> pd.DataFrame:
        path = self.dataset_path or self.ctx.data.dataset_path
        if self.dataset_path:
            path = str(Path(self.dataset_path).resolve())
        return pd.read_parquet(path)

    def _split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return train_val_split(
            df=df,
            label_col=self.ctx.data.label_col,
            test_size=self.ctx.data.test_size,
            random_state=self.ctx.data.random_state,
            fold_col=self.ctx.data.fold_col,
        )

    @staticmethod
    def _sample_weights(train_df: pd.DataFrame) -> np.ndarray:
        if "sample_weight" in train_df.columns:
            return train_df["sample_weight"].fillna(1.0).to_numpy(dtype=np.float32)
        return np.ones(len(train_df), dtype=np.float32)

    def _fit_and_predict(
        self,
        train_bundle,
        val_bundle,
        y_train: np.ndarray,
        y_val: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        model_type = self.ctx.exp.model_type
        run_dir = self.ctx.run_dir
        train_cfg = self.ctx.train
        exp = self.ctx.exp

        if model_type == ModelType.TABULAR_LGBM:
            model = TabularLGBMModel(random_state=train_cfg.seed)
            model.fit(train_bundle.tab, y_train)
            val_prob = model.predict_proba(val_bundle.tab)
            ckpt_dir = ensure_dir(run_dir / "checkpoints")
            model.save(ckpt_dir / "best.pkl")
            model.save(ckpt_dir / "last.pkl")
            return val_prob

        if model_type == ModelType.ESM_LGBM:
            model = ESMLGBMModel(random_state=train_cfg.seed)
            model.fit(train_bundle.esm, y_train)
            val_prob = model.predict_proba(val_bundle.esm)
            ckpt_dir = ensure_dir(run_dir / "checkpoints")
            model.save(ckpt_dir / "best.pkl")
            model.save(ckpt_dir / "last.pkl")
            return val_prob

        model = ModelRegistry.create_torch(model_type, train_bundle, exp.model)
        torch_train_cfg = dict(train_cfg.training)
        torch_train_cfg["save_periodic_every"] = int(train_cfg.output.get("save_periodic_every", 0))
        torch_train_cfg["mixup"] = exp.mixup
        tr_data = TorchData(
            y=y_train,
            tab=train_bundle.tab,
            esm=train_bundle.esm,
            feat3d=train_bundle.feat3d,
            sample_weight=weights,
            graphs=train_bundle.graphs,
        )
        va_data = TorchData(
            y=y_val,
            tab=val_bundle.tab,
            esm=val_bundle.esm,
            feat3d=val_bundle.feat3d,
            graphs=val_bundle.graphs,
        )
        _, _, val_prob = train_torch_model(
            model=model,
            train_data=tr_data,
            val_data=va_data,
            run_dir=run_dir,
            train_cfg=torch_train_cfg,
            tracking_cfg=train_cfg.tracking,
            primary_metric=train_cfg.primary_metric,
            maximize_metric=train_cfg.maximize_metric,
            resume=not self.no_resume,
        )
        return val_prob

    def _finalize(
        self,
        val_df: pd.DataFrame,
        y_val: np.ndarray,
        val_prob: np.ndarray,
        tab_cols: list[str],
    ) -> None:
        ctx = self.ctx
        cal_cfg = ctx.train.calibration
        p_cal = val_prob
        if cal_cfg.get("enabled", True):
            calibrator = ProbabilityCalibrator(method=cal_cfg.get("method", "isotonic"))
            calibrator.fit(val_prob, y_val)
            p_cal = calibrator.predict(val_prob)
            ensure_dir(ctx.run_dir / "calibrators")
            calibrator.save(str(ctx.run_dir / "calibrators" / "calibrator.pkl"))

        metrics = classification_metrics(y_val, val_prob)
        cal_metrics = classification_metrics(y_val, p_cal)
        write_json(
            ctx.run_dir / "metrics.json",
            {
                "raw": metrics,
                "calibrated": cal_metrics,
                "run_name": ctx.exp.name,
                "model_type": ctx.exp.model_type.value,
            },
        )

        label_col = ctx.data.label_col
        pred_df = val_df[[ctx.data.sequence_col, label_col]].copy()
        pred_df["p_bbb_raw"] = val_prob
        pred_df["p_bbb_calibrated"] = p_cal
        pred_df["decision"] = (pred_df["p_bbb_calibrated"] >= 0.5).astype(int)
        ensure_dir(ctx.run_dir / "predictions")
        pred_df.to_parquet(ctx.run_dir / "predictions" / "val_predictions.parquet", index=False)
        write_json(
            ctx.run_dir / "train_metadata.json",
            {
                "exp_cfg": ctx.exp.to_dict(),
                "data_cfg": ctx.data.to_dict(),
                "train_cfg": ctx.train.to_dict(),
                "tab_cols": tab_cols,
            },
        )
        ctx.logger.info("Training completed for %s", ctx.exp.name)
        ctx.logger.info("Metrics raw=%s calibrated=%s", metrics, cal_metrics)

    def train(self) -> None:
        if (
            int(os.environ.get("WORLD_SIZE", "1")) > 1
            and int(os.environ.get("LOCAL_RANK", "0")) != 0
        ):
            return

        df = self._load_dataframe()
        train_df, val_df = self._split_data(df)
        self.ctx.logger.info("Dataset loaded: %d train / %d val", len(train_df), len(val_df))

        builder = FeatureBuilder(self.ctx.data, self.ctx.exp)
        tr_bundle = builder.build(train_df)
        va_bundle = builder.build(val_df)
        y_train = train_df[self.ctx.data.label_col].to_numpy(dtype=int)
        y_val = val_df[self.ctx.data.label_col].to_numpy(dtype=int)
        weights = self._sample_weights(train_df)

        if self.ctx.train.tracking.get("mlflow", False):
            mlflow.set_experiment(
                self.ctx.train.tracking.get("mlflow_experiment", "bbb_classifier")
            )

        val_prob = self._fit_and_predict(tr_bundle, va_bundle, y_train, y_val, weights)
        self._finalize(val_df, y_val, val_prob, tr_bundle.tab_cols)

    def _infer(self, df: pd.DataFrame, tab_cols: list[str] | None = None) -> np.ndarray:
        exp = self.ctx.exp
        if exp.model_type not in CLASSIFIER_MODEL_TYPES:
            raise ValueError(f"{exp.model_type.value} is not a classifier model")

        bundle = FeatureBuilder(self.ctx.data, exp).build(df)
        if tab_cols is not None:
            filtered = [c for c in tab_cols if c in df.columns]
            bundle.tab_cols = filtered
            if exp.features.get("use_tabular", False):
                bundle.tab = tabular_matrix(df, filtered)

        model = ModelRegistry.load_for_inference(
            exp.model_type, self.ctx.run_dir, bundle, exp.model
        )
        y_placeholder = (
            df[self.ctx.data.label_col].to_numpy(dtype=int)
            if self.ctx.data.label_col in df.columns
            else np.zeros(len(df))
        )
        return ModelRegistry.predict_proba(
            exp.model_type, model, bundle, y_placeholder=y_placeholder
        )

    def _calibrate(self, prob: np.ndarray) -> np.ndarray:
        calib_path = self.ctx.run_dir / "calibrators" / "calibrator.pkl"
        if calib_path.exists():
            return ProbabilityCalibrator.load(str(calib_path)).predict(prob)
        return prob

    def predict(
        self,
        df: pd.DataFrame,
        *,
        threshold: float = 0.5,
        top_k: int = 0,
        tab_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        prob = self._infer(df, tab_cols=tab_cols)
        p_cal = self._calibrate(prob)
        out_df = df.copy()
        out_df["p_bbb_raw"] = prob
        out_df["p_bbb_calibrated"] = p_cal
        out_df["decision"] = (out_df["p_bbb_calibrated"] >= threshold).astype(int)
        if top_k > 0:
            out_df = rank_candidates(out_df, prob_col="p_bbb_calibrated", top_k=top_k)
        return out_df

    def evaluate(
        self, df: pd.DataFrame, tab_cols: list[str] | None = None
    ) -> dict[str, dict[str, float]]:
        y = df[self.ctx.data.label_col].to_numpy(dtype=int)
        prob = self._infer(df, tab_cols=tab_cols)
        p_cal = self._calibrate(prob)
        metrics = {
            "raw": classification_metrics(y, prob),
            "calibrated": classification_metrics(y, p_cal),
        }
        write_json(self.ctx.run_dir / "evaluation_metrics.json", metrics)
        out_df = df[[self.ctx.data.sequence_col, self.ctx.data.label_col]].copy()
        out_df["p_bbb_raw"] = prob
        out_df["p_bbb_calibrated"] = p_cal
        out_df.to_parquet(self.ctx.run_dir / "evaluation_predictions.parquet", index=False)
        return metrics


def run_train(args: argparse.Namespace) -> None:
    ClassifierRun.from_train_args(args).train()


def run_predict(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    exp, data, tab_cols = RunContext.from_run_dir(run_dir)
    ctx = RunContext(
        exp=exp,
        data=data,
        train=TrainConfig.from_dict({}),
        run_dir=run_dir,
        logger=get_logger("bbb-classifier"),
    )
    df = _read_table(Path(args.input))
    out_df = ClassifierRun(ctx).predict(
        df, threshold=args.threshold, top_k=args.top_k, tab_cols=tab_cols
    )
    _write_table(out_df, Path(args.output))


def run_evaluate(args: argparse.Namespace) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    run_dir = Path(args.run_dir)
    exp, data, tab_cols = RunContext.from_run_dir(run_dir)
    ctx = RunContext(
        exp=exp,
        data=data,
        train=TrainConfig.from_dict({}),
        run_dir=run_dir,
        logger=get_logger("bbb-classifier"),
    )
    df = pd.read_parquet(args.dataset_path)
    metrics = ClassifierRun(ctx).evaluate(df, tab_cols=tab_cols)
    print(metrics)


def _package_cmd(module: str, command: str, *extra: str) -> list[str]:
    return [sys.executable, "-m", module, command, *extra]


def run_cv(args: argparse.Namespace) -> None:
    exp_cfg = read_yaml(args.exp)
    data_cfg = load_data_config(args.data_config, ensure=True)
    train_cfg = read_yaml(args.train_config)
    dataset_path = args.dataset_path or data_cfg["dataset_path"]
    df = pd.read_parquet(dataset_path)
    if "fold_id" not in df.columns:
        raise ValueError("Dataset must contain fold_id for cv.")

    train_cfg_mod = train_cfg.copy()
    train_cfg_mod["tracking"] = {"mlflow": False, "tensorboard": False}
    train_cfg_mod["calibration"] = dict(train_cfg.get("calibration", {}))
    if args.calibration == "none":
        train_cfg_mod["calibration"]["enabled"] = False
    else:
        train_cfg_mod["calibration"] = {"enabled": True, "method": args.calibration}

    exp_name = exp_cfg.get("name", "exp")
    cv_root = ensure_dir(Path(args.output_root) / exp_name / args.calibration)
    train_script = _package_cmd("bbb_classifier", "train")
    all_metrics, all_preds = [], []

    with tempfile.TemporaryDirectory(prefix="bbb_cv_") as tmp:
        tmp_dir = Path(tmp)
        train_cfg_tmp = tmp_dir / "train_cv.yaml"
        train_cfg_tmp.write_text(yaml.safe_dump(train_cfg_mod), encoding="utf-8")
        for fold in range(5):
            fold_df = df.copy()
            fold_df["fold_id"] = (fold_df["fold_id"] == fold).astype(int).replace({1: 0, 0: 1})
            fold_ds = tmp_dir / f"fold_{fold}.parquet"
            fold_df.to_parquet(fold_ds, index=False)
            run_root = cv_root / f"fold_{fold}"
            subprocess.run(
                [
                    *train_script,
                    "--exp",
                    args.exp,
                    "--data-config",
                    args.data_config,
                    "--train-config",
                    str(train_cfg_tmp),
                    "--output-root",
                    str(run_root),
                    "--dataset-path",
                    str(fold_ds),
                ],
                check=True,
            )
            run_dir = run_root / "models" / exp_name
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            metrics["fold"] = fold
            all_metrics.append(metrics)
            pred_df = pd.read_parquet(run_dir / "predictions" / "val_predictions.parquet")
            pred_df["fold"] = fold
            all_preds.append(pred_df)

    summary = {"experiment": exp_name, "calibration": args.calibration, "folds": all_metrics}
    for split in ("raw", "calibrated"):
        for key in ("pr_auc", "mcc", "brier", "roc_auc", "ece"):
            vals = [m[split][key] for m in all_metrics if key in m.get(split, {})]
            summary[f"{split}_{key}_mean"] = float(np.mean(vals))
            summary[f"{split}_{key}_std"] = float(np.std(vals))

    write_json(cv_root / "cv_summary.json", summary)
    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_parquet(cv_root / "cv_predictions.parquet", index=False)

    label_col = data_cfg["label_col"]
    y_true = preds[label_col].to_numpy(dtype=int)
    y_prob = (
        preds["p_bbb_calibrated"].to_numpy(dtype=float)
        if "p_bbb_calibrated" in preds.columns
        else preds["p_bbb_raw"].to_numpy(dtype=float)
    )
    x, y = _reliability_curve(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", label="ideal")
    if len(x) > 0:
        plt.plot(x, y, marker="o", label="model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability ({exp_name}, {args.calibration})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cv_root / "reliability.png", dpi=180)
    print(f"Saved CV summary to {cv_root / 'cv_summary.json'}")


def run_download_data(args: argparse.Namespace) -> None:
    from bbb_classifier.dataset import cache_dir_from_config, sync_hf_dataset

    output = args.output or cache_dir_from_config(read_yaml(args.data_config))
    cache = sync_hf_dataset(output, repo_id=args.repo, force=args.force)
    print(f"Dataset ready at {cache}")
    print(f"  parquet: {cache / 'peptides.parquet'}")
    print(f"  structures: {cache / 'structures'}")


def run_sweep(args: argparse.Namespace) -> None:
    geo_types = {"struct_egnn_geo"}
    module = "bbb_geo" if args.mode == "geo" else "bbb_classifier"
    for exp in sorted(Path(args.experiments_dir).glob("*.yaml")):
        model_type = yaml.safe_load(exp.read_text(encoding="utf-8")).get("model_type", "")
        is_geo = model_type in geo_types
        if args.mode == "geo" and not is_geo:
            continue
        if args.mode == "classifier" and is_geo:
            continue
        subprocess.run(
            _package_cmd(
                module,
                "train",
                "--exp",
                str(exp),
                "--data-config",
                args.data_config,
                "--train-config",
                args.train_config,
                "--output-root",
                args.output_root,
            ),
            check=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bbb-classifier", description="BBB tabular/ESM classifier CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a classifier experiment")
    add_train_args(train_p)

    predict_p = sub.add_parser("predict", help="Score candidates with a trained run")
    predict_p.add_argument("--run-dir", required=True)
    predict_p.add_argument("--input", required=True)
    predict_p.add_argument("--output", required=True)
    predict_p.add_argument("--top-k", type=int, default=0)
    predict_p.add_argument("--threshold", type=float, default=0.5)

    eval_p = sub.add_parser("evaluate", help="Evaluate a trained run on labeled data")
    eval_p.add_argument("--run-dir", required=True)
    eval_p.add_argument("--dataset-path", required=True)

    cv_p = sub.add_parser("cv", help="Run 5-fold cross-validation")
    cv_p.add_argument("--exp", required=True)
    cv_p.add_argument("--data-config", default="configs/data.yaml")
    cv_p.add_argument("--train-config", default="configs/train.yaml")
    cv_p.add_argument("--dataset-path", default=None)
    cv_p.add_argument("--output-root", default="artifacts/cv")
    cv_p.add_argument("--calibration", choices=["isotonic", "platt", "none"], default="isotonic")

    dl_p = sub.add_parser("download-data", help="Download the HF bbb-peptides dataset")
    dl_p.add_argument("--output", type=Path, default=None)
    dl_p.add_argument("--repo", default="manumartinm/bbb-peptides")
    dl_p.add_argument("--data-config", default="configs/data.yaml")
    dl_p.add_argument("--force", action="store_true")

    sweep_p = sub.add_parser("sweep", help="Train all experiments in a directory")
    sweep_p.add_argument(
        "--mode",
        choices=("classifier", "geo"),
        default="classifier",
        help="Run tabular/ESM (classifier) or structural EGNN (geo) experiments.",
    )
    sweep_p.add_argument("--experiments-dir", default="configs/experiments")
    sweep_p.add_argument("--data-config", default="configs/data.yaml")
    sweep_p.add_argument("--train-config", default="configs/train.yaml")
    sweep_p.add_argument("--output-root", default="artifacts")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
        print(f"Predictions saved to {args.output}")
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "cv":
        run_cv(args)
    elif args.command == "download-data":
        run_download_data(args)
    elif args.command == "sweep":
        run_sweep(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


# Backward-compatible entry points for tests and wrappers
def main_train(argv: list[str] | None = None) -> None:
    main(["train", *(argv or sys.argv[1:])])


def main_predict(argv: list[str] | None = None) -> None:
    main(["predict", *(argv or sys.argv[1:])])


def main_evaluate(argv: list[str] | None = None) -> None:
    main(["evaluate", *(argv or sys.argv[1:])])


def main_cv(argv: list[str] | None = None) -> None:
    main(["cv", *(argv or sys.argv[1:])])


if __name__ == "__main__":
    main()
