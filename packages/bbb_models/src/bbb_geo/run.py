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
import torch
import yaml

from bbb_classifier.dataset import train_val_split
from bbb_classifier.features import rank_candidates
from bbb_classifier.io import ensure_dir, get_logger, read_yaml, write_json
from bbb_classifier.run import add_train_args
from bbb_classifier.training import (
    ProbabilityCalibrator,
    TorchData,
    classification_metrics,
    load_checkpoint,
    predict_torch,
)
from bbb_geo.config import GeoRunContext
from bbb_geo.enums import GEO_MODEL_TYPES
from bbb_geo.features import (
    GeoFeatureBuilder,
    build_struct_batch,
    build_struct_sample,
    load_struct_manifest,
    merge_dataset_with_manifest,
)
from bbb_geo.models import StructEGNNGeo
from bbb_geo.training import evaluate_guidance_gate, train_geo_model


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


def _package_cmd(module: str, command: str) -> list[str]:
    return [sys.executable, "-m", module, command]


class GeoRun:
    def __init__(
        self, ctx: GeoRunContext, *, dataset_path: str | None = None, no_resume: bool = False
    ):
        self.ctx = ctx
        self.dataset_path = dataset_path
        self.no_resume = no_resume

    @classmethod
    def from_train_args(cls, args: argparse.Namespace) -> GeoRun:
        return cls(
            GeoRunContext.from_train_args(args),
            dataset_path=args.dataset_path,
            no_resume=args.no_resume,
        )

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> GeoRun:
        from bbb_classifier.config import TrainConfig

        exp, data, _ = GeoRunContext.from_run_dir(run_dir)
        ctx = GeoRunContext(
            exp=exp,
            data=data,
            train=TrainConfig.from_dict({}),
            run_dir=run_dir,
            logger=get_logger("bbb-geo"),
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
        builder = GeoFeatureBuilder(self.ctx.data, self.ctx.exp)
        tr_bundle = builder.build(train_df)
        va_bundle = builder.build(val_df)
        train_df = GeoFeatureBuilder.apply_plddt_weights(tr_bundle.struct_df, self.ctx.exp)
        val_df = va_bundle.struct_df
        self.ctx.logger.info(
            "Dataset loaded: %d train / %d val (with structures)", len(train_df), len(val_df)
        )

        y_train = train_df[self.ctx.data.label_col].to_numpy(dtype=int)
        y_val = val_df[self.ctx.data.label_col].to_numpy(dtype=int)
        weights = self._sample_weights(train_df)

        if self.ctx.train.tracking.get("mlflow", False):
            mlflow.set_experiment(self.ctx.train.tracking.get("mlflow_experiment", "bbb_geo"))

        tr_feat = {
            "struct_samples": tr_bundle.struct_samples,
            "tab_cols": tr_bundle.tab_cols,
            "tab": None,
            "esm": None,
            "feat3d": None,
        }
        va_feat = {
            "struct_samples": va_bundle.struct_samples,
            "tab_cols": va_bundle.tab_cols,
            "tab": None,
            "esm": None,
            "feat3d": None,
        }
        val_prob = train_geo_model(
            self.ctx.exp.model_type.value,
            tr_feat,
            va_feat,
            y_train,
            y_val,
            weights,
            self.ctx.run_dir,
            self.ctx.train.to_dict(),
            self.ctx.exp.to_dict(),
            resume=not self.no_resume,
        )
        self._finalize(val_df, y_val, val_prob, tr_bundle.tab_cols)

    def _build_struct_samples(
        self,
        df: pd.DataFrame,
        *,
        manifest: str | None = None,
    ) -> tuple[pd.DataFrame, list[dict]]:
        exp = self.ctx.exp
        data = self.ctx.data
        if exp.model_type not in GEO_MODEL_TYPES:
            raise ValueError(
                f"{exp.model_type.value} is not a geo model; use bbb-classifier predict"
            )

        struct_cfg = exp.struct
        manifest_path = manifest or struct_cfg.get("manifest_path") or data.struct_manifest_path
        work_df = df
        if manifest_path and "coords_path" not in df.columns:
            work_df = merge_dataset_with_manifest(
                df, load_struct_manifest(manifest_path), sequence_col=data.sequence_col
            )

        struct_samples: list[dict] = []
        valid_rows: list[pd.Series] = []
        for _, row in work_df.iterrows():
            if pd.isna(row.get("coords_path")):
                continue
            valid_rows.append(row)
            struct_samples.append(
                build_struct_sample(
                    row["coords_path"],
                    str(row[data.sequence_col]),
                    radius=float(struct_cfg.get("radius", 10.0)),
                    num_rbf=int(struct_cfg.get("num_rbf", 16)),
                )
            )
        if not struct_samples:
            raise ValueError("No structural samples found for prediction.")
        return pd.DataFrame(valid_rows).reset_index(drop=True), struct_samples

    def _predict_probs(
        self, struct_samples: list[dict], labels: np.ndarray | None = None
    ) -> np.ndarray:
        exp = self.ctx.exp
        model_cfg = exp.model
        model = StructEGNNGeo(
            hidden_dim=int(model_cfg.get("egnn_hidden", 64)),
            num_layers=int(model_cfg.get("egnn_layers", 3)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
        model.load_state_dict(
            load_checkpoint(self.ctx.run_dir / "checkpoints" / "best.ckpt")["model"]
        )
        y = labels if labels is not None else np.zeros(len(struct_samples))
        td = TorchData(y=y, tab=None, esm=None, feat3d=None, struct_samples=struct_samples)
        return predict_torch(
            model,
            td,
            batch_size=128,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
        manifest: str | None = None,
        threshold: float = 0.5,
        top_k: int = 0,
    ) -> pd.DataFrame:
        aligned_df, struct_samples = self._build_struct_samples(df, manifest=manifest)
        label_col = self.ctx.data.label_col
        labels = (
            aligned_df[label_col].to_numpy(dtype=int)
            if label_col in aligned_df.columns
            else np.zeros(len(struct_samples))
        )
        prob = self._predict_probs(struct_samples, labels)
        p_cal = self._calibrate(prob)
        out_df = aligned_df.copy()
        out_df["p_bbb_raw"] = prob
        out_df["p_bbb_calibrated"] = p_cal
        out_df["decision"] = (out_df["p_bbb_calibrated"] >= threshold).astype(int)
        if top_k > 0:
            out_df = rank_candidates(out_df, prob_col="p_bbb_calibrated", top_k=top_k)
        return out_df

    def evaluate(
        self, df: pd.DataFrame, *, manifest: str | None = None
    ) -> dict[str, dict[str, float]]:
        aligned_df, struct_samples = self._build_struct_samples(df, manifest=manifest)
        y = aligned_df[self.ctx.data.label_col].to_numpy(dtype=int)
        prob = self._predict_probs(struct_samples, y)
        p_cal = self._calibrate(prob)
        metrics = {
            "raw": classification_metrics(y, prob),
            "calibrated": classification_metrics(y, p_cal),
        }
        write_json(self.ctx.run_dir / "evaluation_metrics.json", metrics)
        out_df = aligned_df[[self.ctx.data.sequence_col, self.ctx.data.label_col]].copy()
        out_df["p_bbb_raw"] = prob
        out_df["p_bbb_calibrated"] = p_cal
        out_df.to_parquet(self.ctx.run_dir / "evaluation_predictions.parquet", index=False)
        return metrics

    def probe(
        self,
        *,
        manifest: str,
        dataset_path: str | Path,
        max_samples: int = 50,
    ) -> dict:
        df = pd.read_parquet(dataset_path)
        merged = merge_dataset_with_manifest(
            df, load_struct_manifest(manifest), sequence_col=self.ctx.data.sequence_col
        ).head(max_samples)
        _, struct_samples = build_struct_batch(merged, self.ctx.data.sequence_col)

        model_cfg = self.ctx.exp.model
        model = StructEGNNGeo(
            hidden_dim=int(model_cfg.get("egnn_hidden", 64)),
            num_layers=int(model_cfg.get("egnn_layers", 3)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
        model.load_state_dict(
            load_checkpoint(self.ctx.run_dir / "checkpoints" / "best.ckpt")["model"]
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        validation = self.ctx.exp.validation
        report = evaluate_guidance_gate(
            model=model,
            struct_samples=struct_samples,
            max_samples=max_samples,
            grad_norm_threshold=float(validation.get("gate_grad_norm_threshold", 1e-3)),
            corr_threshold=float(validation.get("gate_corr_threshold", 0.1)),
            device=device,
        )
        return report


def run_train(args: argparse.Namespace) -> None:
    GeoRun.from_train_args(args).train()


def run_predict(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    exp, data, _ = GeoRunContext.from_run_dir(run_dir)
    from bbb_classifier.config import TrainConfig

    ctx = GeoRunContext(
        exp=exp,
        data=data,
        train=TrainConfig.from_dict({}),
        run_dir=run_dir,
        logger=get_logger("bbb-geo"),
    )
    df = _read_table(Path(args.input))
    out_df = GeoRun(ctx).predict(
        df, manifest=args.manifest, threshold=args.threshold, top_k=args.top_k
    )
    _write_table(out_df, Path(args.output))


def run_evaluate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    exp, data, _ = GeoRunContext.from_run_dir(run_dir)
    from bbb_classifier.config import TrainConfig

    ctx = GeoRunContext(
        exp=exp,
        data=data,
        train=TrainConfig.from_dict({}),
        run_dir=run_dir,
        logger=get_logger("bbb-geo"),
    )
    df = pd.read_parquet(args.dataset_path)
    metrics = GeoRun(ctx).evaluate(df, manifest=getattr(args, "manifest", None))
    print(metrics)


def run_probe(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    exp, data, _ = GeoRunContext.from_run_dir(run_dir)
    from bbb_classifier.config import TrainConfig

    ctx = GeoRunContext(
        exp=exp,
        data=data,
        train=TrainConfig.from_dict({}),
        run_dir=run_dir,
        logger=get_logger("bbb-geo"),
    )
    report = GeoRun(ctx).probe(
        manifest=args.manifest,
        dataset_path=args.dataset,
        max_samples=args.max_samples,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_cv(args: argparse.Namespace) -> None:
    from bbb_classifier.dataset import load_data_config

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
    train_cmd = _package_cmd("bbb_geo", "train")
    all_metrics, all_preds = [], []

    with tempfile.TemporaryDirectory(prefix="bbb_geo_cv_") as tmp:
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
                    *train_cmd,
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
            sigma_path = run_dir / "metrics_multisigma.json"
            if sigma_path.exists():
                sigma_metrics = json.loads(sigma_path.read_text(encoding="utf-8"))
                metrics["multisigma"] = sigma_metrics
                metrics["low_sigma_mean_pr_auc"] = sigma_metrics.get("summary", {}).get(
                    "low_sigma_mean_pr_auc"
                )
            gate_path = run_dir / "guidance_gate.json"
            if gate_path.exists():
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
                metrics["guidance_gate"] = gate
                metrics["guidance_gate_pass"] = bool(gate.get("gate_pass", False))
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
    low_sigma_vals = [
        m.get("low_sigma_mean_pr_auc")
        for m in all_metrics
        if m.get("low_sigma_mean_pr_auc") is not None
    ]
    if low_sigma_vals:
        summary["low_sigma_mean_pr_auc_mean"] = float(np.mean(low_sigma_vals))
        summary["low_sigma_mean_pr_auc_std"] = float(np.std(low_sigma_vals))
    gate_vals = [
        float(bool(m.get("guidance_gate_pass"))) for m in all_metrics if "guidance_gate_pass" in m
    ]
    if gate_vals:
        summary["guidance_gate_pass_rate"] = float(np.mean(gate_vals))

    write_json(cv_root / "cv_summary.json", summary)
    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_parquet(cv_root / "cv_predictions.parquet", index=False)

    y_true = preds[data_cfg["label_col"]].to_numpy(dtype=int)
    y_prob = (
        preds["p_bbb_calibrated"].to_numpy(dtype=float)
        if "p_bbb_calibrated" in preds.columns
        else preds["p_bbb_raw"].to_numpy(dtype=float)
    )
    x, y = _reliability_curve(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", label="ideal")
    if len(x) > 0:
        plt.plot(x, y, marker="o", label="geo")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability ({exp_name}, {args.calibration})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cv_root / "reliability.png", dpi=180)
    print(f"Saved CV summary to {cv_root / 'cv_summary.json'}")


def _parse_list(raw: str, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def run_sweep_stability(args: argparse.Namespace) -> None:
    exp_cfg = read_yaml(args.exp)
    exp_name = exp_cfg.get("name", "exp09_struct_egnn_noise")
    cv_cmd = _package_cmd("bbb_geo", "cv")
    output_dir = ensure_dir(Path(args.output_root) / exp_name)

    caps = _parse_list(args.coord_sigma_caps, float)
    aux_weights = _parse_list(args.aux_weights, float)
    rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="bbb_geo_sweep_") as tmp:
        tmp_dir = Path(tmp)
        for cap in caps:
            for aux in aux_weights:
                variant = dict(exp_cfg)
                variant["name"] = f"{exp_name}_cap{cap:g}_aux{aux:g}"
                struct_cfg = dict(variant.get("struct", {}))
                struct_cfg["coord_sigma_cap"] = float(cap)
                struct_cfg["aux_weight"] = float(aux)
                variant["struct"] = struct_cfg

                exp_path = tmp_dir / f"{variant['name']}.yaml"
                exp_path.write_text(yaml.safe_dump(variant), encoding="utf-8")

                cmd = [
                    *cv_cmd,
                    "--exp",
                    str(exp_path),
                    "--data-config",
                    args.data_config,
                    "--train-config",
                    args.train_config,
                    "--output-root",
                    str(args.cv_output_root),
                    "--calibration",
                    args.calibration,
                ]
                if args.dataset_path:
                    cmd.extend(["--dataset-path", args.dataset_path])
                subprocess.run(cmd, check=True)

                cv_summary_path = (
                    Path(args.cv_output_root)
                    / variant["name"]
                    / args.calibration
                    / "cv_summary.json"
                )
                summary = json.loads(cv_summary_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "experiment": str(variant["name"]),
                        "coord_sigma_cap": float(cap),
                        "aux_weight": float(aux),
                        "raw_pr_auc_mean": summary.get("raw_pr_auc_mean", np.nan),
                        "calibrated_pr_auc_mean": summary.get("calibrated_pr_auc_mean", np.nan),
                        "low_sigma_mean_pr_auc_mean": summary.get(
                            "low_sigma_mean_pr_auc_mean", np.nan
                        ),
                        "guidance_gate_pass_rate": summary.get("guidance_gate_pass_rate", np.nan),
                    }
                )

    if not rows:
        raise RuntimeError("No sweep rows were generated")

    df = pd.DataFrame(rows)
    score_col = args.selection_metric
    if score_col not in df.columns:
        raise ValueError(f"selection_metric '{score_col}' not found in sweep results")
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df.to_parquet(output_dir / "stability_sweep.parquet", index=False)
    write_json(
        output_dir / "stability_sweep_best.json",
        {"selection_metric": score_col, "best": df.iloc[0].to_dict()},
    )
    print(f"Saved sweep leaderboard to {output_dir / 'stability_sweep.parquet'}")
    print(f"Best config: {df.iloc[0].to_dict()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bbb-geo", description="BBB geometry EGNN CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a geometry EGNN experiment")
    add_train_args(train_p)

    predict_p = sub.add_parser("predict", help="Score candidates with a trained geo run")
    predict_p.add_argument("--run-dir", required=True)
    predict_p.add_argument("--input", required=True)
    predict_p.add_argument("--output", required=True)
    predict_p.add_argument("--manifest", default=None)
    predict_p.add_argument("--top-k", type=int, default=0)
    predict_p.add_argument("--threshold", type=float, default=0.5)

    eval_p = sub.add_parser("evaluate", help="Evaluate a trained geo run on labeled data")
    eval_p.add_argument("--run-dir", required=True)
    eval_p.add_argument("--dataset-path", required=True)
    eval_p.add_argument("--manifest", default=None)

    probe_p = sub.add_parser("probe", help="Geometry-sensitivity gate for diffusion guidance")
    probe_p.add_argument("--run-dir", required=True)
    probe_p.add_argument("--manifest", required=True)
    probe_p.add_argument("--dataset", required=True)
    probe_p.add_argument("--output", default="artifacts/geometry_sensitivity.json")
    probe_p.add_argument("--max-samples", type=int, default=50)

    cv_p = sub.add_parser("cv", help="Run 5-fold cross-validation")
    cv_p.add_argument("--exp", required=True)
    cv_p.add_argument("--data-config", default="configs/data.yaml")
    cv_p.add_argument("--train-config", default="configs/train.yaml")
    cv_p.add_argument("--dataset-path", default=None)
    cv_p.add_argument("--output-root", default="artifacts/cv")
    cv_p.add_argument("--calibration", choices=["isotonic", "platt", "none"], default="isotonic")

    sweep_p = sub.add_parser(
        "sweep-stability", help="Sweep exp09 stability hyperparameters with CV"
    )
    sweep_p.add_argument("--exp", default="configs/experiments/exp09_struct_egnn_noise.yaml")
    sweep_p.add_argument("--data-config", default="configs/data.yaml")
    sweep_p.add_argument("--train-config", default="configs/train_cv.yaml")
    sweep_p.add_argument("--dataset-path", default=None)
    sweep_p.add_argument("--coord-sigma-caps", default="8,12,16")
    sweep_p.add_argument("--aux-weights", default="0.1,0.2,0.3")
    sweep_p.add_argument("--calibration", choices=["isotonic", "platt", "none"], default="isotonic")
    sweep_p.add_argument("--selection-metric", default="low_sigma_mean_pr_auc_mean")
    sweep_p.add_argument("--cv-output-root", default="artifacts/cv")
    sweep_p.add_argument("--output-root", default="artifacts/sweeps")

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
    elif args.command == "probe":
        run_probe(args)
    elif args.command == "cv":
        run_cv(args)
    elif args.command == "sweep-stability":
        run_sweep_stability(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
