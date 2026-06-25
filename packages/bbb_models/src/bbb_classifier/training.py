from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from bbb_classifier.io import ensure_dir
from bbb_geo.features import apply_coord_noise
from bbb_geo.models import sample_edm_sigma

# --- Metrics ---


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (y_prob >= left) & (y_prob < right if i < n_bins - 1 else y_prob <= right)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += (np.sum(mask) / max(n, 1)) * abs(acc - conf)
    return float(ece)


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.nan_to_num(np.asarray(y_prob).astype(float), nan=0.5, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=10),
    }


# --- Calibration ---


def sanitize_probabilities(p: np.ndarray, *, fill: float = 0.5) -> np.ndarray:
    out = np.nan_to_num(np.asarray(p, dtype=float).reshape(-1), nan=fill, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0)


class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic") -> None:
        self.method = method
        self.model = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> None:
        p = sanitize_probabilities(p)
        y = np.asarray(y).astype(int).reshape(-1)
        if self.method == "platt":
            lr = LogisticRegression(max_iter=200)
            lr.fit(p.reshape(-1, 1), y)
            self.model = lr
        else:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y)
            self.model = iso

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = sanitize_probabilities(p)
        if self.model is None:
            return p
        if self.method == "platt":
            return self.model.predict_proba(p.reshape(-1, 1))[:, 1]
        return self.model.predict(p)

    def save(self, path: str) -> None:
        joblib.dump({"method": self.method, "model": self.model}, path)

    @staticmethod
    def load(path: str) -> ProbabilityCalibrator:
        data = joblib.load(path)
        obj = ProbabilityCalibrator(method=data["method"])
        obj.model = data["model"]
        return obj


# --- Checkpoints ---


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    torch.save(state, p)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


# --- Early stopping ---


class EarlyStopper:
    def __init__(self, patience: int = 8, maximize: bool = True):
        self.patience = patience
        self.maximize = maximize
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return False
        improved = value > self.best if self.maximize else value < self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


# --- Losses & mixup ---


def bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if sample_weight is None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return criterion(logits, labels)
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
        reduction="none",
    )
    return (losses * sample_weight).mean()


def apply_mixup(
    batch: dict[str, Any],
    alpha: float = 0.2,
    prob: float = 0.5,
) -> dict[str, Any]:
    if alpha <= 0 or np.random.rand() > prob:
        return batch
    y = batch["y"]
    n = y.shape[0]
    if n < 2:
        return batch
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(n, device=y.device)

    out = dict(batch)
    out["y"] = lam * y + (1.0 - lam) * y[perm]
    for key in ("tab", "esm", "feat3d"):
        if key in out and out[key] is not None:
            out[key] = lam * out[key] + (1.0 - lam) * out[key][perm]
    return out


# --- Torch training engine ---


@dataclass
class TorchData:
    y: np.ndarray
    tab: np.ndarray | None
    esm: np.ndarray | None
    feat3d: np.ndarray | None
    sample_weight: np.ndarray | None = None
    graphs: list[dict[str, torch.Tensor]] | None = None
    struct_samples: list[dict[str, torch.Tensor | str | float]] | None = None


def _sample_to_device(
    sample: dict[str, torch.Tensor | str | float], device: torch.device
) -> dict[str, torch.Tensor | str | float]:
    out: dict[str, torch.Tensor | str | float] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _iter_batches(data: TorchData, batch_size: int, shuffle: bool) -> list[np.ndarray]:
    n = data.y.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, n, batch_size)]


def _select(data: TorchData, ids: np.ndarray, device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {
        "y": torch.tensor(data.y[ids], dtype=torch.float32, device=device).unsqueeze(1),
    }
    if data.sample_weight is not None:
        out["sample_weight"] = torch.tensor(
            data.sample_weight[ids], dtype=torch.float32, device=device
        ).unsqueeze(1)
    if data.tab is not None:
        out["tab"] = torch.tensor(data.tab[ids], dtype=torch.float32, device=device)
    if data.esm is not None:
        out["esm"] = torch.tensor(data.esm[ids], dtype=torch.float32, device=device)
    if data.feat3d is not None:
        out["feat3d"] = torch.tensor(data.feat3d[ids], dtype=torch.float32, device=device)
    if data.graphs is not None:
        out["graphs"] = [data.graphs[i] for i in ids]
    if data.struct_samples is not None:
        out["struct_samples"] = [
            _sample_to_device(data.struct_samples[i], device=device) for i in ids
        ]
    return out


def _model_inputs(batch: dict[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    model_name = model.__class__.__name__
    if model_name == "StructEGNNGeo":
        return {"graphs": batch.get("struct_samples")}
    return {k: v for k, v in batch.items() if k not in {"y", "sample_weight", "struct_samples"}}


def _apply_struct_noise(
    struct_samples: list[dict[str, torch.Tensor | str | float]] | None,
    sigma_values: torch.Tensor,
) -> list[dict[str, torch.Tensor | str | float]] | None:
    if struct_samples is None:
        return None
    out: list[dict[str, torch.Tensor | str | float]] = []
    for sample, sigma in zip(struct_samples, sigma_values, strict=False):
        noisy = dict(sample)
        coords = sample["coords"]
        noisy["coords"] = apply_coord_noise(coords, float(sigma.item()))
        noisy["sigma"] = float(sigma.item())
        out.append(noisy)
    return out


def _struct_multitask_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
    sample_weight: torch.Tensor | None,
    struct_cfg: dict[str, Any],
) -> torch.Tensor:
    struct_samples = batch.get("struct_samples")
    if struct_samples is None:
        raise ValueError("struct_samples required for structural model training")
    sigma_values = sample_edm_sigma(
        len(struct_samples),
        sigma_data=float(struct_cfg.get("sigma_data", 16.0)),
        sigma_min=float(struct_cfg.get("sigma_min", 0.0004)),
        sigma_max=float(struct_cfg.get("sigma_max", 160.0)),
        low_mid_bias=float(struct_cfg.get("low_mid_bias", 0.7)),
        coord_sigma_cap=struct_cfg.get("coord_sigma_cap", 16.0),
        device=labels.device,
    )
    noisy_samples = _apply_struct_noise(struct_samples, sigma_values)
    logits, aux_pred, aux_target = model.forward_with_aux(noisy_samples)  # type: ignore[attr-defined]
    cls_loss = bce_loss(logits, labels, pos_weight=pos_weight, sample_weight=sample_weight)
    aux_weight = float(struct_cfg.get("aux_weight", 0.2))
    aux_loss = torch.nn.functional.mse_loss(aux_pred, aux_target)
    return cls_loss + aux_weight * aux_loss


def predict_torch(
    model: torch.nn.Module, data: TorchData, batch_size: int, device: torch.device
) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for ids in _iter_batches(data, batch_size=batch_size, shuffle=False):
            batch = _select(data, ids, device)
            logits = model(**_model_inputs(batch, model))
            p = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            probs.append(p)
    return np.concatenate(probs, axis=0)


def train_torch_model(
    model: torch.nn.Module,
    train_data: TorchData,
    val_data: TorchData,
    run_dir: Path,
    train_cfg: dict[str, Any],
    tracking_cfg: dict[str, Any],
    primary_metric: str = "pr_auc",
    maximize_metric: bool = True,
    resume: bool = True,
) -> tuple[torch.nn.Module, dict[str, float], np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    start_epoch = 0
    best_metric = -np.inf if maximize_metric else np.inf

    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    tb_dir = ensure_dir(run_dir / "tensorboard")
    writer = SummaryWriter(str(tb_dir)) if tracking_cfg.get("tensorboard", True) else None
    stopper = EarlyStopper(patience=int(train_cfg.get("patience", 8)), maximize=maximize_metric)

    last_ckpt = ckpt_dir / "last.ckpt"
    if resume and last_ckpt.exists():
        state = load_checkpoint(last_ckpt)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_metric = float(state.get("best_metric", best_metric))

    use_mlflow = bool(tracking_cfg.get("mlflow", False))
    if use_mlflow:
        mlflow.start_run(run_name=run_dir.name)

    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])

    for epoch in range(start_epoch, epochs):
        model.train()
        batch_losses = []
        non_finite_batches = 0
        mix_cfg = train_cfg.get("mixup", {})
        mix_enabled = bool(mix_cfg.get("enabled", False))
        mix_alpha = float(mix_cfg.get("alpha", 0.2))
        mix_prob = float(mix_cfg.get("prob", 0.5))
        struct_cfg = train_cfg.get("struct", {})
        is_struct_geo = model.__class__.__name__ == "StructEGNNGeo"
        for ids in _iter_batches(train_data, batch_size=batch_size, shuffle=True):
            batch = _select(train_data, ids, device)
            if mix_enabled and not is_struct_geo:
                batch = apply_mixup(batch, alpha=mix_alpha, prob=mix_prob)
            labels = batch["y"]
            pos_ratio = labels.mean().detach()
            pos_weight = ((1 - pos_ratio) / torch.clamp(pos_ratio, min=1e-6)).reshape(1)
            if is_struct_geo:
                loss = _struct_multitask_loss(
                    model,
                    batch,
                    labels,
                    pos_weight,
                    batch.get("sample_weight"),
                    struct_cfg,
                )
            else:
                logits = model(**_model_inputs(batch, model))
                loss = bce_loss(
                    logits,
                    labels,
                    pos_weight=pos_weight,
                    sample_weight=batch.get("sample_weight"),
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss):
                non_finite_batches += 1
                continue
            if float(train_cfg.get("grad_clip", 0.0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            optimizer.step()
            batch_losses.append(loss.detach().cpu().item())

        val_prob = predict_torch(model, val_data, batch_size=batch_size, device=device)
        val_metrics = classification_metrics(val_data.y, val_prob)
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        if non_finite_batches > 0:
            print(f"[warn] epoch={epoch}: skipped {non_finite_batches} non-finite batches")
        metric_value = float(val_metrics.get(primary_metric, np.nan))

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        if use_mlflow:
            mlflow.log_metrics({"train_loss": train_loss, **val_metrics}, step=epoch)

        save_checkpoint(
            last_ckpt,
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_metric": best_metric,
            },
        )

        improved = metric_value > best_metric if maximize_metric else metric_value < best_metric
        if improved:
            best_metric = metric_value
            save_checkpoint(
                ckpt_dir / "best.ckpt",
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_metric": best_metric,
                },
            )

        periodic_every = int(train_cfg.get("save_periodic_every", 0))
        if periodic_every > 0 and (epoch + 1) % periodic_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch + 1}.ckpt",
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_metric": best_metric,
                },
            )

        if stopper.step(metric_value):
            break

    if use_mlflow:
        mlflow.end_run()
    if writer is not None:
        writer.close()

    best = load_checkpoint(ckpt_dir / "best.ckpt")
    model.load_state_dict(best["model"])
    final_prob = predict_torch(model, val_data, batch_size=batch_size, device=device)
    final_prob = sanitize_probabilities(final_prob)
    final_metrics = classification_metrics(val_data.y, final_prob)
    return model, final_metrics, final_prob
