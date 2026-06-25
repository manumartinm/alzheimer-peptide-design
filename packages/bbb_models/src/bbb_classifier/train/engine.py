from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from bbb_geo.features.struct_graph import apply_coord_noise
from bbb_geo.models.struct_egnn import sample_edm_sigma
from bbb_classifier.train.calibration import sanitize_probabilities
from bbb_classifier.train.checkpoints import load_checkpoint, save_checkpoint
from bbb_classifier.train.early_stop import EarlyStopper
from bbb_classifier.train.losses import bce_loss
from bbb_classifier.train.metrics import classification_metrics
from bbb_classifier.train.mixup import apply_mixup
from bbb_classifier.utils.io import ensure_dir


@dataclass
class TorchData:
    y: np.ndarray
    tab: np.ndarray | None
    esm: np.ndarray | None
    feat3d: np.ndarray | None
    sample_weight: np.ndarray | None = None
    graphs: list[dict[str, torch.Tensor]] | None = None
    struct_samples: list[dict[str, torch.Tensor | str | float]] | None = None


def _sample_to_device(sample: dict[str, torch.Tensor | str | float], device: torch.device) -> dict[str, torch.Tensor | str | float]:
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
        out["sample_weight"] = torch.tensor(data.sample_weight[ids], dtype=torch.float32, device=device).unsqueeze(1)
    if data.tab is not None:
        out["tab"] = torch.tensor(data.tab[ids], dtype=torch.float32, device=device)
    if data.esm is not None:
        out["esm"] = torch.tensor(data.esm[ids], dtype=torch.float32, device=device)
    if data.feat3d is not None:
        out["feat3d"] = torch.tensor(data.feat3d[ids], dtype=torch.float32, device=device)
    if data.graphs is not None:
        out["graphs"] = [data.graphs[i] for i in ids]
    if data.struct_samples is not None:
        out["struct_samples"] = [_sample_to_device(data.struct_samples[i], device=device) for i in ids]
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
    for sample, sigma in zip(struct_samples, sigma_values):
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


def predict_torch(model: torch.nn.Module, data: TorchData, batch_size: int, device: torch.device) -> np.ndarray:
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
                ckpt_dir / f"epoch_{epoch+1}.ckpt",
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
