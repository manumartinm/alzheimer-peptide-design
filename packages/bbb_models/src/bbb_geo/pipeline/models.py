from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from bbb_classifier.train.metrics import classification_metrics
from bbb_classifier.utils.io import write_json
from bbb_geo.features.membrane_potential import amphipathicity_score, per_residue_hydrophobicity
from bbb_geo.features.struct_graph import apply_coord_noise
from bbb_classifier.train.engine import TorchData, predict_torch, train_torch_model
from bbb_geo.models import StructEGNNGeo


def fit_and_predict(
    model_type: str,
    tr_feat: dict[str, Any],
    va_feat: dict[str, Any],
    y_train: np.ndarray,
    y_val: np.ndarray,
    sample_weight: np.ndarray,
    run_dir: Path,
    train_cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
    *,
    resume: bool,
) -> np.ndarray:
    model_cfg = exp_cfg.get("model", {})
    if model_type != "struct_egnn_geo":
        raise ValueError(model_type)
    model = StructEGNNGeo(
        hidden_dim=int(model_cfg.get("egnn_hidden", 64)),
        num_layers=int(model_cfg.get("egnn_layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        chem_dropout=float(model_cfg.get("chem_dropout", 0.2)),
        sigma_data=float(model_cfg.get("sigma_data", 16.0)),
    )

    torch_train_cfg = dict(train_cfg.get("training", {}))
    torch_train_cfg["save_periodic_every"] = int(train_cfg.get("output", {}).get("save_periodic_every", 0))
    torch_train_cfg["mixup"] = exp_cfg.get("mixup", {})
    torch_train_cfg["struct"] = exp_cfg.get("struct", {})
    tr_data = TorchData(
        y=y_train,
        tab=tr_feat.get("tab"),
        esm=tr_feat.get("esm"),
        feat3d=tr_feat.get("feat3d"),
        sample_weight=sample_weight,
        struct_samples=tr_feat["struct_samples"],
    )
    va_data = TorchData(
        y=y_val,
        tab=va_feat.get("tab"),
        esm=va_feat.get("esm"),
        feat3d=va_feat.get("feat3d"),
        struct_samples=va_feat["struct_samples"],
    )
    model, _, val_prob = train_torch_model(
        model=model,
        train_data=tr_data,
        val_data=va_data,
        run_dir=run_dir,
        train_cfg=torch_train_cfg,
        tracking_cfg=train_cfg.get("tracking", {}),
        primary_metric=train_cfg.get("primary_metric", "pr_auc"),
        maximize_metric=bool(train_cfg.get("maximize_metric", True)),
        resume=resume,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_prob = predict_torch(model, va_data, batch_size=int(torch_train_cfg.get("batch_size", 128)), device=device)
    sigma_metrics = evaluate_multi_sigma(
        model=model,
        val_data=va_data,
        sigma_values=exp_cfg.get("validation", {}).get("sigma_values", [0.0, 2.0, 4.0, 8.0]),
        batch_size=int(torch_train_cfg.get("batch_size", 128)),
        device=device,
    )
    write_json(run_dir / "metrics_multisigma.json", sigma_metrics)
    guidance_gate = evaluate_guidance_gate(
        model=model,
        struct_samples=va_feat["struct_samples"],
        max_samples=int(exp_cfg.get("validation", {}).get("gate_max_samples", 50)),
        grad_norm_threshold=float(exp_cfg.get("validation", {}).get("gate_grad_norm_threshold", 1e-3)),
        corr_threshold=float(exp_cfg.get("validation", {}).get("gate_corr_threshold", 0.1)),
        device=device,
    )
    write_json(run_dir / "guidance_gate.json", guidance_gate)
    return val_prob


def _with_sigma(
    struct_samples: list[dict[str, torch.Tensor | str | float]],
    sigma: float,
) -> list[dict[str, torch.Tensor | str | float]]:
    out: list[dict[str, torch.Tensor | str | float]] = []
    for sample in struct_samples:
        s = dict(sample)
        coords = sample["coords"]
        if sigma > 0:
            coords = apply_coord_noise(coords, sigma=float(sigma))
        s["coords"] = coords
        s["sigma"] = float(sigma)
        out.append(s)
    return out


def evaluate_multi_sigma(
    *,
    model: torch.nn.Module,
    val_data: TorchData,
    sigma_values: list[float] | tuple[float, ...],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    report: dict[str, Any] = {"per_sigma": {}, "summary": {}}
    if val_data.struct_samples is None:
        return report

    scores: list[float] = []
    for sigma in [float(s) for s in sigma_values]:
        noisy_structs = _with_sigma(val_data.struct_samples, sigma=sigma)
        noisy_data = TorchData(
            y=val_data.y,
            tab=None,
            esm=None,
            feat3d=None,
            sample_weight=val_data.sample_weight,
            struct_samples=noisy_structs,
        )
        probs = predict_torch(model, noisy_data, batch_size=batch_size, device=device)
        metrics = classification_metrics(val_data.y, probs)
        report["per_sigma"][f"{sigma:g}"] = metrics
        if sigma <= 4.0 and np.isfinite(metrics.get("pr_auc", np.nan)):
            scores.append(float(metrics["pr_auc"]))
    report["summary"]["low_sigma_mean_pr_auc"] = float(np.mean(scores)) if scores else float("nan")
    report["summary"]["sigma_values"] = [float(s) for s in sigma_values]
    return report


def evaluate_guidance_gate(
    *,
    model: StructEGNNGeo,
    struct_samples: list[dict[str, torch.Tensor | str | float]],
    max_samples: int,
    grad_norm_threshold: float,
    corr_threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    samples = struct_samples[:max_samples]
    grad_norms: list[float] = []
    amp_scores: list[float] = []
    probs: list[float] = []
    for sample in samples:
        base = _sample_to_device(sample, device=device)
        grad_norms.append(_grad_norm(model, base))
        coords = base["coords"]
        hydro = per_residue_hydrophobicity(str(base["sequence"]), device=coords.device, dtype=coords.dtype)
        amp_scores.append(float(amphipathicity_score(coords, hydro).item()))
        with torch.no_grad():
            probs.append(float(torch.sigmoid(model.forward(graphs=[base])).item()))

    corr = float(np.corrcoef(probs, amp_scores)[0, 1]) if len(probs) > 1 else float("nan")
    mean_grad = float(np.mean(grad_norms)) if grad_norms else 0.0
    gate_pass = mean_grad > grad_norm_threshold and (np.isnan(corr) or abs(corr) > corr_threshold)
    return {
        "mean_grad_norm": mean_grad,
        "median_grad_norm": float(np.median(grad_norms)) if grad_norms else 0.0,
        "prob_amp_correlation": corr,
        "n_samples": len(samples),
        "thresholds": {
            "grad_norm_threshold": float(grad_norm_threshold),
            "corr_threshold": float(corr_threshold),
        },
        "gate_pass": bool(gate_pass),
        "recommendation": "hybrid" if gate_pass else "physics_only",
    }


def _sample_to_device(
    sample: dict[str, torch.Tensor | str | float],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor | str | float]:
    out: dict[str, torch.Tensor | str | float] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _grad_norm(model: StructEGNNGeo, sample: dict[str, torch.Tensor | str | float]) -> float:
    coords = sample["coords"].detach().clone().requires_grad_(True)
    test_sample = dict(sample)
    test_sample["coords"] = coords
    logp = model.log_prob([test_sample]).sum()
    logp.backward()
    if coords.grad is None:
        return 0.0
    return float(torch.linalg.norm(coords.grad).item())
