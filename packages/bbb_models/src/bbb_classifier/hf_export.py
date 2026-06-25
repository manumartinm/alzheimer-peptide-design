from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ModelKind = Literal["classifier", "geo"]

CLASSIFIER_RUN_CANDIDATES = (
    "exp03_esm_tab_mlp",
    "exp06_esm_tab_mlp_aug",
)
GEO_RUN_NAME = "exp09_struct_egnn_noise"

INFERENCE_FILES = (
    "checkpoints/best.ckpt",
    "checkpoints/best.pkl",
    "calibrators/calibrator.pkl",
    "train_metadata.json",
    "metrics.json",
)
GEO_EXTRA_FILES = (
    "metrics_multisigma.json",
    "guidance_gate.json",
)


@dataclass
class HFModelExportConfig:
    run_dir: Path
    output_dir: Path
    kind: ModelKind
    repo_id: str | None = None


def resolve_classifier_run_dir(base_dir: Path, run_name: str | None = None) -> Path:
    models_root = base_dir / "artifacts" / "models"
    if run_name:
        run_dir = models_root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Classifier run not found: {run_dir}")
        return run_dir
    for candidate in CLASSIFIER_RUN_CANDIDATES:
        run_dir = models_root / candidate
        if run_dir.exists():
            return run_dir
    raise FileNotFoundError(
        "No classifier run found under artifacts/models/. "
        f"Tried: {', '.join(CLASSIFIER_RUN_CANDIDATES)}"
    )


def resolve_geo_run_dir(base_dir: Path, run_name: str | None = None) -> Path:
    run_dir = base_dir / "artifacts" / "models" / (run_name or GEO_RUN_NAME)
    if not run_dir.exists():
        raise FileNotFoundError(f"Geo run not found: {run_dir}")
    return run_dir


def _copy_inference_files(run_dir: Path, output_dir: Path, *, kind: ModelKind) -> list[str]:
    copied: list[str] = []
    for rel in INFERENCE_FILES:
        src = run_dir / rel
        if not src.exists():
            continue
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    if kind == "geo":
        for rel in GEO_EXTRA_FILES:
            src = run_dir / rel
            if not src.exists():
                continue
            dst = output_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    checkpoint = output_dir / "checkpoints" / "best.ckpt"
    pkl = output_dir / "checkpoints" / "best.pkl"
    if not checkpoint.exists() and not pkl.exists():
        raise FileNotFoundError(f"No checkpoint found in exported bundle from {run_dir}")
    return copied


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_classifier_readme(run_dir: Path, repo_id: str | None) -> str:
    meta = _load_json(run_dir / "train_metadata.json")
    metrics = _load_json(run_dir / "metrics.json")
    exp_cfg = meta["exp_cfg"]
    run_name = exp_cfg.get("name", exp_cfg["model_type"])
    cal = metrics.get("calibrated", {})
    repo_line = f"\nbase_model: {repo_id}\n" if repo_id else ""
    return f"""---
license: mit
tags:
  - peptide
  - blood-brain-barrier
  - bbb
  - classification
  - esm
library_name: bbb-models
pipeline_tag: text-classification
{repo_line}---

# BBB Classifier — `{run_name}`

Sequence/tabular BBB permeability classifier for peptide candidates. Fuses ESM-2 embeddings with physicochemical descriptors and isotonic-calibrated probabilities.

## Model summary

| Field | Value |
|-------|-------|
| Experiment | `{run_name}` |
| Architecture | `{exp_cfg['model_type']}` |
| ESM dim | `{exp_cfg.get('model', {}).get('esm_dim', 'n/a')}` |
| Hidden dim | `{exp_cfg.get('model', {}).get('hidden_dim', 'n/a')}` |
| PR-AUC (calibrated) | `{cal.get('pr_auc', 'n/a')}` |
| ROC-AUC (calibrated) | `{cal.get('roc_auc', 'n/a')}` |
| MCC (calibrated) | `{cal.get('mcc', 'n/a')}` |

## Files

- `checkpoints/best.ckpt` — PyTorch weights
- `calibrators/calibrator.pkl` — isotonic calibration
- `train_metadata.json` — experiment + data config for inference
- `metrics.json` — validation metrics

## Usage

Install the project and download this repo:

```bash
git clone https://github.com/your-org/TFG.git
cd TFG/bbb_models && uv sync
hf download {"REPO_ID" if not repo_id else repo_id} --local-dir ./bbb-classifier
```

Score candidates (input must include sequence + tabular descriptor columns used at training time):

```bash
uv run python scripts/classifier/predict.py \\
  --run-dir ./bbb-classifier \\
  --input candidates.parquet \\
  --output scored.parquet
```

## Dataset

Trained on the TFG BBB peptide dataset. Companion dataset: [`manumartinm/bbb-peptides`](https://huggingface.co/datasets/manumartinm/bbb-peptides).

## Citation

If you use this model, cite the TFG BBB peptide modeling pipeline.
"""


def _render_geo_readme(run_dir: Path, repo_id: str | None) -> str:
    meta = _load_json(run_dir / "train_metadata.json")
    metrics = _load_json(run_dir / "metrics.json")
    gate = _load_json(run_dir / "guidance_gate.json") if (run_dir / "guidance_gate.json").exists() else {}
    exp_cfg = meta["exp_cfg"]
    run_name = exp_cfg.get("name", exp_cfg["model_type"])
    cal = metrics.get("calibrated", {})
    repo_line = f"\nbase_model: {repo_id}\n" if repo_id else ""
    return f"""---
license: mit
tags:
  - peptide
  - blood-brain-barrier
  - bbb
  - egnn
  - diffusion-guidance
  - protein-design
library_name: bbb-models
pipeline_tag: graph-ml
{repo_line}---

# BBB Geo EGNN — `{run_name}`

Geometry-only EGNN that predicts BBB permeability from 3D coordinates (`p_geo`). Used as a differentiable guidance signal in BoltzGen diffusion.

## Model summary

| Field | Value |
|-------|-------|
| Experiment | `{run_name}` |
| Architecture | `{exp_cfg['model_type']}` |
| EGNN hidden | `{exp_cfg.get('model', {}).get('egnn_hidden', 'n/a')}` |
| EGNN layers | `{exp_cfg.get('model', {}).get('egnn_layers', 'n/a')}` |
| PR-AUC (calibrated) | `{cal.get('pr_auc', 'n/a')}` |
| Guidance gate | `{gate.get('recommendation', 'n/a')}` (`gate_pass={gate.get('gate_pass', 'n/a')}`) |

## Files

- `checkpoints/best.ckpt` — PyTorch weights
- `calibrators/calibrator.pkl` — isotonic calibration
- `train_metadata.json` — experiment + struct config
- `metrics.json`, `metrics_multisigma.json`, `guidance_gate.json`

## Usage

```bash
cd TFG/bbb_models && uv sync
hf download {"REPO_ID" if not repo_id else repo_id} --local-dir ./bbb-geo
```

Structural prediction (requires coords manifest or `coords_path` column):

```bash
uv run python scripts/geo/predict.py \\
  --run-dir ./bbb-geo \\
  --input candidates.parquet \\
  --manifest ../dataset/data/processed/peptides_struct_manifest.parquet \\
  --output scored.parquet
```

BoltzGen diffusion guidance:

```bash
boltzgen run target.yaml \\
  --protocol peptide-anything \\
  --config design guidance.bbb_ckpt=/abs/path/bbb-geo/checkpoints/best.ckpt \\
  --config design guidance.bbb_weight=0.3 guidance.membrane_weight=0.7
```

## Dataset

Trained on Boltz-folded peptide structures. Companion dataset: [`manumartinm/bbb-peptides`](https://huggingface.co/datasets/manumartinm/bbb-peptides).

## Citation

If you use this model, cite the TFG BBB structural guidance pipeline.
"""


def export_model_bundle(cfg: HFModelExportConfig) -> dict[str, Any]:
    run_dir = cfg.run_dir.resolve()
    output_dir = cfg.output_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = _copy_inference_files(run_dir, output_dir, kind=cfg.kind)
    readme = (
        _render_geo_readme(run_dir, cfg.repo_id)
        if cfg.kind == "geo"
        else _render_classifier_readme(run_dir, cfg.repo_id)
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    meta = _load_json(run_dir / "train_metadata.json")
    stats = {
        "kind": cfg.kind,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "run_name": meta["exp_cfg"].get("name", meta["exp_cfg"]["model_type"]),
        "model_type": meta["exp_cfg"]["model_type"],
        "files": copied + ["README.md"],
        "repo_id": cfg.repo_id,
    }
    (output_dir / "export_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def export_classifier_hf(
    *,
    base_dir: str | Path = ".",
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> dict[str, Any]:
    base = Path(base_dir)
    run_dir = resolve_classifier_run_dir(base, run_name)
    out = output_dir or base / "artifacts" / "hf_release" / "bbb-classifier"
    return export_model_bundle(
        HFModelExportConfig(run_dir=run_dir, output_dir=Path(out), kind="classifier", repo_id=repo_id)
    )


def export_geo_hf(
    *,
    base_dir: str | Path = ".",
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> dict[str, Any]:
    base = Path(base_dir)
    run_dir = resolve_geo_run_dir(base, run_name)
    out = output_dir or base / "artifacts" / "hf_release" / "bbb-geo"
    return export_model_bundle(
        HFModelExportConfig(run_dir=run_dir, output_dir=Path(out), kind="geo", repo_id=repo_id)
    )
