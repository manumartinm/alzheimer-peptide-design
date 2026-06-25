# Reproducibilidad

## v1 (actual — jun 2026)

### Qué está cubierto

| Componente | Reproducibilidad |
|------------|------------------|
| Dataset gold | `tfg-bbb-build` + parquets en `dataset/data/processed/` |
| HF release | `tfg-bbb-export-hf --variant full` → `data/hf_release/` (825 filas) |
| Clasificador | `scripts/classifier/train.py` + configs + `train_metadata.json` |
| EGNN geo | `scripts/geo/train.py` + `exp09` + `train_geo.yaml` |
| CV | `scripts/classifier/cv.py`, `scripts/geo/cv.py` |
| Remote GPU | `bbb_models/scripts/vast_launch.sh` + `sync_artifacts.sh` |

### Comandos mínimos

```bash
# Entorno
cd TFG && uv sync

# Dataset
cd dataset && uv run tfg-bbb-build
uv run tfg-bbb-export-hf --variant full

# Clasificador local
cd ../bbb_models
uv run python scripts/classifier/train.py \
  --exp configs/experiments/exp03_esm_tab_mlp.yaml

# Geo local
uv run python scripts/geo/train.py \
  --exp configs/experiments/exp09_struct_egnn_noise.yaml \
  --train-config configs/train_geo.yaml

# Tests
uv run pytest
```

### Artefactos (no en git)

- `bbb_models/artifacts/` — checkpoints, métricas, MLflow local
- `dataset/boltz-experiments/` — runs de folding Boltz
- `dataset/data/hf_release/structures/` — coords + CIF (grande)

Política: no commitear checkpoints; usar `sync_artifacts.sh` desde Vast o paths locales ignorados.

### Tracking

- MLflow: `bbb_models/mlflow.db` (local)
- TensorBoard: bajo cada run dir
- Post-geo: `metrics_multisigma.json`, `guidance_gate.json`

## v2 (roadmap)

> Objetivo: `uv sync` + `dvc pull` + API keys → regenerar todo el pipeline.

| Tarea | Paquete |
|-------|---------|
| `dataset/dvc.yaml`: fetch_raw → build → augment → fold → export_hf | `dataset/` |
| Arreglar paths en `bbb_models/dvc.yaml` + stages geo/CV | `bbb_models/` |
| DVC remote (S3/GDrive) para raw, structures, checkpoints | root |
| Checksums SHA256 en descargas B3Pred | `dataset/src/tfg_bbb/sources.py` |
| DATA_CARD con provenance (git SHA + dvc.lock) | `dataset/` |
| CI: `dvc repro --dry` | `.github/workflows/` |

### Comandos previstos (v2)

```bash
uv sync
dvc pull                    # descargar datos y modelos cacheados
dvc repro build_gold        # regenerar dataset desde raw
dvc repro train             # re-entrenar clasificador
dvc repro train_geo         # re-entrenar EGNN
```

Ver [VAST_TRAINING.md](VAST_TRAINING.md) para campañas remotas mientras v2 no esté completo.
