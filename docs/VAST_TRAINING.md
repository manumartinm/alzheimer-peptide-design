# Entrenamiento remoto en Vast.ai (bbb_models)

> Status: IMPLEMENTED (`packages/packages/bbb_models/scripts/vast/` + `infra/vast/_common.sh`).

Entrena el clasificador BBB o el EGNN geométrico en una **instancia Vast existente** (creada manualmente en la web). No busca ofertas ni provisiona máquinas: solo sube código + dataset y lanza el job.

## Qué se sube

| Contenido local | Destino remoto |
|-----------------|----------------|
| `packages/bbb_models/` (sin `artifacts/`, `.venv`) | `/workspace/packages/bbb_models/` |
| `packages/dataset/data/hf_release/` | `/workspace/packages/dataset/data/hf_release/` |

El release HF es obligatorio para geo: contiene `peptides.parquet` + `structures/<hash>/coords.npz` (825 péptidos con estructura). Generarlo antes:

```bash
cd packages/dataset
uv run tfg-bbb-export-hf --variant full
```

## Launcher principal

Desde la raíz del monorepo:

```bash
bash packages/packages/bbb_models/scripts/vast/launch.sh
bash packages/packages/bbb_models/scripts/vast/upload_workspace.sh <INSTANCE_ID>
bash packages/packages/bbb_models/scripts/vast/setup_instance.sh <INSTANCE_ID>
bash packages/packages/bbb_models/scripts/vast/run_train.sh <INSTANCE_ID>
```

Variables de entorno:

| Variable | Default | Descripción |
|----------|---------|-------------|
| `MODE` | `geo` | `geo` → `scripts/geo/train.py`; `classifier` → `scripts/classifier/train.py` |
| `EXP` | `exp09_struct_egnn_noise.yaml` | Config de experimento |
| `TRAIN_CONFIG` | `train_geo.yaml` | Hiperparámetros (geo usa LR/grad_clip conservadores) |
| `DATA_CONFIG` | `data.vast.yaml` | Paths remotos al HF release |
| `OUTPUT_ROOT` | `artifacts` | Carpeta de salida remota |
| `SMOKE=1` | — | Usa `train_smoke.yaml`, 2 epochs, salida en `artifacts/smoke_geo` |
| `CV=1` | — | Lanza `scripts/geo/cv.py` con `train_cv.yaml` |
| `NO_RESUME=1` | activo | Entrenamiento limpio (`--no-resume`) |
| `FORCE_CPU=1` | — | Fuerza CPU si la GPU es incompatible (p. ej. RTX 5090 + PyTorch antiguo) |

Ejemplos:

```bash
SMOKE=1 bash packages/bbb_models/scripts/vast/run_train.sh 42405703
CV=1 bash packages/bbb_models/scripts/vast/run_cv.sh 42405703
MODE=classifier EXP=configs/experiments/exp03_esm_tab_mlp.yaml \
  TRAIN_CONFIG=configs/train.yaml \
  bash packages/bbb_models/scripts/vast/run_train.sh 42405703
```

## Scripts auxiliares

Todos en `packages/bbb_models/scripts/vast/`:

| Script | Uso |
|--------|-----|
| `monitor.sh <INSTANCE_ID>` | Tail del log más reciente en `/workspace/output/` |
| `sync_artifacts.sh <INSTANCE_ID>` | Descarga `artifacts/` remoto a local |
| `status.sh` | Estado de la instancia |
| `setup_instance.sh` | Solo pip install en remoto |
| `upload_workspace.sh` | Solo upload (sin train) |
| `run_train.sh` / `run_cv.sh` | Train/CV en instancia ya preparada |
| `destroy.sh` | Destruir instancia (destructivo) |

Logs remotos: `/workspace/output/<exp>_train.log` o `*_cv.log`. PIDs en `last_train.pid` / `last_cv.pid`.

## SSH y claves

Helpers en `infra/vast/_common.sh`:

- Usuario SSH: **`root@`** (no `vastai@`).
- Identidad: `~/.ssh/id_ed25519` (`VAST_SSH_IDENTITY` para override).
- Flag: `-o IdentitiesOnly=yes`.
- Preferir endpoint proxy Vast (`ssh9.vast.ai:PORT`) sobre IP directa.

Registrar la clave en la cuenta Vast con **contenido** del `.pub`, no la ruta del fichero:

```bash
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
```

Si la clave quedó registrada como path (`/Users/.../id_ed25519.pub`), borrarla y volver a crear. `ensure_vast_ssh_key` detecta y corrige esto automáticamente.

## Config de datos remota

`configs/data.vast.yaml`:

```yaml
dataset_path: /workspace/packages/dataset/data/hf_release/peptides.parquet
dataset_root: /workspace/packages/dataset/data/hf_release
struct_manifest_path: ""   # vacío: geo resuelve coords desde hf_release
```

`build_features` usa `structure_coords_path` relativo en el parquet cuando no hay manifest.

## GPU conocidas / limitaciones

| Hardware | Notas |
|----------|-------|
| A100 / H100 | Recomendado para geo |
| RTX 5090 (sm_120) | PyTorch 2.4.x del image puede fallar → `FORCE_CPU=1` o imagen más nueva |
| CPU | Válido para smoke; entrenamiento completo muy lento (~horas) |

## Artefactos esperados (geo)

Tras entrenar, en `artifacts/models/exp09_struct_egnn_noise/` (o subcarpeta CV):

- `checkpoints/best.pt`, `checkpoints/last.pt`
- `metrics.json`, `metrics_multisigma.json`
- `guidance_gate.json`
- `val_predictions.parquet`, `train_metadata.json`
- calibrador isotónico (si enabled)

Descargar:

```bash
bash packages/bbb_models/scripts/vast/sync_artifacts.sh <INSTANCE_ID>
```

## Flujo recomendado

1. Export HF release local (`tfg-bbb-export-hf --variant full`).
2. Crear instancia Vast (GPU A100/H100, disco ≥ 30 GB).
3. `SMOKE=1 packages/bbb_models/scripts/vast/run_train.sh <ID>` → verificar SSH, datos y pip.
4. `packages/bbb_models/scripts/vast/run_train.sh <ID>` → entrenamiento completo.
5. `monitor.sh` hasta convergencia; revisar warnings `skipped N non-finite batches`.
6. `sync_artifacts.sh` → consolidar localmente.
