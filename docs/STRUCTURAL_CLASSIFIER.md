# Structural BBB Classifier: EGNN geométrico (`bbb_geo`)

> Status: IMPLEMENTED — modelo único `struct_egnn_geo` (`p_geo`). El oracle de secuencia/tabular sigue siendo el clasificador fusion `exp03` en `bbb_classifier`.

Código: `bbb_models/src/bbb_geo/` (`struct_egnn.py`, `struct_graph.py`, `membrane_potential.py`, `struct_loader.py`, `infer/struct_guidance.py`).

## 1. Rol en el pipeline

| Artefacto | Modelo | Input | Uso |
|-----------|--------|-------|-----|
| Oracle / G3 / reward | `exp03_esm_tab_mlp` (`bbb_classifier`) | secuencia + tabular + ESM | filtrado y TD3B |
| Guidance difusión | `exp09_struct_egnn_noise` (`bbb_geo`) | coordenadas 3D + química por residuo + σ | gradiente `-∇_x E` en BoltzGen |

Se eliminó el modelo fusion estructural `struct_egnn_full` del flujo: un solo EGNN geométrico entrenado con ruido EDM.

## 2. Grafo estructural

Módulo `features/struct_graph.py`:

- **Nodos** (por residuo): one-hot AA + hidrofobicidad Kyte-Doolittle + carga. Acoplados a posiciones → gradiente w.r.t. coords.
- **Aristas**: grafo por radio (10 Å en Cα) con RBF de distancia + vector unitario relativo.
- **Salida**: `coords (n,3)`, `node_feats`, `edge_index` — autograd sobre coords.

## 3. Modelo `struct_egnn_geo`

EGNN equivariante E(n) en PyTorch puro (sin torch_geometric).

- Condicionado en `sigma` vía embedding `c_noise` (misma familia que BoltzGen).
- Cabezas: logit BBB + regresores auxiliares (momento hidrofóbico 3D, fracción helicoidal, radio de giro).
- `chem_dropout` durante entrenamiento para evitar atajos por composición.

## 4. Potencial de anfipaticidad

`features/membrane_potential.py` — término analítico diferenciable:

```
mu_vec = sum_i h_i * (c_i - c_bar)
amphipathicity = || mu_vec ||
```

Roles: target auxiliar en entrenamiento + término garantizado en la energía de guidance (`w2 * amphipathicity`).

## 5. Entrenamiento con ruido (exp09)

Config: `configs/experiments/exp09_struct_egnn_noise.yaml`

```yaml
model_type: struct_egnn_geo
struct:
  coord_sigma_cap: 8.0      # cap de ruido en coords (estabilidad numérica)
  aux_weight: 0.1           # peso pérdida auxiliar geométrica
  low_mid_bias: 0.7         # muestreo σ sesgado a banda baja-media
  plddt_weight_floor: 0.1
validation:
  sigma_values: [0.0, 2.0, 4.0, 8.0]
  gate_grad_norm_threshold: 0.001
  gate_corr_threshold: 0.1
```

Hiperparámetros geo: `configs/train_geo.yaml` — `lr: 5e-4`, `grad_clip: 0.5`, `batch_size: 64`, `epochs: 80`.

### Fuentes de datos

- **Local / manifest:** `struct.manifest_path` → `peptides_struct_manifest.parquet`
- **HF release / Vast:** `dataset_root` + columna `structure_coords_path` en `peptides.parquet` (825 filas con coords)

### Estabilidad numérica

Durante el entrenamiento, batches con NaN/Inf se saltan con warning:

```
[warn] epoch=N: skipped M non-finite batches
```

Mitigaciones aplicadas (jun 2026):

1. `coord_sigma_cap`: 16 → **8**
2. `aux_weight`: 0.2 → **0.1**
3. LR y grad_clip más conservadores en `train_geo.yaml`

Sweep adicional: `scripts/geo/sweep_stability.py` sobre `coord_sigma_caps` y `aux_weights`.

## 6. Salidas post-entrenamiento (automáticas)

Al finalizar `scripts/geo/train.py`:

| Archivo | Contenido |
|---------|-----------|
| `metrics.json` | PR-AUC, MCC, Brier en validación (σ=0 implícito) |
| `metrics_multisigma.json` | Métricas en σ = 0, 2, 4, 8 |
| `guidance_gate.json` | Pass/fail para activar guidance (‖∇ log p_geo‖ + correlación anfipaticidad) |
| `val_predictions.parquet` | Predicciones calibradas en val |
| `train_metadata.json` | Features, dims, paths — obligatorio para inferencia |
| `checkpoints/best.pt` | Mejor checkpoint por PR-AUC |

Flag `--no-resume` fuerza entrenamiento desde cero.

## 7. Comandos

```bash
cd TFG/bbb_models

# Entrenar
uv run python scripts/geo/train.py \
  --exp configs/experiments/exp09_struct_egnn_noise.yaml \
  --data-config configs/data.yaml \
  --train-config configs/train_geo.yaml \
  --output-root artifacts

# CV 5-fold
uv run python scripts/geo/cv.py \
  --exp configs/experiments/exp09_struct_egnn_noise.yaml \
  --train-config configs/train_geo.yaml

# Gate manual (probe)
uv run python scripts/geo/probe.py \
  --run-dir artifacts/models/exp09_struct_egnn_noise \
  --manifest ../dataset/data/processed/peptides_struct_manifest.parquet

# Sweep estabilidad
uv run python scripts/geo/sweep_stability.py \
  --coord-sigma-caps 4,8,12 \
  --aux-weights 0.05,0.1,0.2
```

Entrenamiento remoto: ver [VAST_TRAINING.md](VAST_TRAINING.md).

## 8. Tests

- construcción de grafo: shapes, simetría de aristas;
- equivariancia: rotación/traslación invariante en score;
- forward con ruido en sweep de σ;
- gradiente finito y no nulo w.r.t. coords;
- potencial de membrana: gradiente analítico vs autograd.
