# Structural BBB Classifier: Geometric EGNN (`bbb_geo`)

> Status: IMPLEMENTED — single model `struct_egnn_geo` (`p_geo`). The sequence/tabular oracle remains the fusion classifier `exp03` in `bbb_classifier`.

Code: `packages/bbb_models/src/bbb_geo/` (`struct_egnn.py`, `struct_graph.py`, `membrane_potential.py`, `struct_loader.py`, `infer/struct_guidance.py`).

## 1. Role in the pipeline

| Artifact | Model | Input | Use |
|----------|-------|-------|-----|
| Oracle / G3 / reward | `exp03_esm_tab_mlp` (`bbb_classifier`) | sequence + tabular + ESM | filtering and TD3B |
| Diffusion guidance | `exp09_struct_egnn_noise` (`bbb_geo`) | 3D coords + per-residue chemistry + σ | gradient `-∇_x E` in BoltzGen |

The structural fusion model `struct_egnn_full` was removed from the flow: a single geometric EGNN trained with EDM noise.

## 2. Structural graph

Module `features/struct_graph.py`:

- **Nodes** (per residue): one-hot AA + Kyte-Doolittle hydrophobicity + charge. Coupled to positions → gradient w.r.t. coords.
- **Edges:** radius graph (10 Å on Cα) with distance RBF + relative unit vector.
- **Output:** `coords (n,3)`, `node_feats`, `edge_index` — autograd over coords.

## 3. Model `struct_egnn_geo`

E(n)-equivariant EGNN in pure PyTorch (no torch_geometric).

- Conditioned on `sigma` via `c_noise` embedding (same family as BoltzGen).
- Heads: BBB logit + auxiliary regressors (3D hydrophobic moment, helical fraction, radius of gyration).
- `chem_dropout` during training to avoid composition shortcuts.

## 4. Amphipathicity potential

`features/membrane_potential.py` — differentiable analytic term:

```
mu_vec = sum_i h_i * (c_i - c_bar)
amphipathicity = || mu_vec ||
```

Roles: auxiliary training target + guaranteed term in guidance energy (`w2 * amphipathicity`).

## 5. Noise-aware training (exp09)

Config: `configs/experiments/exp09_struct_egnn_noise.yaml`

```yaml
model_type: struct_egnn_geo
struct:
  coord_sigma_cap: 8.0      # noise cap on coords (numerical stability)
  aux_weight: 0.1           # auxiliary geometric loss weight
  low_mid_bias: 0.7         # σ sampling biased to low-mid band
  plddt_weight_floor: 0.1
validation:
  sigma_values: [0.0, 2.0, 4.0, 8.0]
  gate_grad_norm_threshold: 0.001
  gate_corr_threshold: 0.1
```

Geo hyperparameters: `configs/train_geo.yaml` — `lr: 5e-4`, `grad_clip: 0.5`, `batch_size: 64`, `epochs: 80`.

### Data sources

- **Local / manifest:** `struct.manifest_path` → `peptides_struct_manifest.parquet`
- **HF release / Vast:** `dataset_root` + `structure_coords_path` column in `peptides.parquet` (825 rows with coords)

### Numerical stability

During training, batches with NaN/Inf are skipped with a warning:

```
[warn] epoch=N: skipped M non-finite batches
```

Mitigations applied (Jun 2026):

1. `coord_sigma_cap`: 16 → **8**
2. `aux_weight`: 0.2 → **0.1**
3. More conservative LR and grad_clip in `train_geo.yaml`

Additional sweep: `bbb-geo sweep-stability` over `coord_sigma_caps` and `aux_weights`.

## 6. Post-training outputs (automatic)

At the end of `bbb-geo train`:

| File | Contents |
|------|----------|
| `metrics.json` | PR-AUC, MCC, Brier on validation (σ=0 implicit) |
| `metrics_multisigma.json` | Metrics at σ = 0, 2, 4, 8 |
| `guidance_gate.json` | Pass/fail to enable guidance (‖∇ log p_geo‖ + amphipathicity correlation) |
| `val_predictions.parquet` | Calibrated predictions on val |
| `train_metadata.json` | Features, dims, paths — required for inference |
| `checkpoints/best.pt` | Best checkpoint by PR-AUC |

`--no-resume` forces training from scratch.

## 7. Commands

```bash
cd packages/bbb_models

# Train
uv run python bbb-geo train \
  --exp configs/experiments/exp09_struct_egnn_noise.yaml \
  --data-config configs/data.yaml \
  --train-config configs/train_geo.yaml \
  --output-root artifacts

# 5-fold CV
uv run python bbb-geo cv \
  --exp configs/experiments/exp09_struct_egnn_noise.yaml \
  --train-config configs/train_geo.yaml

# Manual gate (probe)
uv run python bbb-geo probe \
  --run-dir artifacts/models/exp09_struct_egnn_noise \
  --manifest ../dataset/data/processed/peptides_struct_manifest.parquet

# Stability sweep
uv run python bbb-geo sweep-stability \
  --coord-sigma-caps 4,8,12 \
  --aux-weights 0.05,0.1,0.2
```

Remote training: see [vast-training.md](../infrastructure/vast-training.md).

## 8. Tests

- graph construction: shapes, edge symmetry;
- equivariance: rotation/translation invariance in score;
- forward with noise across σ sweep;
- finite, non-zero gradient w.r.t. coords;
- membrane potential: analytic gradient vs autograd.
