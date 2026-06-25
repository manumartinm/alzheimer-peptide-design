# Project Architecture

**Project:** Bachelor's thesis — *In silico* design of BBB-compatible cyclic phosphomimetic peptides that modulate GSK3β.

## Scientific goal

Design peptides computationally that:

1. Bind the GSK3β substrate-recognition groove (hotspots: R96, R180, K205).
2. Avoid the ATP-binding cleft (off-target toxicity via Wnt signaling).
3. Cross the blood–brain barrier (BBB).

## Monorepo layout

```
alzheimer-peptide-design/
├── packages/
│   ├── dataset/           → BBB data curation (CLI: bbb-dataset-build)
│   ├── bbb_models/        → classifier + geometric EGNN
│   ├── boltzgen_design/   → GSK3β orchestration (guidance, filters)
│   └── boltzgen/          → diffusion engine (submodule)
└── docs/                  → this documentation
```

Unified environment: `uv sync` at the repo root installs all packages in editable mode.

## Phase 1: BBB classifier (completed)

**Location:** `packages/bbb_models/` (`bbb_classifier`, `bbb_geo`)

Permeability oracle that feeds ranking, filtering, and (indirectly) generation.

- **Data:** B3Pred D1 + optional expansion. Length 6–30 aa, 90% deduplication, cluster-aware folds.
- **HF release:** `packages/dataset/data/hf_release/` — 825 peptides with Boltz structures (`bbb-dataset-export-hf --variant full`).
- **Oracle (sequence):** `exp03_esm_tab_mlp` — ESM-2 + tabular, isotonic calibration.
- **Guidance (geometry):** `exp09_struct_egnn_geo` — EGNN with EDM noise; see [structural-classifier.md](../models/structural-classifier.md).
- **Execution:** `uv run python bbb-classifier train` or `bbb-geo train`; remote on [vast-training.md](../infrastructure/vast-training.md).

## Phase 2: BoltzGen generation (in progress)

**Location:** `packages/boltzgen_design/` + hooks in `packages/boltzgen/src/boltzgen/model/modules/diffusion.py`

### Geometric guidance (reverse SDE)

Differentiable potentials during inference:

- **U_h:** reward for proximity (≤ 5 Å) to positive hotspots.
- **U_a:** penalty for proximity to the ATP cleft.

### BBB and non-differentiability

The sequence classifier (`exp03`) uses ESM-2 and tabular descriptors — **not differentiable w.r.t. 3D coordinates**. Therefore:

- **Gradient guidance in SDE:** ATP hotspots + geometric EGNN `p_geo` + amphipathicity potential.
- **Sequence BBB signal:** TD3B (amortized reward with WDCE + KL anchor) using `p_bbb_calibrated` from the oracle.

The `struct_egnn_geo` model (exp09) trained on folded structures enables differentiable BBB guidance. See [structural-bbb-guidance.md](../models/structural-bbb-guidance.md).

## Phase 3: Filtering and MD (in progress)

### Five-gate cascade

1. **G1:** >70% hotspots engaged ≤ 5 Å.
2. **G2:** ATP repulsion score below threshold.
3. **G3:** p_BBB ≥ 0.6 + solubility.
4. **G4:** ipTM ≥ 0.75, pLDDT ≥ 85, cyclic closure RMSD ≤ 1.2 Å.
5. **G5:** sequence liabilities (polybasic, deamidation, Met/Cys, aggregation).

### MD validation

Top 10–30 Pareto candidates → OpenMM, CHARMM36m, 100 ns exploratory, 500 ns for the lead.

## Conventions for contributors

1. Read [agent-context.md](agent-context.md) for full detail.
2. Respect DVC when a stage exists (`dvc repro`, not ad-hoc scripts).
3. Do not backpropagate 3D coordinates through ESM-2 or tabular descriptors.
4. Use `uv run` from the repo root.
5. Do not commit large artifacts; use ignored paths or DVC (v2).
6. Update docs if you change the architecture.
