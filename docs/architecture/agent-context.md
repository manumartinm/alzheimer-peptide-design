# GSK3β Modulator Design: Project Context & Agent Guidelines

**Last Updated:** June 2026
**Project:** Bachelor's Thesis (TFG) - *In silico* design of BBB-compatible cyclic peptide modulators of GSK3β for Tau hyperphosphorylation.

> **⚠️ AGENT INSTRUCTION:**
> If you are an AI agent reading this file, treat it as the **Source of Truth** for the project's architecture, methodology, and planned tasks. Use this context to understand *why* certain architectural decisions were made (especially regarding differentiability and diffusion guidance) and *how* to execute the next steps.

---

## 1. High-Level Project Goal
The objective is to computationally design cyclic phosphomimetic peptides that modulate the **GSK3β kinase** (implicated in Alzheimer's disease via Tau hyperphosphorylation).
Unlike traditional inhibitors that target the highly conserved ATP cleft (causing Wnt-pathway off-target toxicity), this project aims for **substrate-selective modulation**. The peptides must:
1. Engage the substrate-recognition groove (hotspots: R96, R180, K205).
2. Avoid the ATP-binding cleft.
3. Cross the Blood-Brain Barrier (BBB).

---

## 2. Repository Structure

*   `pyproject.toml`: Root uv workspace configuration for a single shared environment.
*   `packages/bbb_models/`: BBB models workspace — `bbb_classifier` (tabular/ESM) + `bbb_geo` (EGNN guidance).
*   `packages/boltzgen/`: Git submodule (`manumartinm/boltzgen`, branch `add-bbb-head-and-md`) with the diffusion engine and MD utilities.
*   `packages/boltzgen_design/`: Orchestration layer for GSK3β design (target prep, geometric guidance, TD3B utilities, filtering scripts).
*   `packages/dataset/`: Dataset curation package + `tfg-bbb-build` CLI pipeline.
*   `docs/`: Project documentation and agent guidelines (this folder).

---

## 3. Phase 1: BBB Classifier (Completed)
**Location:** `packages/bbb_models/` (`src/bbb_classifier`, `src/bbb_geo`)

This phase acts as the "delivery oracle" for the generative pipeline. It predicts whether a given peptide sequence will cross the BBB.

*   **Data Pipeline:** Curated from B3Pred D1 with optional B3Pdb/Brainpeps expansion. Filtered for canonical amino acids, length 6-30 by default (configurable), and deduplicated at 90% sequence identity with cluster-aware folds to reduce leakage.
*   **HF release:** `packages/dataset/data/hf_release/` — 825 peptides with Boltz structures (`tfg-bbb-export-hf --variant full`). Used for geo training and Vast upload.
*   **Features:**
    *   ESM-2 language model embeddings ($d=128$).
    *   24 tabular physicochemical descriptors (charge, pI, GRAVY, instability, etc.).
*   **Architecture (oracle):** `exp03_esm_tab_mlp` — MLP fusing ESM-2 and tabular branches. Heavily regularized (dropout $p=0.3$, weight decay $\lambda=10^{-3}$).
*   **Architecture (guidance):** `exp09_struct_egnn_geo` — geometry-only EGNN with EDM noise conditioning. See [structural-classifier.md](../models/structural-classifier.md).
*   **Calibration:** Isotonic regression on validation probs. Use `p_bbb_calibrated` for gating/reward.
*   **Agent Execution:** Prefer `uv run python scripts/classifier/train.py` or `scripts/geo/train.py`. DVC stages exist but geo/Vast workflows are often run directly. Remote training: [vast-training.md](../infrastructure/vast-training.md).

---

## 4. Phase 2: Generative Pipeline with BoltzGen (In Progress)
**Location:** `packages/boltzgen_design/` + guidance hooks in `packages/boltzgen/src/boltzgen/model/modules/diffusion.py`

This is the core generative phase using **BoltzGen** (an all-atom diffusion model).

### 4.1. Target Preparation
*   **Target:** GSK3β structure.
*   **Positive Mask (Hotspots):** R96, R180, K205 (primary) and F67, Q89, N95 (secondary).
*   **Negative Mask:** ATP-binding cleft residues.
*   **Bounding Box:** 8-12 Å shell around the substrate-recognition groove.

### 4.2. Geometric Guidance (Reverse SDE)
During inference, the reverse diffusion SDE is guided by differentiable spatial potentials:
*   $U_h$: Rewards atoms being $\le 5$ Å from the positive hotspots.
*   $U_a$: Penalizes proximity (Lennard-Jones style repulsion) to the ATP cleft.
*   **Agent Note:** Inject $w_h \nabla \log p_h - w_a \nabla \log p_a$ into the BoltzGen sampler.

### 4.3. TD3B Fine-Tuning for BBB Permeability
> **🚨 CRITICAL ARCHITECTURAL CONSTRAINT:**
> The BBB classifier requires discrete sequences (for ESM-2) and tabular features (like pI, GRAVY). **These are non-differentiable with respect to 3D Cartesian coordinates.** Therefore, BBB guidance CANNOT be applied as a standard gradient during the reverse SDE.

> **🧪 STRUCTURAL BBB GUIDANCE (implemented, wiring in progress):** A geometry-only classifier `p_geo(BBB | x, sigma)` over coordinates makes the BBB energy differentiable wrt coordinates and can be injected into the reverse SDE alongside geometric potentials. The global tabular/ESM oracle (`exp03`) stays in reward/G3, out of the guidance gradient. Training: `exp09_struct_egnn_noise` + post-hoc `guidance_gate.json`. See [structural-bbb-guidance.md](../models/structural-bbb-guidance.md), [structural-classifier.md](../models/structural-classifier.md), [boltz-folding.md](../data/boltz-folding.md), [vast-training.md](../infrastructure/vast-training.md).

*   **Solution (current):** We use the **TD3B framework** (Transition-Directed Discrete Diffusion).
*   **Mechanism:** BBB permeability is injected as a **gated reward signal** during an amortized fine-tuning phase.
*   **Implementation:**
    1. Generate a batch of trajectories.
    2. Decode the sequences and score them with the trained BBB classifier.
    3. Use the continuous calibrated probability $p_{\text{BBB}}^{\text{cal}}$ as a reward.
    4. Fine-tune BoltzGen weights ($\theta$) using Weighted Denoising Cross-Entropy (WDCE) loss, anchored with a KL divergence penalty to the original prior to prevent mode collapse.

---

## 5. Phase 3: Filtering Cascade & MD Validation (In Progress)

### 5.1. The 5-Gate Filtering Cascade
Agents must implement a script to filter the ~60,000 generated candidates through 5 strict gates:
1.  **G1 (Hotspots):** $>70\%$ of hotspot residues engaged within 5 Å.
2.  **G2 (ATP-avoidance):** Repulsion score below a safety threshold.
3.  **G3 (BBB & Developability):** $p_{\text{BBB}}^{\text{cal}} \ge 0.6$ + solubility checks.
4.  **G4 (Structural Confidence):** BoltzGen metrics: ipTM $\ge 0.75$, pLDDT $\ge 85$, cyclic-closure RMSD $\le 1.2$ Å.
5.  **G5 (Sequence Liabilities):** Reject polybasic clusters, deamidation motifs, oxidation-prone residues (Met/Cys), and TANGO aggregation hotspots.

### 5.2. Molecular Dynamics (MD) Validation
*   **Selection:** Top 10-30 Pareto-optimal candidates surviving the cascade.
*   **Setup:** OpenMM, CHARMM36m force field, TIP3P water, 0.15 M salt, 310.15 K.
*   **Protocol:** 100 ns exploratory runs per candidate. The best lead is extended to 500 ns.
*   **Metrics:** Interface RMSD, hotspot contact persistence, ATP-cleft separation, polar SASA.

---

## 6. AI Agent Standard Operating Procedures (SOPs)

When tasked with continuing this project, follow these rules:
1.  **Read this file first:** Always refresh your context with this architecture document.
2.  **Respect DVC:** For the BBB classifier, do not run python scripts manually if a `dvc.yaml` stage exists. Use `dvc repro`.
3.  **Acknowledge Non-Differentiability:** Never attempt to backpropagate 3D coordinates through ESM-2 or tabular descriptors. Always use RL/TD3B for sequence-based constraints.
4.  **Use the unified environment:** Run commands through the root uv workspace (`uv sync`, `uv run ...`) instead of creating per-package virtual environments.
5.  **Math reference:** Use [theoretical-framework.md](theoretical-framework.md) as the canonical formulas reference for SDE guidance, TD3B, and filtering gates.
6.  **Geo training stability:** If you see many `skipped non-finite batches`, lower `coord_sigma_cap` / `aux_weight` in exp09 or use `train_geo.yaml`. Run `sweep_stability.py` before long Vast jobs.
7.  **Vast workflow:** Upload `bbb_models/` + `hf_release/` only. Use `infra/vast/bbb_models/run_train.sh` on an existing instance; see [vast-training.md](../infrastructure/vast-training.md).
8.  **Artifact policy:** Do not commit large artifacts/checkpoints to git; keep them in ignored paths or track with DVC metadata only.
9.  **Document Changes:** If you alter the pipeline architecture, update [agent-context.md](agent-context.md) to reflect the new reality.
