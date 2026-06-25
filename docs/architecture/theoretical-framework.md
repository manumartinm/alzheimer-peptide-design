# Theoretical Framework

This document translates the OxML poster (`entregas/oxml_bio/main.tex`) into a structured technical reference.

It explains:
- why the project is designed this way;
- how each stage connects mathematically and operationally;
- how agents should implement future BoltzGen-related work without violating core assumptions.

## 1. Biological Motivation

Target: **GSK3β** in Tau hyperphosphorylation context.

Problem with conventional strategy:
- ATP-competitive inhibition is often non-selective across kinases.
- ATP-cleft targeting increases risk of off-target pathway disruption (notably Wnt signaling).

Project strategy:
- move from ATP-cleft inhibition to **substrate-selective modulation**;
- design cyclic phosphomimetic peptides to engage substrate-recognition residues;
- enforce BBB compatibility as a translational constraint.

## 2. End-to-End Pipeline (4 Stages)

Stage 1 — Dataset:
- curated BBB peptide dataset with sequence and descriptors.

Stage 2 — BBB Classifier:
- calibrated predictor (`p_bbb_calibrated`) from ESM + tabular features.

Stage 3 — Classifier-Guided Diffusion (BoltzGen + TD3B):
- all-atom peptide generation with geometric guidance.
- BBB integration via reward-based amortized tuning, not reverse-SDE gradient.

Stage 4 — Filtering + MD:
- five-gate triage and Pareto selection;
- top candidates enter explicit-solvent MD.

## 3. Stage 2 Formalism: BBB Classifier

Poster-level formulation:

- ESM latent:
  \[
  \mathbf{h}_{\mathrm{ESM}}=W_e\left(\frac{1}{L}\sum_{i=1}^{L}H_i\right)+b_e
  \]
- fused latent:
  \[
  \mathbf{z}=[\mathbf{h}_{\mathrm{ESM}};\mathbf{h}_{\mathrm{tab}}]
  \]
- calibrated probability:
  \[
  p_{\mathrm{BBB}}^{\mathrm{cal}}=g_{\mathrm{iso}}\big(\sigma(\mathrm{MLP}(\mathbf{z}))\big)
  \]

Interpretation:
- raw logits produce rank signal;
- isotonic calibration improves probability reliability for thresholding and reward weighting.

## 4. Stage 3 Formalism: BoltzGen with Geometric Guidance

BoltzGen is treated as an all-atom diffusion prior over coordinates \(x\in\mathbb{R}^{N\times 3}\).

Forward/reverse VP-SDE view:
\[
\mathrm{d}x_t=-\tfrac{1}{2}\beta_tx_t\,\mathrm{d}t+\sqrt{\beta_t}\,\mathrm{d}w_t
\]
\[
\mathrm{d}x_t=\left[-\tfrac{1}{2}\beta_tx_t-\beta_ts_{\theta_0}(x_t,t,c)\right]\mathrm{d}t+\sqrt{\beta_t}\,\mathrm{d}\bar{w}_t
\]

Geometric guidance used during reverse diffusion:
\[
\tilde{s}_{\theta}(x_t,c)=s_{\theta_0}(x_t,t,c)+w_h\nabla\log p_h-w_a\nabla\log p_a
\]

Where:
- \(p_h\): hotspot-contact rewarding potential;
- \(p_a\): ATP-cleft avoidance potential;
- \(w_h,w_a\): controllable guidance weights.

## 5. Critical Constraint: Why BBB Is Not a Reverse-SDE Gradient

Key architectural rule from the poster:

- BBB scoring depends on discrete sequence representations (ESM embeddings and derived features).
- reverse SDE evolves continuous 3D coordinates.
- direct BBB gradient wrt coordinates is not well-defined in this pipeline design.

Therefore (sequence-only classifier):
- **do not inject BBB term in reverse-SDE guidance**;
- inject BBB only via TD3B reward fine-tuning.

**Planned extension (structural classifier):** a geometry-only model `p_geo(BBB | x, sigma)` plus an analytic amphipathicity potential can provide a coordinate-differentiable energy term at low noise levels. The fusion oracle (`p_full`) remains sequence+structure for reward/G3. See [structural-bbb-guidance.md](../models/structural-bbb-guidance.md).

This is the most important implementation guardrail for agents.

## 6. TD3B Amortized Fine-Tuning

Poster concept:
\[
p^\star(y\mid d^\star,x)\propto p_{\theta_0}(y)\exp\!\left(R(y;d^\star,x)/\alpha\right)
\]

Reward design:
\[
R=g_\psi(y,x)\cdot g_{\mathrm{BBB}}(y)\cdot \sigma\!\left(\frac{d^\star f_\phi}{\tau}\right)
\]

Interpretation:
- affinity quality, BBB delivery, and directional objective are gated together;
- TD3B distills guidance into model weights so generation improves without requiring unstable high-weight inference guidance.

Training objective (poster-level):
\[
\min_{\theta}\ \mathcal{L}_{\mathrm{WDCE}}+\lambda_{\mathrm{ctr}}\mathcal{L}_{\mathrm{ctr}}+\lambda_{\mathrm{reg}}\mathrm{KL}(p_\theta\Vert p_{\theta_0})
\]

## 7. Stage 4: Five-Gate Filtering Logic

After generation, candidates pass strict triage:

1. **G1 Hotspot Satisfaction**
   - soft surrogate used for differentiable scoring:
   \[
   M_{\mathrm{hot}}^{\mathrm{soft}}=\frac{1}{|H|}\sum_{r\in H}\sigma\!\left(\alpha\left(5\text{\AA}-\min_{a\in y}\|x_a-x_r\|\right)\right)
   \]
2. **G2 ATP-Cleft Avoidance**
   - reject structures with high ATP-proximity penalty.
3. **G3 BBB + Developability**
   - enforce calibrated BBB threshold and developability checks.
4. **G4 Structural Confidence**
   - ipTM/pLDDT/closure consistency thresholds.
5. **G5 Sequence Liabilities**
   - remove problematic motifs (aggregation, oxidation risk, etc.).

Only top Pareto survivors progress to MD.

## 8. Expected Compute Behavior

From poster narrative:
- large candidate generation batches (order of tens of thousands per target);
- filtering compresses candidate space to a small MD-ready subset;
- explicit-solvent MD used as the final computational validation stage before experimental work.

## 9. Agent Implementation Checklist

If you are an agent implementing next steps:

1. Keep geometric guidance and BBB reward pathways separate.
2. Preserve equation-level consistency between implementation and reported methodology.
3. Maintain calibrated BBB probability as the operational confidence signal.
4. Keep filtering gates auditable and threshold-driven.
5. Record all outputs needed for reproducibility (configs, random seeds, scoring logs, candidate provenance).

## 10. Scope Boundaries and Claim Discipline

This framework supports **pre-experimental in silico prioritization**, not clinical or biochemical efficacy claims.

Any downstream documentation or paper text must clearly state:
- predictions are computational;
- wet-lab assays are future validation work.
