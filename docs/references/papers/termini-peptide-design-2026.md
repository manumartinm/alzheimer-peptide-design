# Termini: Modification-Aware AI for Peptide Design and Discovery of Potent Antimicrobials

**PDF:** `2026.04.09.717597v1.full (2).pdf`
**Preprint:** bioRxiv (posted April 10, 2026)
**DOI:** https://doi.org/10.64898/2026.04.09.717597
**Authors:** Jing Xu, Marcelo D. T. Torres, Chen Li, Jian Li, Fuyi Li, Jiangning Song, Cesar de la Fuente-Nunez
**Affiliations:** Monash University, University of Adelaide, University of Pennsylvania

---

## One-sentence summary

**Termini** is an integrative AI pipeline that combines **ESM-2 latent diffusion** for peptide generation with **species-specific Chemprop classifiers**, **MIC regression**, and **toxicity filtering** — explicitly modeling **N-terminal acetylation** and **C-terminal amidation** — achieving a **92.5% experimental hit rate** (111/120 peptides active) across 11 pathogens with in vivo validation.

---

## 1. Problem it solves

Antimicrobial resistance (AMR) is projected to become the leading cause of death worldwide by 2050. Antimicrobial peptides (AMPs) are promising antibiotic alternatives, but the design space is enormous (≤50 residues, 20 canonical amino acids → astronomical combinatorial space).

Existing generative AMP pipelines have three key limitations:

1. **Terminal modifications ignored** — N-acetylation and C-amidation are routinely used in peptide therapeutics but treated as post-hoc optimization, not design variables.
2. **Narrow validation panels** — Most studies test against 1–3 bacterial species.
3. **Modest hit rates** — Prior generative studies achieve 25–70% activity among synthesized candidates.

Termini addresses all three with a modification-aware, multi-species, generative–predictive framework.

---

## 2. Key advances (contributions)

1. **Termini framework** — End-to-end pipeline: generate → classify (15 species) → regress (MIC) → filter toxicity → synthesize → validate.

2. **ESM-2 latent diffusion** — Denoising in ESM-2 embedding space with optional physicochemical conditioning (net charge, hydrophobic ratio) and classifier-free guidance.

3. **Modification-aware prediction** — SMILES encoding captures N-acetyl and C-amide chemistry; models predict how terminal capping shifts activity.

4. **Largest validation to date** — 120 peptides (60 unique sequences × 2 terminal states) tested against **11 clinically relevant pathogens** including ESKAPEE members.

5. **Exceptional hit rate** — 111/120 (92.5%) active in vitro; 63/120 broad-spectrum (Gram+ and Gram−).

6. **Terminal modifications enhance potency** — C-terminal amidation frequently lowers MICs; effects are sequence-dependent but directionally predictable.

7. **In vivo proof-of-concept** — Lead peptides reduce *A. baumannii* burden in murine skin infection model, comparable to polymyxin B.

---

## 3. Methodology

### 3.1 Pipeline overview (Figure 1A)

```
32,095 AMP sequences (5–30 aa, curated from 12+ databases)
        ↓
ESM-2 latent diffusion → 1,000 peptides/length → 26,000 candidates
        ↓
Multi-tier computational screening:
  ├── Species-specific classifiers (15 bacteria/fungi, Chemprop D-MPNN)
  ├── MIC regression models (log(MIC+1), Chemprop)
  └── Toxicity classifier (hemolysis/cytotoxicity, Chemprop)
        ↓
Top candidates synthesized (120 peptides, 60 unique backbones)
        ↓
In vitro: MIC assays (11 species) + cytotoxicity/hemolysis
        ↓
In vivo: murine skin scarification model (A. baumannii)
```

### 3.2 ESM-2 latent diffusion model

Unlike structure-based methods (BoltzGen, RFdiffusion), Termini operates purely in **sequence embedding space**:

```
Forward:  ESM-2 encode(sequence) → z
          Add Gaussian noise iteratively → z_t

Reverse:  Denoiser (repurposed ESM-2 transformer + lightweight MLP)
          Predicts denoised embeddings (MSE loss)

Decode:   ESM-2 LM head → argmax per position → peptide sequence
```

**Conditioning:**
- Diffusion timestep (always).
- Continuous properties: **net charge** and **hydrophobic ratio** (optional).
- Lightweight additive conditioning mechanism.
- **Conditional dropout** during training → **classifier-free guidance** at inference (unconditional or property-conditioned generation).

**Design choice vs AMP-diffusion:** Termini reuses ESM-2 forward dynamics with minimal changes rather than a dedicated denoising transformer with sinusoidal timestep embeddings and scale–shift modulation. This keeps generated sequences closer to the native ESM representation space.

### 3.3 Predictive screening models

| Model | Architecture | Input | Output | Purpose |
|-------|-------------|-------|--------|---------|
| **Classifier** (×15 species) | Chemprop D-MPNN + Morgan count fingerprints | SMILES (with terminal mods) | P(active), threshold MIC ≤32 µg/mL | Binary activity filter |
| **Regressor** (×15 species) | Chemprop D-MPNN + Morgan count | SMILES | log(MIC+1) | Potency ranking |
| **Toxicity** | Chemprop D-MPNN | SMILES | P(toxic), threshold >50% hemolysis/cytotoxicity at 100 µg/mL | Safety filter |

Best classifier variant: **Chemprop + Morgan count fingerprints** (AUC >0.80 for 9/15 species on held-out test).

Regression: Pearson r up to 0.55–0.65 on held-out test (species-dependent).

### 3.4 Terminal modification encoding

Peptides encoded as **SMILES** (not raw sequence) to represent:

| State | N-terminus | C-terminus | Example |
|-------|-----------|-----------|---------|
| Free/Free | Free | Free acid | Standard peptide |
| Acetyl/Free | N-acetyl | Free acid | Increased stability |
| Free/Amide | Free | C-amidated | Common therapeutic form |
| Acetyl/Amide | N-acetyl | C-amidated | Maximum stability |

This allows predictive models to learn modification–activity relationships directly.

### 3.5 Training data scale

| Dataset | Size | Source |
|---------|------|--------|
| Generative training | 32,095 AMPs (5–30 aa) | APD, dbAMP, DRAMP, dbaasp, CAMP, LAMP, + 24 classifier training sets |
| Activity pairs | 37,119 peptide–species pairs (15 species) | dbaasp |
| Toxicity | 8,834 entries | dbaasp (hemolysis/cytotoxicity) |

---

## 4. Key results

### 4.1 Generative quality

- t-SNE of ESM embeddings: generated peptides cluster tightly with known AMPs, separate from AMP-dissimilar controls.
- Amino acid composition: enriched K, L, R; depleted D, E (canonical AMP signature).
- Physicochemical properties: higher net charge and pI vs controls; hydrophobicity profiles match natural AMPs.
- **Benchmark vs 11 generative baselines** (AMPdesigner, ampdiffusion, ampgan, PepDiffusion, etc.): Termini retains the largest fraction of candidates through multi-step filtering and CD-HIT deduplication at 70% identity.

### 4.2 In vitro validation (120 peptides, 11 species)

| Metric | Result |
|--------|--------|
| Active (any species) | **111/120 (92.5%)** |
| MIC ≤16 µmol/L (any species) | 89/120 |
| MIC ≤32 µmol/L (any species) | 100/120 |
| Broad-spectrum (Gram+ and Gram−) | 63/120 |
| Predictive filter precision (best species) | Up to 0.81 (L. monocytogenes) |
| Hit rate vs random screening | ~50–90× enrichment (random: 1–2%) |

**Top broad-spectrum candidates:**
- Peptide 9644: `KFKAKLLKKFFKQFKKFL` (Free/Free) — median MIC 4 µmol/L across 11 organisms.
- Peptide 7375: `RRRIKWRRGAIRPAVIRLVKSV` (Free/Amide) — multiple MICs ≤16 µmol/L.

### 4.3 Terminal modification effects

- C-terminal amidation frequently **lowers MICs** across shared test panels (sequence-dependent).
- Model-predicted directional changes upon modification concordant with experimental MIC shifts in many cases.
- Matched backbone pairs (Free/Free vs Free/Amide) enable direct quantification of modification benefit.

### 4.4 Toxicity filtering

- Toxicity classifier effectively enriches non-hemolytic candidates.
- Most synthesized peptides showed minimal cytotoxicity at tested concentrations.
- HC50 values varied; toxicity and hemolysis related but not redundant.

### 4.5 In vivo (murine skin infection, *A. baumannii*)

| Peptide | Sequence | Termini | Day 2 bacterial reduction |
|---------|----------|---------|--------------------------|
| 8032 | KKWKKFFKAAKKFAKKIG | Free/Free | Significant (p=0.0066) |
| 8034 | KKWKKFFKAAKKFAKKIG | Free/Amide | Significant (p=0.0067) |
| 4194 | WQWRVRLRINKVLPGR | Free/Amide | Significant (p=0.0121) |
| Polymyxin B (control) | — | — | Significant (p=0.0111) |

- 8032 vs 8034 (matched pair): C-terminal amidation does **not** compromise in vivo efficacy.
- Effect transient by day 4 (partial infection resolution under dosing regimen).

### 4.6 Structural/mechanistic diversity

- CD spectroscopy: peptides span helical, β-enriched, mixed, and disordered structures — no convergence to α-helicity.
- Membrane assays (NPN uptake, DiSC3-5 depolarization): multiple interaction phenotypes, not a single mechanism.

---

## 5. Comparison with your thesis (OxML Bio)

| Aspect | Termini (this paper) | Your thesis |
|--------|---------------------|-------------|
| **Goal** | Antibacterial AMP discovery | BBB-compatible GSK3β cyclic peptide modulators |
| **Generator** | ESM-2 latent diffusion (sequence) | BoltzGen all-atom diffusion (3D structure) |
| **Classifier** | Chemprop D-MPNN on SMILES (15 species) | ESM-2 + tabular MLP ($p_{BBB}^{cal}$) |
| **Conditioning** | Net charge + hydrophobic ratio (classifier-free guidance) | Hotspots + ATP avoidance (SDE) + TD3B rewards |
| **Modifications** | N-acetyl, C-amide (first-class design variable) | Cyclization, phosphomimetics |
| **Filtering** | Classify → regress MIC → toxicity → synthesize | 5-gate cascade + Pareto |
| **Length** | 5–30 aa | 5–50 aa (dataset); cyclic peptides ~10–14 aa |
| **Validation** | 120 peptides, 11 bacteria, in vivo mice | Pre-experimental in silico; MD planned |
| **Hit rate** | 92.5% in vitro | Pilot: ipTM>0.75, $p_{BBB}$>0.7 |

### 5.1 Strong parallels

```
TERMINI                           YOUR THESIS
───────                           ───────────
ESM-2 embeddings                  ESM-2 embeddings (BBB classifier branch)
Physicochemical conditioning      24 tabular descriptors + TD3B reward
Multi-stage ML filtering          5-gate cascade (G1–G5)
Terminal chemistry as design var    Cyclic constraints + phosphomimetics
Classifier-free guidance concept  TD3B amortization (non-diff. rewards)
```

Both pipelines follow the same architecture pattern:

> **Generative model → learned classifier/regressor → multi-stage filter → experimental validation**

### 5.2 Key differences

| | Termini | Your thesis |
|---|---------|-------------|
| Representation | Sequence/SMILES (1D) | All-atom 3D coordinates |
| Property optimized | Antibacterial MIC | BBB permeability + substrate binding |
| Reward timing | Post-generation filtering | TD3B fine-tune + post-hoc gates |
| Structure awareness | None (sequence only) | Full 3D geometry (BoltzGen) |
| Target specificity | 15 bacterial species | Single kinase (GSK3β) with selectivity constraints |

---

## 6. How to apply it to your thesis

### 6.1 Direct applications (high priority)

#### A) ESM-2 latent diffusion as a sequence-prototyping layer

Termini validates that **ESM-2 latent diffusion + physicochemical conditioning** produces biologically plausible peptides. You could add a lightweight sequence-generation step **before** BoltzGen:

```
ESM-2 diffusion (conditioned on net charge, hydrophobic ratio)
  → BBB classifier filter ($p_{BBB}^{cal} \ge 0.6$)
  → BoltzGen structure generation for top sequences
```

This reduces BoltzGen compute by pre-filtering sequences in embedding space.

#### B) Classifier-free guidance for BBB properties

Termini conditions generation on net charge and hydrophobic ratio with conditional dropout. Mirror this for BBB:

- Condition ESM-2 diffusion on **target physicochemical properties** from Cavaco et al. BBBpS hallmarks (charge ≈+2, hydrophobic ~35%).
- Use classifier-free guidance to steer toward BBB-compatible property regions.

#### C) Terminal modifications as first-class design variables

Termini's key insight: **don't treat chemical modifications as post-hoc optimization**.

For your cyclic GSK3β peptides:

| Modification | Termini analogue | Your design |
|-------------|-----------------|-------------|
| C-terminal amidation | Standard in Termini | Already in BoltzGen `peptide-anything` (amide) |
| N-terminal acetylation | Improves stability | Consider for BBB peptides |
| Cyclization | Not in Termini | Your primary constraint (disulfide/backbone) |
| Phosphomimetics | Not in Termini | Your substrate-selectivity feature |

Encode modifications in SMILES or extended sequence tokens so your BBB classifier can learn modification–permeability relationships.

#### D) Cite as precedent for ESM-2 + property-conditioned generation

For the OxML poster:

> *"Following Termini (Xu et al., 2026), we combine ESM-2 representations with physicochemical property conditioning for peptide design, extending this paradigm from antimicrobial activity to blood–brain barrier permeability and 3D structure generation via BoltzGen."*

### 6.2 Medium-effort applications

#### E) Adopt Termini's multi-tier filtering structure

Map Termini's funnel to your gates:

| Termini stage | Your equivalent | Enhancement |
|---------------|----------------|-------------|
| Species classifier (15 models) | BBB classifier | Already have |
| MIC regressor | — | Add binding affinity regressor (TD3B $g_\psi$) |
| Toxicity classifier | G5 liabilities + developability | Add hemolysis/cytotoxicity predictor |
| CD-HIT dedup (70%) | Dataset dedup (90%) | Apply at candidate stage too |
| Modification-aware SMILES | Cyclic/peptide encoding | Extend feature pipeline |

#### F) Chemprop as complementary predictor

Termini uses Chemprop (graph neural network on SMILES) for activity. Your thesis uses ESM-2 + tabular MLP. Consider a **third branch**:

```
ESM-2 branch  +  Tabular branch  +  Chemprop/SMILES branch → fusion
```

Chemprop naturally handles terminal modifications and could improve BBB prediction for modified/cyclic peptides.

#### G) Hit-rate benchmarking framework

Termini reports 92.5% hit rate vs ~1–2% random baseline. Define analogous metrics for your pipeline:

| Metric | Definition |
|--------|-----------|
| Gate pass rate | % candidates surviving G1→G5 |
| BBB enrichment | Mean $p_{BBB}^{cal}$ of survivors vs random peptides |
| Compute efficiency | Active candidates per GPU-hour |

### 6.3 Concepts to borrow, not replicate

| Termini feature | Applicability to your thesis |
|-----------------|---------------------------|
| 15-species antibacterial panel | Low — different therapeutic target |
| Murine infection model | Future experimental stage, not current |
| AMP training data (32k) | Different label space (BBB+/BBB−) |
| Sequence-only generation | Complementary to, not replacement for, BoltzGen |

---

## 7. Three-paper synthesis for your thesis

Your `docs/references/papers/` folder now covers three complementary paradigms:

| Paper | Paradigm | Role in your pipeline |
|-------|----------|----------------------|
| **Cavaco et al. (2024)** | Rule-based BBB physicochemical determinants | Feature validation + interpretable thresholds |
| **Termini (2026)** | ESM-2 latent diffusion + ML filtering | Sequence generation + property conditioning pattern |
| **Proteína-Complexa (2026)** | 3D generative + test-time compute | Structure generation + inference search pattern |

**Unified architecture for your thesis:**

```
Cavaco rules ──→ inform tabular features (24 descriptors)
Termini pattern ──→ ESM-2 + property conditioning + multi-tier filter
Complexa pattern ──→ BoltzGen 3D generation + TD3B/Best-of-N optimization

Combined:
  ESM-2 latent prior (Termini) +
  BBB classifier (Cavaco-informed features) +
  BoltzGen all-atom generation (Complexa-like) +
  TD3B reward amortization (your extension) +
  5-gate cascade (Termini-like funnel)
```

---

## 8. Conclusions for your thesis

1. **Termini validates your ESM-2-centric approach** — The best peptide ML pipelines use ESM-2 embeddings as the generative and predictive backbone. Your BBB classifier's ESM-2 branch follows state-of-the-art practice.

2. **Property conditioning works** — Conditioning diffusion on physicochemical properties (charge, hydrophobicity) with classifier-free guidance is proven effective. Directly applicable to conditioning on BBB-relevant properties from Cavaco et al.

3. **Modifications must be design variables** — Terminal chemistry (acetyl, amide) and, by extension, cyclization and phosphomimetics should be encoded in the predictive models, not applied post-hoc.

4. **Multi-tier filtering is essential** — Termini's 92.5% hit rate comes from aggressive computational pre-screening before synthesis. Your 5-gate cascade serves the same function; quantifying pass rates at each gate strengthens the thesis narrative.

5. **Sequence and structure generation are complementary** — Termini generates sequences; BoltzGen generates 3D structures. A two-stage pipeline (ESM-2 diffusion → BoltzGen) could improve compute efficiency and BBB enrichment.

6. **Most actionable next step** — Implement property-conditioned ESM-2 sampling (charge, hydrophobic ratio targets from Cavaco BBBpS hallmarks) as a cheap pre-filter before BoltzGen campaigns.

---

## 9. Resources

- Preprint: https://doi.org/10.64898/2026.04.09.717597
- Your BBB classifier: `packages/bbb_models/src/bbb_classifier/`
- Your feature pipeline: `packages/dataset/src/bbb_dataset/features.py`
- Your poster: `entregas/oxml_bio/main.tex`
- Related summaries:
  - `docs/references/papers/bbb-transport-meta-analysis-2024.md` (Cavaco — BBB determinants)
  - `docs/references/papers/proteina-complexa-iclr-2026.md` (Complexa — 3D generative + test-time compute)
