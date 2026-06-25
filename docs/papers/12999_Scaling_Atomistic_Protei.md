# Proteína-Complexa: Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time Compute

**PDF:** `12999_Scaling_Atomistic_Protei.pdf`  
**Venue:** ICLR 2026 (OpenReview submission #12999)  
**Authors:** Kieran Didi, Zuobai Zhang, Guoqing Zhou, Danny Reidenbach, Zhonglin Cao, Sooyoung Cha, et al. (NVIDIA, Oxford, Mila, SNU)  
**Project page:** https://research.nvidia.com/labs/genair/proteina-complexa/  
**Code:** https://github.com/NVIDIA-BioNeMo/Proteina-Complexa

---

## One-sentence summary

Proteína-Complexa unifies **atomistic generative models** (flow matching) with **inference-time optimization** (beam search, MCTS, etc.) to design protein binders without ProteinMPNN, outperforming both purely generative methods (RFdiffusion) and hallucination approaches (BindCraft, BoltzDesign) under a normalized compute budget.

---

## 1. Problem it solves

Protein binder design has historically been split into two seemingly incompatible paradigms:

| Paradigm | Examples | Strength | Weakness |
|----------|----------|----------|----------|
| **Generative** | RFdiffusion, BoltzGen, APM | Learned prior, diversity | No fine-grained interface optimization |
| **Hallucination** | BindCraft, BoltzDesign, AlphaDesign | Optimizes AF2/Boltz scores | No generative prior; expensive and ad-hoc |

Complexa argues this is a **false dichotomy** — analogous to LLMs and image models, where a pretrained base model plus adaptive inference-time compute beats either approach alone.

---

## 2. Key advances (contributions)

1. **Teddymer** — New synthetic dataset of ~3.5M protein–protein dimer clusters, built from domain–domain interactions in the AlphaFold DB (TED annotations). Scales far beyond filtered PDB multimers (~46k).

2. **Latent target conditioning** — Novel mechanism to condition the generator on the target **without noise**: the target stays clean while only the binder is generated via flow matching. Includes hotspot tokens at interface residues.

3. **Translation noise** — Global translation noise on binder Cα during training to force correct interface positioning (critical for binders, irrelevant for monomers).

4. **Test-time compute scaling** — First systematic application of search algorithms during denoising in protein design: Best-of-N, Beam Search, Feynman–Kac Steering (FKS), MCTS.

5. **All-atom generation without inverse folding** — Sequence and atomistic structure co-generated; no post-hoc ProteinMPNN/LigandMPNN required.

6. **Flexible rewards** — ipAE from structure predictors + Rosetta hydrogen-bond energies; need not be differentiable.

7. **Extensibility** — Demonstrated on protein targets, small molecules, and an enzyme design benchmark (AME, 41 tasks).

---

## 3. Methodology

### 3.1 Base architecture: extended La-Proteína

Complexa builds on **La-Proteína** (flow matching with a partially latent representation):

```
Full protein
├── Explicit Cα        → x^Cα  (backbone coordinates)
└── Per-residue latent → z     (sequence + non-Cα atoms, 8 dims)

Components:
  Encoder E  : protein → z
  Decoder D  : (x^Cα, z) → Atom37 structure + sequence
  Denoiser v^φ : flow matching over (x^Cα, z)
```

- **Rectified flow matching:** linear interpolant $x_t = t x_1 + (1-t) x_0$, target $x_1 - x_0$.
- **Separate schedules** for Cα (exponential, fast) and z (quadratic, slow) at inference.
- Transformer-only with pair-biased attention — no AlphaFold triangular layers → **fast**.

### 3.2 Target conditioning during denoising

> **Note:** This is not "target denoising." The target is **not corrupted**. Only the binder goes through the generative process.

```
Denoiser input:
  [noisy_binder (x^Cα_t, z_t)  ;  clean_target (c^target)]

c^target = Atom37 coords + AA identity + hotspot tokens (binary)

Pair representations: computed jointly over binder + target
Encoder/decoder: FROZEN during conditional denoiser training
```

**Translation noise** on binder Cα: $\vec{d} \sim \mathcal{N}(0, c_d^2 I)$ with $c_d = 0.2$ nm, broadcast over N residues.

### 3.3 Teddymer: dataset construction

```
AFDB50 (47M structures with TED annotation)
  → split multi-domain monomers into separate chains
  → extract dimers with spatial proximity (≥4 residues within ≤10 Å)
  → filter for complete CATH annotations
  → 10M dimers → cluster → 3.5M Teddymer clusters
  → filter: interface pLDDT > 70, ipAE < 10, interface length > 10
  → 510k cluster representatives for training
```

### 3.4 Staged training

| Stage | Data | Goal |
|-------|------|------|
| 1 | AFDB monomers | Autoencoder (VAE) |
| 2 | PDB | Fine-tune autoencoder |
| 3 | AFDB Foldseek clusters | Pretrain flow model (monomers) |
| 4a | Teddymer + PDB multimers | Binder design (protein) |
| 4b | PLINDER + AFDB (LoRA) | Binder design (small molecule) |

### 3.5 In-silico success metrics

**Protein targets** (AlphaFold2-Multimer via ColabDesign):

- ipAE < 7.0 Å
- complex_pLDDT > 0.9
- binder_scRMSD < 1.5 Å

**Small-molecule targets** (RosettaFold-3):

- min-ipAE < 2
- binder_scRMSD < 2 Å
- ligand_scRMSD < 5 Å

### 3.6 Inference-time optimization

Rewards **do not require gradients**. They are evaluated after full rollout → decode → fold with a structure predictor.

#### Beam search (most effective on hard targets)

```
Maintain N parallel denoising trajectories (beam)
Every K steps:
  1. Branch L children per trajectory → N×L candidates
  2. Full stochastic rollout of each candidate
  3. Decode → fold → compute reward R (ipAE, H-bond...)
  4. Select top-N → new beam
Repeat until clean sample
```

#### Other algorithms

| Algorithm | Idea | When to use |
|-----------|------|-------------|
| **Best-of-N** | Generate N samples, filter by reward | Easy targets |
| **FKS** | Importance sampling over tilted distribution $p \cdot e^{\beta R}$ | Medium targets |
| **MCTS** | Trajectory tree with exploration/exploitation balance | Very hard targets |
| **G&H** | Initialize BindCraft from a generative sample | Speeds up easy targets |

---

## 4. Key results

### 4.1 Base generative model (no inference-time optimization)

**Table 1 — Small-molecule targets** (200 samples/target, length 100):

| Model | SAM | OQO | FAD | IAI | Time [s] ↓ | Novelty |
|-------|-----|-----|-----|-----|------------|---------|
| RFdiffusion-AllAtom | 2 | 3 | 5 | 8 | 87.4 | 0.72 |
| **Complexa** | **10** | **6** | **17** | **19** | **13.5** | 0.71 |

**Table 2 — Protein targets** (200 binders/target, 40–250 residues; mean unique successes):

| Model | Self | MPNN-FI | MPNN | Time [s] | Best method (of 19 targets) |
|-------|------|---------|------|----------|-----------------------------|
| RFdiffusion | — | — | 4.68 | 70.8 | 3 |
| Protpardelle-1c | — | — | 0.73 | 8.13 | 0 |
| APM | 0.31 | 1.52 | 3.15 | 73.1 | 1 |
| **Complexa** | **9.10** | **13.6** | **14.4** | **15.6** | **14** |

- Even Complexa's **self-generated** sequences outperform all MPNN baselines.
- ~5× faster than RFdiffusion; ~4× faster than APM.

### 4.2 Test-time compute scaling

- **Easy targets** (12): Best-of-N already beats BindCraft/BoltzDesign/AlphaDesign.
- **Hard targets** (7): Beam Search, FKS, and MCTS required; hallucination baselines fail at matched compute.
- **VEGFA** (2-chain, hard): Complexa methods dominate clearly (Fig. 8).
- **TNF-α, H1, IL17A** (multi-chain, very hard): no baseline succeeds in <32 GPU-h; Complexa finds 15/7/1 unique successes with >100 GPU-h.

**Table 3 — Beam search with different rewards** (averaged over targets):

| Configuration | Unique successes ↑ | Interface H-bonds (mean) ↑ |
|---------------|-------------------|------------------------------|
| No reward | 77.00 | 5.271 |
| + $f_{ipAE}$ | 83.36 | 5.524 |
| + $f_{H-Bond}$ | 82.36 | 7.154 |
| + both | **86.26** | 6.518 |

### 4.3 Enzyme design (AME benchmark, 41 tasks)

Complexa outperforms RFdiffusion2 on **38/41 tasks**, with self-generated sequences and with LigandMPNN.

### 4.4 Ablations

- Without **Teddymer** → performance drops sharply (PDB alone is insufficient).
- Without **translation noise** → poor binder placement at the interface.

---

## 5. Comparison with your thesis (OxML Bio)

Your pipeline (`TFG/entregas/oxml_bio/main.tex`):

```
BBB dataset (523 peptides)
  → BBB classifier (ESM-2 + tabular, p_BBB^cal)
  → BoltzGen (cyclic peptides vs GSK3β)
  → TD3B fine-tune (BBB + affinity + direction oracle rewards)
  → 5-gate cascade → MD → assay
```

### 5.1 Conceptual similarities

| Concept | Complexa | Your thesis |
|---------|----------|-------------|
| Generative prior | La-Proteína flow model | BoltzGen VP-SDE |
| Fixed target during generation | Clean embeddings in denoiser | GSK3β CIF + `binding_types` |
| Hotspots | Tokens at interface residues | R96/R180/K205 + F67/Q89/N95 |
| Non-differentiable rewards | Rollout + scoring at inference | TD3B WDCE amortization |
| Post-hoc filtering | Unique-success clustering | 5 gates + Pareto |
| All-atom | Yes, no MPNN | Yes (BoltzGen `peptide-anything`) |

### 5.2 Key differences

| Aspect | Complexa | Your thesis |
|--------|----------|-------------|
| **Designed ligand** | Protein 40–250 aa | Cyclic peptide 5–50 aa |
| **Pharmacokinetic constraint** | None | **BBB permeability** (unique) |
| **When rewards are optimized** | During denoising (inference) | Before (TD3B fine-tune) + after (filters) |
| **Primary reward** | ipAE / H-bond (structure) | $p_{BBB}^{cal}$ (sequence/pharmacokinetics) |
| **Selectivity** | Generic hotspots | Substrate groove vs ATP cleft |
| **Differentiable guidance** | Does not use Dhariwal–Nichol | Yes: $\nabla \log p_h$, $-\nabla \log p_a$ in SDE |

### 5.3 Architecture diagram

```
COMPLEXA                          YOUR THESIS
────────                          ───────────
Teddymer pretrain                 BBB dataset (523 seq)
       ↓                                 ↓
Flow + target cond.               ESM-2 + tabular classifier
       ↓                                 ↓
Beam/MCTS at inference     vs.    TD3B fine-tune (offline)
  (reward: ipAE, H-bond)          (reward: BBB, affinity, direction)
       ↓                                 ↓
Binder without MPNN               BoltzGen + geometric guidance
       ↓                                 ↓
In-silico success filter          5 gates + Pareto → MD
```

---

## 6. How to apply it to your thesis

### 6.1 Direct applications (high priority)

#### A) Best-of-N with a composite reward

After TD3B fine-tuning, generate N≈1000–5000 candidates with BoltzGen and rank by:

$$R_{\text{total}} = w_1 \cdot p_{\text{BBB}}^{\text{cal}} + w_2 \cdot M_{\text{hot}}^{\text{soft}} - w_3 \cdot U_{\text{ATP}} + w_4 \cdot \text{ipTM}$$

You already have the infrastructure in `bbb_classifier/scripts/predict.py` and `infer/rank.py`. This is the direct equivalent of Complexa's Best-of-N, using your BBB classifier as the primary reward.

#### B) Document the paradigm in the OxML poster

Add a positioning sentence to `main.tex`:

> *Following the generative + test-time compute paradigm of Proteína-Complexa (Didi et al., ICLR 2026), we combine a pretrained diffusion prior (BoltzGen) with amortized reward optimization (TD3B) and post-hoc multi-gate filtering for BBB-compatible cyclic peptide design.*

#### C) H-bond reward at the substrate groove

Complexa shows that optimizing hydrogen bonds improves interfaces. For GSK3β:

- Detect H-bonds between peptide and hotspots R96/R180/K205 (HBPlus).
- Add $f_{H\text{-Bond}}$ to the post-hoc ranking reward or the TD3B reward.
- Consistent with L803-mts substrate-recognition logic.

### 6.2 Medium-effort applications

#### D) Hybrid TD3B + inference search pipeline

```
Offline:  TD3B fine-tune → tilted prior p*_θ
Online:   Best-of-N / beam-lite over p*_θ with R_total
Post:     5-gate cascade (as now)
```

TD3B amortizes the BBB reward cost offline; inference search refines geometry and affinity online. Combines the best of both papers.

#### E) "Lite" beam search without modifying BoltzGen

If BoltzGen does not expose search during the SDE:

1. Generate a batch of 100 designs.
2. Resample top-20 by $R_{\text{total}}$.
3. Re-generate with different seeds or slightly varied guidance weights.
4. Repeat for 3–5 rounds.

#### F) Ablation experiments for the thesis

| Experiment | What it demonstrates |
|------------|---------------------|
| BoltzGen alone vs + TD3B | Value of offline amortization |
| TD3B vs TD3B + Best-of-N | Value of test-time compute |
| Geometric guidance alone vs + BBB reward | Separating geometry from delivery |
| With vs without $f_{H\text{-Bond}}$ | Contribution of interface chemistry |

### 6.3 Out of scope for the thesis

| Complexa idea | Feasibility | Reason |
|---------------|-------------|--------|
| Pretrain on Teddymer-like data | Low | No GSK3β–peptide pairs at scale |
| Retrain your own flow model | Low | BoltzGen is already pretrained |
| MCTS during BoltzGen SDE | Medium-low | Requires access to the internal sampling loop |
| Fold class guidance | Low | Short cyclic peptides, not globular domains |
| Use Complexa directly | Low | Designs proteins, not BBB cyclic peptides |

---

## 7. Cross-references with your bibliography

| Paper | In your poster | Relation to Complexa |
|-------|----------------|---------------------|
| Stark et al. — BoltzGen | Generator | Same class of all-atom model |
| Cao et al. — TD3B | Reward amortization | Alternative/complement to beam search |
| Dhariwal & Nichol | SDE guidance | Complexa does not use it; prefers search |
| Didi et al. — Complexa | *(add)* | Generative + test-time compute paradigm |
| Lin et al. — ESM-2 | BBB classifier | No equivalent in Complexa |

---

## 8. Conclusions for your thesis

1. **Complexa validates your conceptual architecture**: generative prior + reward optimization + filtering is the state-of-the-art paradigm in binder design.

2. **Your differentiating contribution is the BBB constraint** — Complexa does not have it. The BBB classifier is analogous to Complexa's ipAE, but operates in a completely different space (pharmacokinetics vs structural confidence).

3. **The clearest implementable gap**: add **Best-of-N at inference** on top of TD3B, using $p_{BBB}^{cal}$ as the reward. Low risk, high narrative impact.

4. **TD3B and Complexa are complementary**, not competitors:
   - TD3B → learns the prior tilt **offline** (amortized).
   - Complexa → searches within the prior **online** (per generation).

5. Complexa's **target conditioning** (clean target, noisy binder) is conceptually identical to how BoltzGen fixes the target CIF and only diffuses the designed peptide.

---

## 9. Resources

- Paper: https://arxiv.org/html/2603.27950v1
- OpenReview: https://openreview.net/forum?id=qmCpJtFZra
- GitHub: https://github.com/NVIDIA-BioNeMo/Proteina-Complexa
- HuggingFace model: `nvidia/NV-Proteina-Complexa-Protein-Target-160M-v1`
- Your poster: `TFG/entregas/oxml_bio/main.tex`
- Your BBB classifier: `TFG/bbb_classifier/`
