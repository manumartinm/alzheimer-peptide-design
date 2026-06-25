# BoltzGen: Toward Universal Binder Design

**PDF:** `2025.11.20.689494v1.full (1).pdf`
**Venue:** bioRxiv (Preprint posted November 24, 2025)
**Authors:** Hannes Stark, Felix Faltings, MinGyu Choi, et al. (MIT, Boltz, NVIDIA, UCSF, MPI, etc.)
**Code & Models:** https://github.com/HannesStark/boltzgen

---

## One-sentence summary

**BoltzGen** is a universal all-atom generative model that unifies protein design with structure prediction (SOTA-level), capable of generating multiple binder modalities (nanobodies, full proteins, linear and cyclic peptides) against a wide range of targets, achieving a 66% success rate (nanomolar affinities) on completely novel targets with no PDB homologs.

---

## 1. Problem it solves

Computational binder design has had important limitations:

1. **Specialized models:** Often focus on a single modality (e.g. proteins only or peptides only).
2. **Backbone-only generation:** Require an additional inverse-folding model (such as ProteinMPNN), losing side-chain context during interface generation.
3. **Limited hard-target validation:** Most methods are evaluated on easy targets (similar structures in the training set).
4. **Lack of control:** Hard to impose real-world constraints (e.g. disulfide bonds, binding one pocket while avoiding another).

BoltzGen addresses this by operating in a continuous all-atom space, enabling flexible specification and large-scale wet-lab validation on hard targets.

---

## 2. Main contributions

1. **Universal modality:** One model designs nanobodies, proteins, linear peptides, **cyclic peptides** (covalent/disulfide bonds), and small-molecule binders.
2. **Geometric encoding:** Designed residues are represented with exactly **14 atoms**. Amino-acid identity is not predicted discretely; the model stacks "virtual" atoms on the backbone to signal residue type (e.g. 3 atoms on N and 4 on O = threonine). This keeps the problem fully continuous.
3. **Unified prediction + design:** Trained jointly for structure prediction (folding) and design, matching Boltz-2 / AlphaFold3 folding performance.
4. **Large-scale wet-lab validation:** 8 different lab campaigns, 26 targets.
5. **Specification language (YAML):** Define binders, targets, binding hotspots, and covalent constraints simply.

---

## 3. Methodology: the BoltzGen pipeline

The ecosystem is not just the model but a 6-step computational pipeline to filter thousands of designs down to a viable candidate set:

| Step | Action | Description |
|------|--------|-------------|
| **1. Diffusion (GPU)** | **Generation** | BoltzGen generates structures and sequences jointly from YAML (target + constraints). |
| **2. Inverse folding** | **Optional redesign** | Uses BoltzIF (SolubleMPNN-based) to optimize sequence and improve solubility. |
| **3. Folding (GPU)** | **Verification** | Boltz-2 predicts structure of design + target. RMSD vs step-1 structure is computed. For proteins, binder-only folding checks independent foldability. |
| **4. Affinity (GPU)** | **Small molecules** | Uses Boltz-2 affinity module. |
| **5. Analysis (CPU)** | **Physical metrics** | Counts H-bonds, salt bridges, buried surface (SASA), hydrophobic patches, solubility. |
| **6. Filtering (CPU)** | **Quality-diversity (QD)** | Weighted worst-rank algorithm (robust across all metrics) + greedy selection for **diversity** (low sequence/structure similarity among picks). |

---

## 4. Key wet-lab results

Large experimental validation in vitro and in vivo:

* **9 novel targets:** <30% identity to any bound structure in PDB. Testing ≤15 designs per target, **nanomolar affinity on 66% of targets** (proteins and nanobodies).
* **Peptides against RagC GTPase:** 7 confirmed binders from 29 tested (best affinity 3.5 µM).
* **Disulfide cyclic peptides:** Against RagA:RagC dimer, **14 active binders from 24 tested**.
* **Intrinsically disordered proteins (IDPs):** In vivo binding (live cells) to disordered NPM1 region.
* **Antimicrobial peptides (AMPs):** 19.5% of generated designs inhibited bacterial growth (disrupting GyrA–GyrA interaction).

---

## 5. Application to this project (GSK3β)

**BoltzGen is the central generative engine of this project.** This paper validates the choice.

### 5.1 Why BoltzGen and not RFdiffusion?

* **Cyclic peptides:** BoltzGen supports covalent constraints in YAML (`peptide-anything`). RFdiffusion requires heavy hacks and ProteinMPNN for cyclic peptides.
* **Direct all-atom:** GSK3β substrate-pocket interactions need atomic side-chain precision (e.g. R96/R180). BoltzGen co-generates backbone and sequence.

### 5.2 Critical differences vs default BoltzGen pipeline

* Standard BoltzGen (step 6) uses quality-diversity filtering on basic physical metrics (pTM, SASA, hydrophobic patches).
* **Neuropharmacology extension:** BoltzGen has **no pharmacokinetic or brain-penetration optimization**. This project augments filtering with the **BBB classifier (ESM-2)** and uses **TD3B** to inject that knowledge into the model prior, not only at the end.
* **Directional control:** BoltzGen can target hotspots; this project adds the TD3B direction oracle ($d^*=+1$) for **substrate modulator** behavior vs ATP-competitive inhibition.

### 5.3 Narrative for thesis / poster

1. **SOTA validation:** Cite BoltzGen to justify the engine: *"We use BoltzGen (Stark et al., 2025), an all-atom generative model that recently demonstrated 66% success on novel targets and native cyclic peptide design with disulfide bonds."*
2. **The gap:** *"Despite geometric strength, BoltzGen lacks ADME optimization. We introduce amortized reward fine-tuning (TD3B) to couple blood–brain barrier permeability to geometric diffusion."*
