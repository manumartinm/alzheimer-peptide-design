# State of the Art (SOTA) — GSK3β Modulators & Peptide Inhibitors in Alzheimer's Disease (2024–2026)

## 1. General context (GSK3β in Alzheimer's)

**GSK3β** (glycogen synthase kinase 3 beta) is a crucial therapeutic target in Alzheimer's disease (AD) due to its central role in **Tau hyperphosphorylation** and amyloid-beta (Aβ) pathology. The historical problem with conventional ATP-competitive inhibitors is severe side effects from blocking essential maintenance pathways (such as Wnt signaling). Current SOTA therefore focuses on **substrate-selective modulation** via peptides.

## 2. Recent advances (downloaded papers)

### A. Rational design of kappa-casein-derived peptides

* **Papers:** *Targeting GSK-3β with Peptide Inhibitors: A Rational Computational Strategy for Alzheimer's Disease Intervention* (bioRxiv, December 2024) and *Rational design of k-casein peptides to modulate GSK-3B dynamics for Alzheimer's therapy* (Scientific Reports, March 2026).
* **Approach:** Computational tools (molecular dynamics simulations and HADDOCK docking) to design peptides based on kappa-casein.
* **Key results:**
  * Identified peptides **MP31 (HPDFVAPFPE)** and **PEP8 / PEP44** as top candidates.
  * These peptides interact directly with the ATP-binding pocket and catalytic sites (Asp200), stabilizing the kinase and reducing structural flexibility. This stabilization blocks substrate access.
  * Mutational optimization greatly improved affinity, showing that MM/PBSA-guided design can optimize intermolecular interaction networks.

### B. Akt-activated inhibitors

* **Paper:** *Akt-activated GSK3β inhibitory peptide effectively blocks tau hyperphosphorylation* (PubMed, 2024–2025).
* **Approach:** Developed a GSK3β inhibitory peptide (GIP) **specifically activated by Akt kinase**. Combines the PPPSPxS motif (from LRP6 co-receptor) that directly inhibits GSK3β with the Akt target sequence (RxRxxS).
* **Key results:**
  * In vivo models (3×Tg-AD mice), intravenous administration significantly reduced Tau phosphorylation in the hippocampus and improved memory deficits.
  * Designed for cell permeability, showing neuroprotection without interfering with basal GSK3β functions.

## 3. Comparison with this project (cyclic modulators + BBB permeability)

Current SOTA emphasizes **substrate-specific** peptides with **ability to cross biological barriers** (cell permeability/BBB). This project is **fully aligned** and adds substantial innovations over this SOTA:

1. **Cyclic vs linear structure:** Recent papers (MP31, PEP8) rely mainly on linear peptides. This project proposes **cyclic peptides**, offering greater proteolytic stability in vivo — a crucial step for real drugs.
2. **All-atom generative diffusion (BoltzGen) vs mutagenesis/docking:** While SOTA uses virtual screening and MD on limited mutation libraries (e.g. 48 variants), this project uses pretrained generative AI to explore chemical space efficiently.
3. **Explicit BBB constraint:** The GIP peptide reached the brain empirically in vivo (fused to a penetrating sequence). This project integrates a **BBB classifier into the generative loop (via TD3B)**, making blood–brain barrier permeability a *by-design* constraint rather than a post-hoc patch.

## 4. Downloaded files

Corresponding PDFs are stored outside git. Summary notes live in `docs/references/papers/`.

- `Targeting_GSK3b_BioRxiv.pdf`
- `SciRep_Rational_Design.pdf`
