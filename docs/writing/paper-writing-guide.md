# Paper Writing Guide (ML + Computational Biology)

This guide defines how to write a clear, technically rigorous paper for this project and similar ML/Bio pipelines.

It is designed for:
- human authors preparing manuscripts;
- AI agents generating drafts, revisions, and response documents.

## 1. Writing Objectives

A good paper must simultaneously:

1. Explain the scientific problem and why it matters.
2. Make method decisions reproducible and falsifiable.
3. Report results with calibrated uncertainty and clear limitations.
4. Avoid overclaiming beyond computational evidence.

## 2. Recommended Paper Structure

Use this canonical structure:

1. **Title**
2. **Abstract**
3. **Introduction**
4. **Related Work**
5. **Methods**
6. **Experimental Setup**
7. **Results**
8. **Ablations / Error Analysis**
9. **Discussion**
10. **Limitations and Ethics**
11. **Conclusion**
12. **Reproducibility Appendix**

## 3. Section-by-Section Checklist

### 3.1 Title

Should include:
- method family (e.g., classifier-guided diffusion),
- application domain (BBB peptides / GSK3β),
- scope qualifier if needed (`in silico`).

Avoid:
- causal or efficacy claims not experimentally validated.

### 3.2 Abstract

Must include:
- motivation,
- method summary,
- key quantitative outcomes,
- explicit scope boundary (`pre-experimental`, `computational`).

Target:
- 150 to 250 words for most conference submissions unless template says otherwise.

### 3.3 Introduction

Include:
- disease/biological relevance;
- methodological gap in prior work;
- your hypothesis and contributions in bullet form.

Best practice:
- end with a concise “Contributions” list (3 to 5 bullets).

### 3.4 Related Work

Group by topic, not by citation chronology:

- BBB peptide predictors;
- protein language models;
- diffusion-based molecular/protein design;
- reward-guided or RL fine-tuning methods.

For each group:
- one-sentence strength;
- one-sentence limitation;
- one-sentence relation to your method.

### 3.5 Methods

Should let a trained reader rebuild your approach.

Minimum required:
- data schema and splits;
- feature definitions and encoders;
- model architectures and dimensions;
- objective functions and calibration strategy;
- training algorithm and hyperparameters.

For this project specifically:
- clearly separate geometric guidance in reverse SDE from BBB reward integration via TD3B.

### 3.6 Experimental Setup

Include:
- hardware and runtime details (GPU, precision, wall-time if relevant);
- software stack versions;
- random seed policy;
- evaluation protocols and threshold definitions.

Always include:
- baseline models and why they are fair baselines.

### 3.7 Results

Report:
- core metrics (PR-AUC, ROC-AUC, MCC, Brier for BBB classifier);
- calibration behavior (before vs after isotonic);
- candidate throughput and filtering attrition for generation pipeline.

Use tables/figures that answer one question each.

### 3.8 Ablations and Error Analysis

Essential for credibility:
- remove one module at a time;
- report effect size with confidence intervals when possible;
- include failure modes (false positives, unstable generations, edge-case sequences).

### 3.9 Discussion

Interpret what results mean for:
- mechanism plausibility;
- practical prioritization for wet-lab follow-up;
- transferability to other kinases/targets.

### 3.10 Limitations and Ethics

Mandatory statements:
- predictions are computational;
- no direct therapeutic efficacy claim;
- known biases/data coverage limitations;
- potential misuse boundaries.

## 4. Scientific Style Rules

### Use precise language

Prefer:
- “predicts”, “suggests”, “prioritizes”.

Avoid:
- “proves”, “demonstrates cure”, “clinically effective”.

### Keep claims coupled to evidence

Every quantitative claim must map to:
- metric name;
- dataset/split;
- evaluation protocol.

### Be explicit about uncertainty

Where possible include:
- variance across folds/runs;
- confidence intervals or bootstrapped bounds.

## 5. Figures and Tables

Each figure should have:
- one conceptual focus;
- readable axis labels and units;
- caption that states what is being concluded.

For this project, useful figures are:
- end-to-end 4-stage workflow;
- BBB classifier architecture;
- diffusion guidance + TD3B flow;
- filtering funnel with candidate counts.

## 6. Reproducibility Block (Required)

Include a dedicated section with:

- code location and commit reference;
- environment setup instructions;
- data preparation commands;
- training/evaluation commands;
- artifact naming conventions.

If using DVC:
- list pipeline stages and expected outputs.

## 7. Common Failure Modes in ML/Bio Papers

Avoid these:

1. Mixing calibrated and uncalibrated probabilities without saying so.
2. Reporting only one metric for imbalanced tasks.
3. Hiding threshold choices.
4. Confusing training and inference guidance mechanisms.
5. Claiming biological efficacy from computational ranking only.

## 8. Agent Writing SOP

If you are an AI agent writing manuscript text:

1. Start from evidence tables, not from narrative claims.
2. Add one citation-backed statement at a time.
3. Mark any missing evidence with TODO placeholders instead of inventing values.
4. Ensure equations and implementation descriptions are consistent.
5. Always include a “Limitations” paragraph in any results-heavy section.

## 9. Suggested Internal Review Workflow

Before submission:

1. **Technical pass**: verify equations, metric definitions, and config consistency.
2. **Claim pass**: remove overstatements unsupported by current evidence.
3. **Reproducibility pass**: execute documented commands from clean environment.
4. **Readability pass**: ensure each section answers one clear question.

## 10. Publication-Ready Checklist

Final checklist:

- [ ] Abstract includes motivation, method, key result, and scope boundary.
- [ ] Methods are sufficient for reproduction.
- [ ] Metrics are calibrated, contextualized, and split-aware.
- [ ] Limitations explicitly state computational-only status.
- [ ] Figures are readable and captions are conclusion-oriented.
- [ ] Code/data availability statement is present.
