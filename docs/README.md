# Documentation

Project documentation for *in silico* design of BBB-compatible cyclic phosphomimetic GSK3β modulators.

**Start here:** [architecture/overview.md](architecture/overview.md)
**For AI agents:** [architecture/agent-context.md](architecture/agent-context.md)

## Architecture and context

| Document | Contents |
|----------|----------|
| [architecture/overview.md](architecture/overview.md) | Project goal, pipeline phases, design constraints (non-differentiable ESM, TD3B) |
| [architecture/agent-context.md](architecture/agent-context.md) | Full project context for agents and contributors |
| [architecture/theoretical-framework.md](architecture/theoretical-framework.md) | Formulas: SDE guidance, TD3B, filtering gates |
| [architecture/reproducibility.md](architecture/reproducibility.md) | Reproducibility roadmap (v1 current, v2 planned) |
| [design/rl-md-strategy.md](design/rl-md-strategy.md) | Operational plan for closing the TD3B + MD loop |

## Data pipeline

| Document | Contents |
|----------|----------|
| [data/dataset-pipeline.md](data/dataset-pipeline.md) | `bbb-dataset-build` pipeline + Hugging Face export (`hf_release`) |
| [data/dataset-cleaning.md](data/dataset-cleaning.md) | Cleaning, deduplication, cluster-aware splits |
| [data/data-augmentation.md](data/data-augmentation.md) | Sequence augmentation, mixup, weak labels |
| [data/boltz-folding.md](data/boltz-folding.md) | Offline peptide folding via Boltz API |

## BBB models

| Document | Contents |
|----------|----------|
| [models/bbb-classifier.md](models/bbb-classifier.md) | Tabular/ESM classifier, experiments exp01–exp06 |
| [models/structural-classifier.md](models/structural-classifier.md) | Geometric EGNN `struct_egnn_geo` (exp09), stability, outputs |
| [models/structural-bbb-guidance.md](models/structural-bbb-guidance.md) | Differentiable BBB guidance in diffusion |

## Infrastructure

| Document | Contents |
|----------|----------|
| [infrastructure/vast-training.md](infrastructure/vast-training.md) | Remote `bbb_models` training on Vast.ai (upload, train, sync) |
| [../infra/vast/README.md](../infra/vast/README.md) | Vast script index (`infra/vast/bbb_models/`, `boltzgen_design/`) |

## Writing

| Document | Contents |
|----------|----------|
| [writing/paper-writing-guide.md](writing/paper-writing-guide.md) | Guide for drafting the thesis/paper |
| [writing/experimental-validation.tex](writing/experimental-validation.tex) | LaTeX draft for experimental validation |

## References

Paper summaries in [references/papers/](references/papers/) (PDFs excluded from git).
