# Documentación

Índice de la documentación del proyecto. Para una visión rápida, empieza por [architecture.md](architecture.md).

## Arquitectura y contexto

| Documento | Contenido |
|-----------|-----------|
| [architecture.md](architecture.md) | Objetivo del TFG, fases del pipeline, restricciones de diseño (ESM no diferenciable, TD3B) |
| [AGENT_CONTEXT.md](AGENT_CONTEXT.md) | Contexto completo para agentes/continuadores del proyecto (inglés) |
| [THEORETICAL_FRAMEWORK.md](THEORETICAL_FRAMEWORK.md) | Fórmulas: SDE guidance, TD3B, gates de filtrado |
| [RL_MD_STRATEGY.md](RL_MD_STRATEGY.md) | Estrategia operativa para cerrar el loop TD3B + Molecular Dynamics |

## Pipeline de datos

| Documento | Contenido |
|-----------|-----------|
| [DATASET_PIPELINE.md](DATASET_PIPELINE.md) | Pipeline `tfg-bbb-build` + export Hugging Face (`hf_release`) |
| [DATASET_CLEANING.md](DATASET_CLEANING.md) | Limpieza, deduplicación, splits cluster-aware |
| [DATA_AUGMENTATION.md](DATA_AUGMENTATION.md) | Augmentación de secuencias |
| [BOLTZ_FOLDING.md](BOLTZ_FOLDING.md) | Folding offline con Boltz API |

## Modelos BBB

| Documento | Contenido |
|-----------|-----------|
| [BBB_CLASSIFIER.md](BBB_CLASSIFIER.md) | Clasificador tabular/ESM, experimentos exp01–exp06 |
| [STRUCTURAL_CLASSIFIER.md](STRUCTURAL_CLASSIFIER.md) | EGNN geométrico `struct_egnn_geo` (exp09), estabilidad, salidas |
| [STRUCTURAL_BBB_GUIDANCE.md](STRUCTURAL_BBB_GUIDANCE.md) | Guidance BBB diferenciable en difusión |

## Infraestructura

| Documento | Contenido |
|-----------|-----------|
| [VAST_TRAINING.md](VAST_TRAINING.md) | Workflow remoto bbb_models (upload, train, sync) |
| [reproducibility.md](reproducibility.md) | Roadmap de reproducibilidad (v2) |

## Diseño y escritura

| Documento | Contenido |
|-----------|-----------|
| [PAPER_WRITING_GUIDE.md](PAPER_WRITING_GUIDE.md) | Guía para redactar el paper/TFG |
| [EXPERIMENTAL_VALIDATION.tex](EXPERIMENTAL_VALIDATION.tex) | Borrador LaTeX de validación experimental |

## Referencias

Resúmenes de papers en `papers/` (PDFs excluidos de git).
