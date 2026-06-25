# Arquitectura del proyecto

**Proyecto:** TFG — Diseño *in silico* de péptidos cíclicos fosfomiméticos compatibles con BBB que modulan GSK3β.

## Objetivo científico

Diseñar computacionalmente péptidos que:

1. Se unan al surco de reconocimiento de sustrato de GSK3β (hotspots: R96, R180, K205).
2. Eviten la hendidura de unión al ATP (toxicidad off-target vía vía Wnt).
3. Crucen la barrera hematoencefálica.

## Estructura del monorepo

```
alzheimer-peptide-design/
├── packages/
│   ├── dataset/           → curación de datos BBB (CLI: tfg-bbb-build)
│   ├── bbb_models/        → clasificador + EGNN geométrico
│   ├── boltzgen_design/   → orquestación GSK3β (guidance, filtros)
│   └── boltzgen/          → motor de difusión (submodule)
└── docs/                  → esta documentación
```

Entorno unificado: `uv sync` en la raíz instala todos los paquetes en editable mode.

## Fase 1: Clasificador BBB (completada)

**Ubicación:** `packages/bbb_models/` (`bbb_classifier`, `bbb_geo`)

Oracle de permeabilidad que alimenta ranking, filtrado y (indirectamente) generación.

- **Datos:** B3Pred D1 + expansión opcional. Longitud 6–30 aa, deduplicación al 90%, folds cluster-aware.
- **Release HF:** `dataset/data/hf_release/` — 825 péptidos con estructuras Boltz (`tfg-bbb-export-hf --variant full`).
- **Oracle (secuencia):** `exp03_esm_tab_mlp` — ESM-2 + tabular, calibrado isotónico.
- **Guidance (geometría):** `exp09_struct_egnn_geo` — EGNN con ruido EDM; ver [STRUCTURAL_CLASSIFIER.md](STRUCTURAL_CLASSIFIER.md).
- **Ejecución:** `uv run python scripts/classifier/train.py` o `scripts/geo/train.py`; remoto en [VAST_TRAINING.md](VAST_TRAINING.md).

## Fase 2: Generación con BoltzGen (en progreso)

**Ubicación:** `packages/boltzgen_design/` + hooks en `packages/boltzgen/src/boltzgen/model/modules/diffusion.py`

### Guidance geométrico (SDE inversa)

Potenciales diferenciables durante inferencia:

- **U_h:** recompensa por proximidad (≤ 5 Å) a hotspots positivos.
- **U_a:** penalización por proximidad a la hendidura ATP.

### BBB y no-diferenciabilidad

El clasificador por secuencia (`exp03`) usa ESM-2 y descriptores tabulares — **no diferenciables respecto a coordenadas 3D**. Por tanto:

- **Guidance por gradiente en SDE:** hotspots ATP + EGNN geométrico `p_geo` + potencial de anfipaticidad.
- **Señal BBB por secuencia:** TD3B (reward amortizado con WDCE + anclaje KL) usando `p_bbb_calibrated` del oracle.

El modelo `struct_egnn_geo` (exp09) entrenado sobre estructuras plegadas permite guidance BBB diferenciable. Ver [STRUCTURAL_BBB_GUIDANCE.md](STRUCTURAL_BBB_GUIDANCE.md).

## Fase 3: Filtrado y MD (en progreso)

### Cascada de 5 gates

1. **G1:** >70% hotspots engaged ≤ 5 Å.
2. **G2:** score de repulsión ATP bajo umbral.
3. **G3:** p_BBB ≥ 0.6 + solubilidad.
4. **G4:** ipTM ≥ 0.75, pLDDT ≥ 85, RMSD cierre cíclico ≤ 1.2 Å.
5. **G5:** liabilities de secuencia (polibásico, deamidación, Met/Cys, agregación).

### Validación MD

Top 10–30 candidatos Pareto → OpenMM, CHARMM36m, 100 ns exploratorio, 500 ns para el lead.

## Convenciones para quien continúe

1. Leer [AGENT_CONTEXT.md](AGENT_CONTEXT.md) para detalle completo.
2. Respetar DVC cuando exista stage (`dvc repro`, no scripts ad-hoc).
3. No backpropagar coordenadas 3D a través de ESM-2 o descriptores tabulares.
4. Usar `uv run` desde la raíz del repo.
5. No commitear artefactos grandes; usar paths ignorados o DVC (v2).
6. Actualizar docs si cambias la arquitectura.
