# State of the Art (SOTA) - GSK3β Modulators & Peptide Inhibitors in Alzheimer's Disease (2024-2026)

## 1. Contexto General (GSK3β en Alzheimer)
La quinasa **GSK3β** (Glycogen synthase kinase 3 beta) es una diana terapéutica crucial en la enfermedad de Alzheimer (AD) debido a su papel central en la **hiperfosforilación de la proteína Tau** y la patología del amiloide-beta (Aβ). El problema histórico con los inhibidores competitivos de ATP convencionales es que causan efectos secundarios severos al bloquear vías de mantenimiento esenciales (como la señalización Wnt). Por ello, el SOTA actual se enfoca en la **modulación selectiva de sustrato** mediante péptidos.

## 2. Avances Recientes (Papers descargados)

### A. Diseño Racional de Péptidos derivados de caseína-kappa
* **Paper:** *Targeting GSK-3β with Peptide Inhibitors: A Rational Computational Strategy for Alzheimer’s Disease Intervention (bioRxiv, Diciembre 2024)* y *Rational design of k-casein peptides to modulate GSK-3B dynamics for Alzheimer’s therapy (Scientific Reports, Marzo 2026)*.
* **Enfoque:** Utilizan herramientas computacionales (simulaciones de Dinámica Molecular y docking con HADDOCK) para diseñar péptidos basados en la caseína kappa.
* **Resultados Clave:**
  * Identificaron el péptido **MP31 (HPDFVAPFPE)** y **PEP8 / PEP44** como los mejores candidatos.
  * Estos péptidos interactúan directamente con el bolsillo de unión a ATP y los sitios catalíticos (Asp200), estabilizando la quinasa y reduciendo la flexibilidad estructural. Esta estabilización impide el acceso al sustrato.
  * La optimización mutacional mejoró enormemente la afinidad, demostrando que un diseño guiado por MM/PBSA puede optimizar redes de interacción intermoleculares.

### B. Inhibidores Activados Específicamente por Akt
* **Paper:** *Akt-activated GSK3β inhibitory peptide effectively blocks tau hyperphosphorylation (PubMed, 2024-2025)*.
* **Enfoque:** Desarrollaron un péptido inhibidor de GSK3β (GIP) que es **activado específicamente por la quinasa Akt**. Combina el motivo PPPSPxS (del correceptor LRP6) que inhibe directamente a GSK3β, con la secuencia diana de Akt (RxRxxS).
* **Resultados Clave:**
  * En modelos in vivo (ratones 3×Tg-AD), la administración intravenosa del péptido redujo significativamente la fosforilación de Tau en el hipocampo y mejoró los déficits de memoria.
  * Se diseñó para ser permeable a las células, mostrando neuroprotección sin interferir con las funciones basales de GSK3β.

## 3. Comparación con tu TFG (Moduladores Cíclicos + Permeabilidad BBB)

El SOTA actual subraya la necesidad de péptidos **específicos de sustrato** y con **capacidad de cruzar barreras biológicas** (permeabilidad celular/BBB). Tu TFG está **completamente alineado** y añade innovaciones sustanciales sobre este SOTA:

1. **Estructura Cíclica vs Lineal:** Los papers recientes (como MP31 o PEP8) se basan principalmente en péptidos lineales. Tu TFG propone **péptidos cíclicos**, los cuales ofrecen mayor estabilidad proteolítica in vivo, un paso crucial para fármacos reales.
2. **Generación Difusiva All-Atom (BoltzGen) vs Mutagénesis/Docking:** Mientras el SOTA usa cribado virtual y simulaciones MD sobre bibliotecas limitadas de mutaciones (ej. 48 variaciones), tú utilizas inteligencia artificial generativa pre-entrenada para explorar el espacio químico masivo de forma eficiente.
3. **Restricción Explícita BBB:** El péptido GIP del paper logró llegar al cerebro in vivo de forma empírica (fusionándolo a una secuencia penetrante). Tu enfoque incluye un **clasificador BBB predictivo integrado en el loop generativo (vía TD3B)**, haciendo que la permeabilidad de la Barrera Hematoencefálica sea una restricción de diseño *by-design*, no un parche posterior.

## 4. Archivos Descargados
Los PDFs correspondientes a esta investigación se han descargado en el directorio:
`TFG/documentation/papers/`
- `Targeting_GSK3b_BioRxiv.pdf`
- `SciRep_Rational_Design.pdf`
