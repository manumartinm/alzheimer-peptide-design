from __future__ import annotations

from enum import StrEnum


class ModelType(StrEnum):
    TABULAR_LGBM = "tabular_lgbm"
    ESM_LGBM = "esm_lgbm"
    ESM_TAB_MLP = "esm_tab_mlp"
    ESM_TAB_3DFEAT = "esm_tab_3dfeat"
    ESM_TAB_GNN = "esm_tab_gnn"

    @property
    def is_torch(self) -> bool:
        return self in {
            ModelType.ESM_TAB_MLP,
            ModelType.ESM_TAB_3DFEAT,
            ModelType.ESM_TAB_GNN,
        }

    @property
    def is_lgbm(self) -> bool:
        return self in {ModelType.TABULAR_LGBM, ModelType.ESM_LGBM}


CLASSIFIER_MODEL_TYPES = frozenset(ModelType)


class CalibrationMethod(StrEnum):
    ISOTONIC = "isotonic"
    PLATT = "platt"
    NONE = "none"
