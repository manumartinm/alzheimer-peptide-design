from __future__ import annotations

from enum import StrEnum


class GeoModelType(StrEnum):
    STRUCT_EGNN_GEO = "struct_egnn_geo"


GEO_MODEL_TYPES = frozenset(GeoModelType)
