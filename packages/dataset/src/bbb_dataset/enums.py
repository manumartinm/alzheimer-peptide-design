from __future__ import annotations

from enum import IntEnum, StrEnum


class SourceDb(StrEnum):
    B3PRED_D1 = "B3Pred_D1"
    B3PDB = "B3Pdb"
    BRAINPEPS = "Brainpeps"


class LabelTier(StrEnum):
    GOLD = "gold"
    SILVER = "silver"
    AUGMENTED = "aug"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"


class BbbLabel(IntEnum):
    NEGATIVE = 0
    POSITIVE = 1


class ProcessedArtifact(StrEnum):
    PEPTIDES_BBB = "peptides_bbb.parquet"
    AUGMENTED_EXTRA = "peptides_bbb_augmented_extra.parquet"
    COMBINED = "peptides_bbb_with_augmentation.parquet"
    STRUCT_MANIFEST = "peptides_struct_manifest.parquet"
    PREVIEW_CSV = "peptides_bbb_preview.csv"
    AUGMENTATION_STATS = "augmentation_stats.json"
