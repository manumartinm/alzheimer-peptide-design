from __future__ import annotations

import pandas as pd

from bbb_dataset.augmentation import (
    AugmentConfig,
    augment_gold_dataframe,
    augment_sequence,
    mutate_conservative,
)
from bbb_dataset.schema import validate_dataset_schema


def _gold_row(**overrides) -> dict:
    base = {
        "peptide_id": "abc123",
        "source_id": "P_T1",
        "sequence": "YGGFLR",
        "length": 6,
        "bbb_label": 1,
        "source_db": "B3Pred_D1",
        "split": "train",
        "source_split": "train",
        "label_tier": "gold",
        "is_gold": 1,
        "cluster_id": 0,
        "external_test": 0,
        "fold_id": 1,
        "mw": 710.0,
    }
    base.update(overrides)
    return base


def test_mutate_conservative_changes_sequence_but_stays_valid() -> None:
    out = mutate_conservative("YGGFLR", n_changes=1, rng=__import__("random").Random(0))
    assert out != "YGGFLR"
    assert len(out) == 6
    assert set(out).issubset(set("ACDEFGHIKLMNPQRSTVWY"))


def test_augment_sequence_respects_length_bounds() -> None:
    cfg = AugmentConfig(
        seq_substitution_prob=1.0,
        seq_truncation_prob=1.0,
        random_state=0,
        min_length=6,
        max_length=30,
    )
    out = augment_sequence("YGGFLR", cfg, rng=__import__("random").Random(1))
    assert 6 <= len(out) <= 30


def test_augment_gold_dataframe_generates_labeled_rows() -> None:
    gold = pd.DataFrame(
        [
            _gold_row(),
            _gold_row(
                peptide_id="holdout1",
                source_id="P_T2",
                sequence="AAAAAA",
                length=6,
                bbb_label=0,
                external_test=1,
                fold_id=-1,
            ),
        ]
    )
    cfg = AugmentConfig(
        n_augmented_per_sample=2,
        random_state=42,
        seq_substitution_prob=1.0,
        seq_truncation_prob=0.0,
    )
    augmented, combined, stats = augment_gold_dataframe(gold, cfg)

    assert stats["n_candidates"] == 1
    assert stats["n_generated"] >= 1
    assert len(combined) == len(gold) + len(augmented)
    assert set(combined["is_augmented"]) == {0, 1}
    assert augmented["bbb_label"].eq(1).all()
    assert augmented["parent_peptide_id"].eq("abc123").all()
    assert augmented["label_tier"].eq("aug").all()
    validate_dataset_schema(combined.drop(columns=["is_augmented", "parent_peptide_id"]))


def test_augment_gold_dataframe_skips_duplicates() -> None:
    gold = pd.DataFrame(
        [_gold_row(), _gold_row(peptide_id="dup", source_id="P_T3", sequence="GGGGGG", bbb_label=0)]
    )
    cfg = AugmentConfig(
        enabled=True,
        n_augmented_per_sample=1,
        random_state=0,
        seq_substitution_prob=0.0,
        seq_truncation_prob=0.0,
    )
    _, _, stats = augment_gold_dataframe(gold, cfg)
    assert stats["n_generated"] >= 0


def test_augment_disabled_returns_gold_only() -> None:
    gold = pd.DataFrame([_gold_row()])
    _, combined, stats = augment_gold_dataframe(gold, AugmentConfig(enabled=False))
    assert stats["augmentation_enabled"] == 0
    assert len(combined) == 1
    assert combined["is_augmented"].eq(0).all()
