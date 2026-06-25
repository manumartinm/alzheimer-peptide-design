from __future__ import annotations

import random
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

import pandas as pd
import yaml

from .aa import CANONICAL_AA
from .features import add_feature_columns, compute_features

# Conservative substitutions inspired by BLOSUM-like similarity.
CONSERVATIVE_MAP = {
    "A": "SVG",
    "C": "ST",
    "D": "EN",
    "E": "DQK",
    "F": "YWL",
    "G": "AS",
    "H": "KRQ",
    "I": "LVMT",
    "K": "RHEQ",
    "L": "IVMF",
    "M": "ILV",
    "N": "DQST",
    "P": "AST",
    "Q": "ENKRH",
    "R": "KQH",
    "S": "TAGNC",
    "T": "SAV",
    "V": "AILT",
    "W": "FY",
    "Y": "FW",
}


@dataclass
class AugmentConfig:
    enabled: bool = True
    seq_substitution_prob: float = 0.5
    seq_truncation_prob: float = 0.3
    n_augmented_per_sample: int = 2
    random_state: int = 42
    min_length: int = 6
    max_length: int = 30
    only_non_holdout: bool = True


def load_augment_config(
    path: Path, *, min_length: int, max_length: int, random_seed: int
) -> AugmentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload = payload or {}
    payload.setdefault("min_length", min_length)
    payload.setdefault("max_length", max_length)
    payload.setdefault("random_state", random_seed)
    return AugmentConfig(**payload)


def mutate_conservative(seq: str, n_changes: int = 1, rng: random.Random | None = None) -> str:
    rng = rng or random
    if not seq:
        return seq
    seq_list = list(seq)
    positions = list(range(len(seq_list)))
    rng.shuffle(positions)
    changed = 0
    for pos in positions:
        aa = seq_list[pos]
        candidates = CONSERVATIVE_MAP.get(aa, "")
        if not candidates:
            continue
        seq_list[pos] = rng.choice(list(candidates))
        changed += 1
        if changed >= n_changes:
            break
    return "".join(seq_list)


def truncate_terminal(
    seq: str, max_cut: int = 2, min_len: int = 5, rng: random.Random | None = None
) -> str:
    rng = rng or random
    if len(seq) <= min_len:
        return seq
    cut = rng.randint(1, min(max_cut, len(seq) - min_len))
    if rng.random() < 0.5:
        return seq[cut:]
    return seq[:-cut]


def augment_sequence(seq: str, cfg: AugmentConfig, rng: random.Random | None = None) -> str:
    rng = rng or random
    out = seq
    if rng.random() < cfg.seq_substitution_prob:
        out = mutate_conservative(out, n_changes=1 if len(out) < 12 else 2, rng=rng)
    if rng.random() < cfg.seq_truncation_prob:
        out = truncate_terminal(out, max_cut=2, min_len=cfg.min_length, rng=rng)
    return out


def _valid_sequence(seq: str, cfg: AugmentConfig) -> bool:
    if not seq or not set(seq).issubset(CANONICAL_AA):
        return False
    return cfg.min_length <= len(seq) <= cfg.max_length


def _peptide_id(seq: str) -> str:
    return sha1(seq.encode()).hexdigest()[:12]


def augment_gold_dataframe(
    gold_df: pd.DataFrame,
    cfg: AugmentConfig,
    *,
    sequence_col: str = "sequence",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Build augmented peptide rows from a gold dataset.

    Returns:
        augmented_only: newly generated rows (`is_augmented == 1`)
        combined: original gold rows plus augmented rows
        stats: generation counters
    """
    if not cfg.enabled:
        gold = gold_df.copy()
        gold["is_augmented"] = 0
        gold["parent_peptide_id"] = ""
        return gold.iloc[0:0].copy(), gold, {"augmentation_enabled": 0}

    rng = random.Random(cfg.random_state)
    gold = gold_df.copy()
    gold[sequence_col] = gold[sequence_col].astype(str).str.upper()
    gold["is_augmented"] = 0
    gold["parent_peptide_id"] = ""

    if cfg.only_non_holdout and "external_test" in gold.columns:
        candidates = gold[gold["external_test"] == 0].copy()
    else:
        candidates = gold.copy()

    # Keep validation fold pristine (fold_id==0); augmented rows inherit fold metadata.
    if "fold_id" in candidates.columns:
        candidates = candidates[candidates["fold_id"] != 0].copy()

    existing = set(gold[sequence_col].tolist())
    augmented_rows: list[dict] = []
    stats = {
        "augmentation_enabled": 1,
        "n_candidates": len(candidates),
        "n_requested": int(len(candidates) * max(1, cfg.n_augmented_per_sample)),
        "n_generated": 0,
        "n_skipped_duplicate": 0,
        "n_skipped_invalid": 0,
        "n_skipped_unchanged": 0,
    }

    for _, parent in candidates.iterrows():
        parent_seq = str(parent[sequence_col])
        for aug_idx in range(max(1, cfg.n_augmented_per_sample)):
            new_seq = augment_sequence(parent_seq, cfg, rng=rng).upper()
            if new_seq == parent_seq:
                stats["n_skipped_unchanged"] += 1
                continue
            if not _valid_sequence(new_seq, cfg):
                stats["n_skipped_invalid"] += 1
                continue
            if new_seq in existing:
                stats["n_skipped_duplicate"] += 1
                continue

            row = parent.to_dict()
            row[sequence_col] = new_seq
            row["length"] = len(new_seq)
            row["peptide_id"] = _peptide_id(new_seq)
            row["parent_peptide_id"] = str(parent["peptide_id"])
            row["source_id"] = f"AUG_{parent['source_id']}_{aug_idx}"
            row["label_tier"] = "aug"
            row["is_augmented"] = 1
            augmented_rows.append(row)
            existing.add(new_seq)
            stats["n_generated"] += 1

    if not augmented_rows:
        return pd.DataFrame(columns=gold.columns), gold, stats

    feat_keys = set(compute_features("AAAAAA").keys())
    meta_cols = [c for c in gold.columns if c not in feat_keys]
    augmented_df = pd.DataFrame(augmented_rows)[meta_cols]
    augmented_df = add_feature_columns(augmented_df, sequence_col=sequence_col)
    combined = pd.concat([gold, augmented_df], ignore_index=True, sort=False)
    return augmented_df, combined, stats
