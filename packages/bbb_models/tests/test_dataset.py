import sys
from pathlib import Path

import pandas as pd

from bbb_classifier.data.splits import train_val_split

_dataset_src = Path(__file__).resolve().parents[2] / "dataset" / "src"
if str(_dataset_src) not in sys.path:
    sys.path.append(str(_dataset_src))
from tfg_bbb.clean import deduplicate_by_identity


def test_split_non_empty():
    df = pd.DataFrame(
        {
            "sequence": ["AAAA", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG"],
            "bbb_label": [0, 1, 0, 1, 0, 1],
        }
    )
    tr, va = train_val_split(df, label_col="bbb_label", test_size=0.33, random_state=42)
    assert len(tr) > 0
    assert len(va) > 0


def test_deduplicate_by_identity_prefers_positive():
    df = pd.DataFrame(
        {
            "sequence": ["ACDEFG", "ACDEFA", "TTTTTT", "TTTTTA"],
            "bbb_label": [0, 1, 0, 0],
        }
    )
    out, _ = deduplicate_by_identity(
        df, sequence_col="sequence", label_col="bbb_label", threshold=0.8
    )
    # two clusters should remain
    assert len(out) == 2
    # in the ACDEF* cluster, BBB+ is kept
    assert (out["bbb_label"] == 1).any()
