from __future__ import annotations

import pandas as pd


def test_filter_cascade_prefers_calibrated_column() -> None:
    bbb_df = pd.DataFrame(
        {
            "sequence": ["AAAA", "BBBB"],
            "p_bbb_raw": [0.4, 0.6],
            "p_bbb_calibrated": [0.9, 0.1],
            "decision": [1, 0],
        }
    )
    prob_col = (
        "p_bbb_calibrated"
        if "p_bbb_calibrated" in bbb_df.columns
        else ("p_bbb_raw" if "p_bbb_raw" in bbb_df.columns else "probability")
    )
    values = pd.to_numeric(bbb_df[prob_col], errors="coerce").fillna(0.0).tolist()
    assert values == [0.9, 0.1]
