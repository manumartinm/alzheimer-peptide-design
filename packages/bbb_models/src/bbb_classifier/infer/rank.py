from __future__ import annotations

import pandas as pd


def rank_candidates(df: pd.DataFrame, prob_col: str = "p_bbb_calibrated", top_k: int = 100) -> pd.DataFrame:
    return df.sort_values(prob_col, ascending=False).head(top_k).reset_index(drop=True)
