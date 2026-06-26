from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_final_designs(campaign_dir: Path) -> pd.DataFrame:
    """Load BoltzGen final_* designs with resolved refolded CIF paths."""
    campaign_dir = campaign_dir.resolve()
    ranked_dir = campaign_dir / "final_ranked_designs"
    final_dir = ranked_dir / "final_30_designs"
    metrics_csv = ranked_dir / "final_designs_metrics_30.csv"

    if metrics_csv.exists() and final_dir.is_dir():
        df = pd.read_csv(metrics_csv)
        df["structure_path"] = df.apply(
            lambda r: str(final_dir / f"rank{int(r.final_rank):03d}_{r.id}.cif"),
            axis=1,
        )
        return df

    metrics_files = sorted(ranked_dir.glob("final_designs_metrics_*.csv"))
    if metrics_files:
        df = pd.read_csv(metrics_files[0])
        budget_dir = ranked_dir / metrics_files[0].stem.replace("final_designs_metrics_", "final_")
        if not budget_dir.exists():
            budget_dir = final_dir
        if "final_rank" in df.columns:
            df["structure_path"] = df.apply(
                lambda r: str(budget_dir / f"rank{int(r.final_rank):03d}_{r.id}.cif"),
                axis=1,
            )
        return df

    aggregate = sorted(
        (campaign_dir / "intermediate_designs_inverse_folded").glob("aggregate_metrics*.csv")
    )
    if aggregate:
        return pd.read_csv(aggregate[0])

    raise FileNotFoundError(
        f"No final_designs_metrics_*.csv under {ranked_dir} and no aggregate_metrics in campaign dir"
    )


def normalize_candidate_frame(df: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    out = df.copy()

    if "structure_path" not in out.columns:
        if "file_name" in out.columns and (input_dir / "final_30_designs").is_dir():
            out["structure_path"] = out["file_name"].map(
                lambda n: str(input_dir / "final_30_designs" / n)
            )
        elif "id" in out.columns and "final_rank" in out.columns:
            final_dir = input_dir / "final_30_designs"
            out["structure_path"] = out.apply(
                lambda r: str(final_dir / f"rank{int(r.final_rank):03d}_{r.id}.cif"),
                axis=1,
            )
        else:
            raise ValueError("Could not infer structure_path column for candidates")

    out["structure_path"] = out["structure_path"].map(
        lambda p: (
            str(Path(p).resolve()) if Path(p).is_absolute() else str((input_dir / p).resolve())
        )
    )

    if "sequence" not in out.columns:
        if "designed_chain_sequence" in out.columns:
            out["sequence"] = out["designed_chain_sequence"].astype(str)
        elif "designed_sequence" in out.columns:
            out["sequence"] = out["designed_sequence"].astype(str)
        elif "sequence" in out.columns:
            out["sequence"] = out["sequence"].astype(str)
        else:
            out["sequence"] = ""

    return out
