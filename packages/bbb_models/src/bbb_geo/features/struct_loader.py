from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bbb_geo.features.struct_graph import apply_coord_noise, build_struct_graph


def load_struct_manifest(manifest_path: str | Path) -> pd.DataFrame:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Structural manifest not found: {path}")
    return pd.read_parquet(path)


def plddt_sample_weight(plddt: float | np.floating | None, *, floor: float = 0.1) -> float:
    """Map mean pLDDT (0–100) to a loss weight in [floor, 1.0]."""
    if plddt is None or (isinstance(plddt, float) and np.isnan(plddt)):
        return 1.0
    return float(np.clip(float(plddt) / 100.0, floor, 1.0))


def merge_dataset_with_manifest(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
    sequence_col: str = "sequence",
) -> pd.DataFrame:
    manifest = manifest.copy()
    manifest[sequence_col] = manifest["sequence"].astype(str).str.upper()
    merged = df.copy()
    merged[sequence_col] = merged[sequence_col].astype(str).str.upper()
    return merged.merge(
        manifest[[sequence_col, "coords_path", "plddt", "ptm", "sequence_hash"]],
        on=sequence_col,
        how="inner",
    )


def build_struct_sample(
    coords_path: str | Path,
    sequence: str,
    *,
    radius: float = 10.0,
    num_rbf: int = 16,
    sigma: float = 0.0,
    center: bool = True,
) -> dict[str, torch.Tensor | str | float]:
    payload = np.load(Path(coords_path), allow_pickle=True)
    coords = payload["coords"].astype(np.float32)
    seq_arr = payload.get("sequence")
    if seq_arr is not None:
        if getattr(seq_arr, "ndim", 0) == 0:
            seq_from_file = "".join(str(seq_arr.item()))
        else:
            seq_from_file = "".join(str(x) for x in seq_arr.tolist())
        sequence = seq_from_file or sequence
    graph = build_struct_graph(coords, sequence, radius=radius, num_rbf=num_rbf)
    coords_t = graph["coords"]
    if center and coords_t.shape[0] > 0:
        coords_t = coords_t - coords_t.mean(dim=0, keepdim=True)
        graph["coords"] = coords_t
    if sigma > 0:
        graph["coords"] = apply_coord_noise(graph["coords"], sigma=sigma)
    graph["sigma"] = float(sigma)
    graph["sequence"] = sequence
    return graph


def build_struct_batch(
    df: pd.DataFrame,
    sequence_col: str,
    *,
    radius: float = 10.0,
    num_rbf: int = 16,
    sigma: float = 0.0,
) -> tuple[pd.DataFrame, list[dict[str, torch.Tensor | str | float]]]:
    rows: list[pd.Series] = []
    samples: list[dict[str, torch.Tensor | str | float]] = []
    for _, row in df.iterrows():
        if pd.isna(row.get("coords_path")):
            continue
        rows.append(row)
        samples.append(
            build_struct_sample(
                row["coords_path"],
                str(row[sequence_col]),
                radius=radius,
                num_rbf=num_rbf,
                sigma=sigma,
            )
        )
    if not rows:
        return df.iloc[0:0].copy(), []
    return pd.DataFrame(rows).reset_index(drop=True), samples
