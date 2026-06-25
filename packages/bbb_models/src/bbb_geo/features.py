from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from bbb_classifier.config import DataConfig
    from bbb_geo.config import GeoExperimentConfig

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {a: i for i, a in enumerate(AA)}

# Kyte-Doolittle hydrophobicity (canonical order above).
KD_HYDRO = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

# Charge at pH 7.4 (rough per-residue scalar).
CHARGE_PH7 = {
    "D": -1.0,
    "E": -1.0,
    "K": 1.0,
    "R": 1.0,
    "H": 0.1,
}


def _node_features(sequence: str) -> np.ndarray:
    feats = []
    for aa in sequence.upper():
        one_hot = np.zeros(len(AA), dtype=np.float32)
        idx = AA_TO_ID.get(aa)
        if idx is not None:
            one_hot[idx] = 1.0
        kd = KD_HYDRO.get(aa, 0.0)
        charge = CHARGE_PH7.get(aa, 0.0)
        feats.append(np.concatenate([one_hot, np.array([kd, charge], dtype=np.float32)]))
    return np.stack(feats, axis=0)


def radius_graph(coords: np.ndarray | torch.Tensor, radius: float = 10.0) -> torch.Tensor:
    if isinstance(coords, np.ndarray):
        coords_t = torch.tensor(coords, dtype=torch.float32)
    else:
        coords_t = coords
    n = coords_t.shape[0]
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    dist = torch.cdist(coords_t, coords_t)
    mask = (dist <= radius) & (dist > 0)
    src, dst = torch.where(mask)
    if src.numel() == 0 and n > 1:
        idx = torch.arange(n - 1)
        src = torch.cat([idx, idx + 1])
        dst = torch.cat([idx + 1, idx])
    return torch.stack([src, dst], dim=0).long()


def rbf_edge_features(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    num_rbf: int = 16,
    d_min: float = 0.0,
    d_max: float = 20.0,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros((0, num_rbf), dtype=coords.dtype, device=coords.device)
    src, dst = edge_index
    rel = coords[src] - coords[dst]
    dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
    centers = torch.linspace(d_min, d_max, num_rbf, device=coords.device, dtype=coords.dtype)
    gamma = 1.0 / max((d_max - d_min) / num_rbf, 1e-6) ** 2
    return torch.exp(-gamma * (dist - centers) ** 2)


def build_struct_graph(
    coords: np.ndarray,
    sequence: str,
    radius: float = 10.0,
    num_rbf: int = 16,
) -> dict[str, torch.Tensor]:
    """Build a differentiable-ready structural graph from backbone/Ca coordinates."""
    coords_t = torch.tensor(np.asarray(coords, dtype=np.float32), dtype=torch.float32)
    node_feats = torch.tensor(_node_features(sequence), dtype=torch.float32)
    edge_index = radius_graph(coords_t, radius=radius)
    edge_attr = rbf_edge_features(coords_t, edge_index, num_rbf=num_rbf)
    return {
        "coords": coords_t,
        "node_feats": node_feats,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "sequence": sequence,
    }


def apply_coord_noise(
    coords: torch.Tensor, sigma: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    if sigma <= 0:
        return coords
    noise = torch.randn(coords.shape, generator=generator, dtype=coords.dtype, device=coords.device)
    out = coords + float(sigma) * noise
    if out.shape[0] > 0:
        out = out - out.mean(dim=0, keepdim=True)
    return out


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


def per_residue_hydrophobicity(
    sequence: str, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    values = [KD_HYDRO.get(aa, 0.0) for aa in sequence.upper()]
    return torch.tensor(values, dtype=dtype, device=device)


def per_residue_charge(sequence: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = [CHARGE_PH7.get(aa, 0.0) for aa in sequence.upper()]
    return torch.tensor(values, dtype=dtype, device=device)


def hydrophobic_moment(coords: torch.Tensor, hydro: torch.Tensor) -> torch.Tensor:
    """3D hydrophobic moment vector (Eisenberg-style) from coords and per-residue hydrophobicity."""
    if coords.shape[0] == 0:
        return torch.zeros(3, dtype=coords.dtype, device=coords.device)
    center = coords.mean(dim=0, keepdim=True)
    rel = coords - center
    weighted = rel * hydro.unsqueeze(-1)
    return weighted.sum(dim=0)


def amphipathicity_score(coords: torch.Tensor, hydro: torch.Tensor) -> torch.Tensor:
    """Scalar amphipathicity = || hydrophobic moment ||."""
    return torch.linalg.norm(hydrophobic_moment(coords, hydro))


def membrane_potential_energy(
    coords: torch.Tensor,
    sequence: str,
    *,
    maximize: bool = True,
) -> torch.Tensor:
    """Differentiable membrane/amphipathicity potential (higher = more amphipathic face)."""
    hydro = per_residue_hydrophobicity(sequence, device=coords.device, dtype=coords.dtype)
    score = amphipathicity_score(coords, hydro)
    return score if maximize else -score


def radius_of_gyration(coords: torch.Tensor) -> torch.Tensor:
    if coords.shape[0] == 0:
        return torch.zeros((), dtype=coords.dtype, device=coords.device)
    center = coords.mean(dim=0, keepdim=True)
    rel = coords - center
    return torch.sqrt((rel.pow(2).sum(dim=-1)).mean())


def helical_fraction_proxy(coords: torch.Tensor, threshold: float = 3.8) -> torch.Tensor:
    """Proxy for helical content from consecutive CA distances."""
    if coords.shape[0] < 4:
        return torch.zeros((), dtype=coords.dtype, device=coords.device)
    d = torch.linalg.norm(coords[1:] - coords[:-1], dim=-1)
    return (torch.abs(d - 3.8) <= threshold).float().mean()


# --- GeoFeatureBuilder ---


@dataclass
class GeoFeatureBundle:
    struct_df: pd.DataFrame
    struct_samples: list[dict[str, torch.Tensor | str | float]]
    tab_cols: list[str]


class GeoFeatureBuilder:
    def __init__(self, data: DataConfig, exp: GeoExperimentConfig):
        self.data = data
        self.exp = exp

    def build(self, df: pd.DataFrame) -> GeoFeatureBundle:
        if self.exp.model_type.value != "struct_egnn_geo":
            raise ValueError(f"Unsupported geo model_type: {self.exp.model_type.value}")

        seq_col = self.data.sequence_col
        struct_cfg = self.exp.struct
        manifest_path = struct_cfg.get("manifest_path") or self.data.struct_manifest_path
        merged = df.copy()
        if "coords_path" not in merged.columns:
            if "structure_coords_path" in merged.columns:
                dataset_root = Path(str(self.data.dataset_root or ".")).resolve()
                merged["coords_path"] = merged["structure_coords_path"].map(
                    lambda p: str((dataset_root / str(p)).resolve()) if pd.notna(p) else None
                )
            else:
                if not manifest_path:
                    raise ValueError(
                        "struct manifest_path required when dataset lacks coords_path columns"
                    )
                manifest = load_struct_manifest(manifest_path)
                merged = merge_dataset_with_manifest(df, manifest, sequence_col=seq_col)

        struct_df, struct_samples = build_struct_batch(
            merged,
            seq_col,
            radius=float(struct_cfg.get("radius", 10.0)),
            num_rbf=int(struct_cfg.get("num_rbf", 16)),
            sigma=0.0,
        )
        return GeoFeatureBundle(struct_df=struct_df, struct_samples=struct_samples, tab_cols=[])

    @staticmethod
    def apply_plddt_weights(train_df: pd.DataFrame, exp: GeoExperimentConfig) -> pd.DataFrame:
        if "plddt" not in train_df.columns:
            return train_df
        floor = float(exp.struct.get("plddt_weight_floor", 0.1))
        out = train_df.copy()
        base = out["sample_weight"].astype(float) if "sample_weight" in out.columns else 1.0
        out["sample_weight"] = base * out["plddt"].map(
            lambda v: plddt_sample_weight(v, floor=floor)
        )
        return out
