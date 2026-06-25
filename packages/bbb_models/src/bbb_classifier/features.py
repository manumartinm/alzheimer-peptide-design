from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bbb_classifier.config import DataConfig, ExperimentConfig

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {a: i for i, a in enumerate(AA)}


# --- Tabular ---


def infer_tabular_columns(df: pd.DataFrame, excluded: Iterable[str]) -> list[str]:
    excluded_set = set(excluded)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in numeric_cols if c not in excluded_set]


def tabular_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.zeros((len(df), 0), dtype=np.float32)
    return df[columns].fillna(0.0).to_numpy(dtype=np.float32)


# --- 3D descriptors ---


def feature_matrix_3d(df: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
    cols = columns or []
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    available = [c for c in cols if c in df.columns]
    if not available:
        return np.zeros((len(df), 0), dtype=np.float32)
    return df[available].fillna(0.0).to_numpy(dtype=np.float32)


# --- Sequence graphs ---


def sequence_graph(sequence: str) -> dict[str, torch.Tensor]:
    n = max(len(sequence), 1)
    x = torch.zeros((n, len(AA)), dtype=torch.float32)
    for i, aa in enumerate(sequence[:n]):
        idx = AA_TO_ID.get(aa)
        if idx is not None:
            x[i, idx] = 1.0

    if n == 1:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        src = np.arange(n - 1, dtype=np.int64)
        dst = np.arange(1, n, dtype=np.int64)
        edge_index = torch.tensor(
            np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]),
            dtype=torch.long,
        )
    return {"x": x, "edge_index": edge_index}


# --- ESM embeddings ---

_ESM_MODEL = None
_ESM_ALPHABET = None
_ESM_DEVICE = None
_PROJ_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _composition(seq: str) -> np.ndarray:
    vec = np.zeros(len(AA), dtype=np.float32)
    if not seq:
        return vec
    for aa in seq:
        idx = AA_TO_ID.get(aa)
        if idx is not None:
            vec[idx] += 1
    vec /= max(len(seq), 1)
    return vec


def _hash_noise(seq: str, dim: int) -> np.ndarray:
    digest = hashlib.sha256(seq.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.05, size=(dim,)).astype(np.float32)


def _mock_embedding(seq: str, dim: int) -> np.ndarray:
    base = np.zeros(dim, dtype=np.float32)
    comp = _composition(seq)
    base[: comp.shape[0]] = comp
    base[-1] = float(len(seq))
    base += _hash_noise(seq, dim)
    return base


def _projection(in_dim: int, out_dim: int) -> np.ndarray:
    key = (in_dim, out_dim)
    if key not in _PROJ_CACHE:
        rng = np.random.default_rng(42 + in_dim + out_dim)
        mat = rng.normal(0.0, 1.0 / np.sqrt(in_dim), size=(in_dim, out_dim)).astype(np.float32)
        _PROJ_CACHE[key] = mat
    return _PROJ_CACHE[key]


def _load_esm2():
    global _ESM_MODEL, _ESM_ALPHABET, _ESM_DEVICE
    if _ESM_MODEL is not None:
        return _ESM_MODEL, _ESM_ALPHABET, _ESM_DEVICE
    try:
        import esm
    except Exception:
        return None, None, None
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    _ESM_MODEL, _ESM_ALPHABET, _ESM_DEVICE = model, alphabet, device
    return _ESM_MODEL, _ESM_ALPHABET, _ESM_DEVICE


def _esm_embedding_raw(seq: str) -> np.ndarray | None:
    model, alphabet, device = _load_esm2()
    if model is None:
        return None
    batch_converter = alphabet.get_batch_converter()
    _, _, toks = batch_converter([("pep", seq)])
    toks = toks.to(device)
    with torch.no_grad():
        out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
    rep = out["representations"][model.num_layers][0, 1 : len(seq) + 1, :]
    return rep.mean(dim=0).detach().cpu().numpy().astype(np.float32)


def esm_embedding_from_sequence(
    seq: str, dim: int = 128, cache_dir: str | None = None
) -> np.ndarray:
    cache_root = Path(cache_dir or os.environ.get("BBB_ESM_CACHE_DIR", "artifacts/cache/esm2"))
    cache_root.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(seq.encode("utf-8")).hexdigest()
    fp = cache_root / f"{h}.npy"
    if fp.exists():
        raw = np.load(fp)
    else:
        raw = _esm_embedding_raw(seq)
        if raw is None:
            return _mock_embedding(seq, dim)
        np.save(fp, raw)
    if raw.shape[0] == dim:
        return raw.astype(np.float32)
    proj = _projection(raw.shape[0], dim)
    return (raw @ proj).astype(np.float32)


def batch_esm_embeddings(
    sequences: list[str], dim: int = 128, cache_dir: str | None = None
) -> np.ndarray:
    return np.stack(
        [esm_embedding_from_sequence(s, dim=dim, cache_dir=cache_dir) for s in sequences]
    ).astype(np.float32)


# --- FeatureBuilder ---


@dataclass
class FeatureBundle:
    tab: np.ndarray | None
    esm: np.ndarray | None
    feat3d: np.ndarray | None
    graphs: list[dict[str, torch.Tensor]] | None
    tab_cols: list[str]
    df: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "tab": self.tab,
            "esm": self.esm,
            "feat3d": self.feat3d,
            "graphs": self.graphs,
            "tab_cols": self.tab_cols,
            "df": self.df,
        }


class FeatureBuilder:
    def __init__(self, data: DataConfig, exp: ExperimentConfig):
        self.data = data
        self.exp = exp

    def build(self, df: pd.DataFrame) -> FeatureBundle:
        seq_col = self.data.sequence_col
        feats_cfg = self.exp.features
        esm_cache_dir = self.exp.esm.get("cache_dir", "artifacts/cache/esm2")
        tab_cols = infer_tabular_columns(df, self.data.tabular_exclude)
        x_tab = tabular_matrix(df, tab_cols) if feats_cfg.get("use_tabular", False) else None
        x_esm = (
            batch_esm_embeddings(
                df[seq_col].astype(str).tolist(),
                dim=int(self.exp.model.get("esm_dim", 128)),
                cache_dir=esm_cache_dir,
            )
            if feats_cfg.get("use_esm", False)
            else None
        )
        x_3d = (
            feature_matrix_3d(df, self.data.three_d_columns)
            if feats_cfg.get("use_3d", False)
            else None
        )
        graphs = (
            [sequence_graph(s) for s in df[seq_col].astype(str).tolist()]
            if feats_cfg.get("use_gnn", False)
            else None
        )
        return FeatureBundle(
            tab=x_tab,
            esm=x_esm,
            feat3d=x_3d,
            graphs=graphs,
            tab_cols=tab_cols,
            df=df,
        )


def rank_candidates(
    df: pd.DataFrame, prob_col: str = "p_bbb_calibrated", top_k: int = 100
) -> pd.DataFrame:
    return df.sort_values(prob_col, ascending=False).head(top_k).reset_index(drop=True)
