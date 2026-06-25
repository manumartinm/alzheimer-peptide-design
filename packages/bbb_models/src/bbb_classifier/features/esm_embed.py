from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {a: i for i, a in enumerate(AA)}
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
    emb = rep.mean(dim=0).detach().cpu().numpy().astype(np.float32)
    return emb


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
