from __future__ import annotations

import numpy as np
import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {a: i for i, a in enumerate(AA)}


def seq_graph(sequence: str) -> dict[str, torch.Tensor]:
    """Sequence-only chain graph (legacy fallback; not structural)."""
    return sequence_graph(sequence)


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
