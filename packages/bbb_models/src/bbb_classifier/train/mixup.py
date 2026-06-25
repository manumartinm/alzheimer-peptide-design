from __future__ import annotations

from typing import Any

import numpy as np
import torch


def apply_mixup(
    batch: dict[str, Any],
    alpha: float = 0.2,
    prob: float = 0.5,
) -> dict[str, Any]:
    if alpha <= 0 or np.random.rand() > prob:
        return batch
    y = batch["y"]
    n = y.shape[0]
    if n < 2:
        return batch
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(n, device=y.device)

    out = dict(batch)
    out["y"] = lam * y + (1.0 - lam) * y[perm]
    for key in ("tab", "esm", "feat3d"):
        if key in out and out[key] is not None:
            out[key] = lam * out[key] + (1.0 - lam) * out[key][perm]
    return out
