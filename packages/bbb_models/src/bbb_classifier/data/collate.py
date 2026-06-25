from __future__ import annotations

from typing import Any

import numpy as np
import torch


def stack_or_none(values: list[np.ndarray | None]) -> torch.Tensor | None:
    if not values or values[0] is None:
        return None
    return torch.tensor(np.stack(values), dtype=torch.float32)


def collate_torch_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    y = torch.tensor([s["y"] for s in samples], dtype=torch.float32).unsqueeze(1)
    tab = stack_or_none([s.get("tab") for s in samples])
    esm = stack_or_none([s.get("esm") for s in samples])
    feat3d = stack_or_none([s.get("feat3d") for s in samples])
    graphs = [s.get("graph") for s in samples]
    return {"y": y, "tab": tab, "esm": esm, "feat3d": feat3d, "graphs": graphs}
