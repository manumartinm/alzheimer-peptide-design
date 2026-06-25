from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bbb_classifier.utils.io import ensure_dir


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    torch.save(state, p)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")
