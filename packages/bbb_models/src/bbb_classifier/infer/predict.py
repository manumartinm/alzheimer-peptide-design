from __future__ import annotations

import numpy as np


def apply_threshold(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(prob) >= threshold).astype(int)
