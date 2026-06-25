from __future__ import annotations

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def sanitize_probabilities(p: np.ndarray, *, fill: float = 0.5) -> np.ndarray:
    """Replace non-finite scores with a neutral probability for sklearn calibrators."""
    out = np.nan_to_num(np.asarray(p, dtype=float).reshape(-1), nan=fill, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0)


class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic") -> None:
        self.method = method
        self.model = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> None:
        p = sanitize_probabilities(p)
        y = np.asarray(y).astype(int).reshape(-1)
        if self.method == "platt":
            lr = LogisticRegression(max_iter=200)
            lr.fit(p.reshape(-1, 1), y)
            self.model = lr
        else:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y)
            self.model = iso

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = sanitize_probabilities(p)
        if self.model is None:
            return p
        if self.method == "platt":
            return self.model.predict_proba(p.reshape(-1, 1))[:, 1]
        return self.model.predict(p)

    def save(self, path: str) -> None:
        joblib.dump({"method": self.method, "model": self.model}, path)

    @staticmethod
    def load(path: str) -> ProbabilityCalibrator:
        data = joblib.load(path)
        obj = ProbabilityCalibrator(method=data["method"])
        obj.model = data["model"]
        return obj
