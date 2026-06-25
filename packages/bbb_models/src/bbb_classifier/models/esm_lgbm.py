from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

from sklearn.ensemble import HistGradientBoostingClassifier


class ESMLGBMModel:
    def __init__(self, random_state: int = 42):
        if LGBMClassifier is not None:
            self.model = LGBMClassifier(
                n_estimators=800,
                learning_rate=0.025,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
            )
        else:
            self.model = HistGradientBoostingClassifier(random_state=random_state)

    def fit(self, x_esm: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x_esm, y)

    def predict_proba(self, x_esm: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba(x_esm)
        return p[:, 1] if p.ndim == 2 else p

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str | Path) -> "ESMLGBMModel":
        wrapper = ESMLGBMModel()
        wrapper.model = joblib.load(path)
        return wrapper
