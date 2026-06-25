from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency fallback
    LGBMClassifier = None

from sklearn.ensemble import HistGradientBoostingClassifier


class TabularLGBMModel:
    def __init__(self, random_state: int = 42):
        if LGBMClassifier is not None:
            self.model = LGBMClassifier(
                n_estimators=600,
                learning_rate=0.03,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
            )
        else:
            self.model = HistGradientBoostingClassifier(random_state=random_state)

    def fit(self, x_tab: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x_tab, y)

    def predict_proba(self, x_tab: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba(x_tab)
        return p[:, 1] if p.ndim == 2 else p

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str | Path) -> TabularLGBMModel:
        wrapper = TabularLGBMModel()
        wrapper.model = joblib.load(path)
        return wrapper
