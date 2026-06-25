from __future__ import annotations


class EarlyStopper:
    def __init__(self, patience: int = 8, maximize: bool = True):
        self.patience = patience
        self.maximize = maximize
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return False
        improved = value > self.best if self.maximize else value < self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
