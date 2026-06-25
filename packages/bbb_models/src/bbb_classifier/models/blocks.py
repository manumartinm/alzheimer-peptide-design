from __future__ import annotations

import torch.nn as nn


def mlp(d_in: int, d_hidden: int, d_out: int, dropout: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(d_hidden, d_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(d_hidden, d_out),
    )
