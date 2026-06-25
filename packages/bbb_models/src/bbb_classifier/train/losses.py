from __future__ import annotations

import torch
import torch.nn as nn


def bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if sample_weight is None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return criterion(logits, labels)
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
        reduction="none",
    )
    weighted = losses * sample_weight
    return weighted.mean()
