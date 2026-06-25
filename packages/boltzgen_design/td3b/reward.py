from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TD3BRewardConfig:
    tau: float = 0.2
    alpha: float = 0.5
    direction_target: int = 1


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def td3b_reward(
    affinity_score: float,
    bbb_probability: float,
    directional_score: float,
    cfg: TD3BRewardConfig,
) -> float:
    """R = g_affinity * g_BBB * sigma(d* f / tau)."""
    direction_term = sigmoid((cfg.direction_target * directional_score) / max(cfg.tau, 1e-6))
    return float(affinity_score * bbb_probability * direction_term)


def importance_weight(reward: float, alpha: float) -> float:
    return math.exp(reward / max(alpha, 1e-6))

