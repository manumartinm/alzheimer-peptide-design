from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .reward import TD3BRewardConfig, importance_weight, td3b_reward


@dataclass
class WeightedSample:
    sequence: str
    affinity_score: float
    bbb_probability: float
    directional_score: float
    reward: float
    weight: float


def build_weighted_samples(
    candidates: list[dict[str, Any]],
    cfg: TD3BRewardConfig,
) -> list[WeightedSample]:
    weighted: list[WeightedSample] = []
    for item in candidates:
        reward = td3b_reward(
            affinity_score=float(item["affinity_score"]),
            bbb_probability=float(item["bbb_probability"]),
            directional_score=float(item["directional_score"]),
            cfg=cfg,
        )
        weighted.append(
            WeightedSample(
                sequence=item["sequence"],
                affinity_score=float(item["affinity_score"]),
                bbb_probability=float(item["bbb_probability"]),
                directional_score=float(item["directional_score"]),
                reward=reward,
                weight=importance_weight(reward, cfg.alpha),
            )
        )
    return sorted(weighted, key=lambda x: x.weight, reverse=True)


def dump_weighted_replay(samples: list[WeightedSample], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = [sample.__dict__ for sample in samples]
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

