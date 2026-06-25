from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReplayItem:
    sequence: str
    metadata: dict[str, Any]
    reward: float
    weight: float


@dataclass
class ReplayBuffer:
    max_size: int = 50_000
    items: list[ReplayItem] = field(default_factory=list)

    def add(self, item: ReplayItem) -> None:
        self.items.append(item)
        if len(self.items) > self.max_size:
            self.items = sorted(self.items, key=lambda x: x.weight, reverse=True)[: self.max_size]

    def topk(self, k: int) -> list[ReplayItem]:
        return sorted(self.items, key=lambda x: x.weight, reverse=True)[:k]

