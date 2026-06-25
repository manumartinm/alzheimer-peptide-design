from __future__ import annotations


def dominates(a: dict, b: dict, keys: list[str]) -> bool:
    ge_all = all(float(a.get(k, 0.0)) >= float(b.get(k, 0.0)) for k in keys)
    gt_any = any(float(a.get(k, 0.0)) > float(b.get(k, 0.0)) for k in keys)
    return ge_all and gt_any


def pareto_front(candidates: list[dict], keys: list[str]) -> list[dict]:
    front: list[dict] = []
    for c in candidates:
        dominated = False
        for other in candidates:
            if other is c:
                continue
            if dominates(other, c, keys):
                dominated = True
                break
        if not dominated:
            front.append(c)
    return front

