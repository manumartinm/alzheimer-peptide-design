from __future__ import annotations

from dataclasses import dataclass, fields

ALL_GATES = frozenset({"g1", "g2", "g3", "g4", "g5", "g6"})


@dataclass
class GateThresholds:
    g1_hotspot_fraction_min: float = 0.70
    g2_atp_repulsion_max: float = 0.20
    g3_bbb_min: float = 0.60
    g4_iptm_min: float = 0.75
    g4_plddt_min: float = 85.0
    g4_closure_rmsd_max: float = 1.2
    g6_selectivity_margin_min: float = 0.30
    g6_alpha_contact_max: float = 0.35

    @classmethod
    def from_mapping(cls, mapping: dict | None) -> GateThresholds:
        if not mapping:
            return cls()
        kwargs = {}
        key_map = {
            "gate_g1_hotspot_min_fraction": "g1_hotspot_fraction_min",
            "gate_g2_atp_repulsion_max": "g2_atp_repulsion_max",
            "gate_g3_bbb_min": "g3_bbb_min",
            "gate_g4_iptm_min": "g4_iptm_min",
            "gate_g4_plddt_min": "g4_plddt_min",
            "gate_g4_closure_rmsd_max": "g4_closure_rmsd_max",
            "gate_g6_selectivity_margin_min": "g6_selectivity_margin_min",
            "gate_g6_alpha_contact_max": "g6_alpha_contact_max",
        }
        for src, dst in key_map.items():
            if src in mapping:
                kwargs[dst] = float(mapping[src])
        return cls(**kwargs)


def gate_g1(candidate: dict, cfg: GateThresholds) -> bool:
    return float(candidate.get("hotspot_fraction", 0.0)) >= cfg.g1_hotspot_fraction_min


def gate_g2(candidate: dict, cfg: GateThresholds) -> bool:
    return float(candidate.get("atp_repulsion", 1e9)) <= cfg.g2_atp_repulsion_max


def gate_g3(candidate: dict, cfg: GateThresholds) -> bool:
    bbb = candidate.get("bbb_probability")
    if bbb is None or (isinstance(bbb, float) and np_isnan(bbb)):
        return False
    return float(bbb) >= cfg.g3_bbb_min


def gate_g4(candidate: dict, cfg: GateThresholds) -> bool:
    closure = candidate.get("closure_rmsd", 1e9)
    closure_val = float(closure) if closure is not None and not np_isnan(closure) else 1e9
    return (
        float(candidate.get("iptm", 0.0)) >= cfg.g4_iptm_min
        and float(candidate.get("plddt", 0.0)) >= cfg.g4_plddt_min
        and closure_val <= cfg.g4_closure_rmsd_max
    )


def gate_g5(candidate: dict) -> bool:
    return bool(candidate.get("passes_sequence_liability", False))


def gate_g6(candidate: dict, cfg: GateThresholds) -> bool:
    margin = candidate.get("selectivity_margin")
    alpha_frac = candidate.get("alpha_contact_fraction", 1.0)
    if margin is None or np_isnan(margin):
        return False
    if alpha_frac is not None and np_isnan(alpha_frac):
        alpha_frac = 1.0
    return (
        float(margin) >= cfg.g6_selectivity_margin_min
        and float(alpha_frac) <= cfg.g6_alpha_contact_max
    )


def np_isnan(value: object) -> bool:
    try:
        import math

        return math.isnan(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def evaluate_gates(
    candidate: dict,
    cfg: GateThresholds,
    *,
    require: frozenset[str] = ALL_GATES,
) -> dict[str, bool]:
    gate_map = {
        "g1": gate_g1(candidate, cfg),
        "g2": gate_g2(candidate, cfg),
        "g3": gate_g3(candidate, cfg),
        "g4": gate_g4(candidate, cfg),
        "g5": gate_g5(candidate),
        "g6": gate_g6(candidate, cfg),
    }
    return {
        "pass_g1_hotspot": gate_map["g1"],
        "pass_g2_atp": gate_map["g2"],
        "pass_g3_bbb": gate_map["g3"],
        "pass_g4_quality": gate_map["g4"],
        "pass_g5_liability": gate_map["g5"],
        "pass_g6_selectivity": gate_map["g6"],
        "pass_all_gates": all(gate_map[g] for g in require),
    }


def gate_thresholds_as_dict(cfg: GateThresholds) -> dict[str, float]:
    return {f.name: getattr(cfg, f.name) for f in fields(cfg)}
