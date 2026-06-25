from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateThresholds:
    g1_hotspot_fraction_min: float = 0.70
    g2_atp_repulsion_max: float = 0.20
    g3_bbb_min: float = 0.60
    g4_iptm_min: float = 0.75
    g4_plddt_min: float = 85.0
    g4_closure_rmsd_max: float = 1.2


def gate_g1(candidate: dict, cfg: GateThresholds) -> bool:
    return float(candidate.get("hotspot_fraction", 0.0)) >= cfg.g1_hotspot_fraction_min


def gate_g2(candidate: dict, cfg: GateThresholds) -> bool:
    return float(candidate.get("atp_repulsion", 1e9)) <= cfg.g2_atp_repulsion_max


def gate_g3(candidate: dict, cfg: GateThresholds) -> bool:
    return float(candidate.get("bbb_probability", 0.0)) >= cfg.g3_bbb_min


def gate_g4(candidate: dict, cfg: GateThresholds) -> bool:
    return (
        float(candidate.get("iptm", 0.0)) >= cfg.g4_iptm_min
        and float(candidate.get("plddt", 0.0)) >= cfg.g4_plddt_min
        and float(candidate.get("closure_rmsd", 1e9)) <= cfg.g4_closure_rmsd_max
    )


def gate_g5(candidate: dict) -> bool:
    return bool(candidate.get("passes_sequence_liability", False))
