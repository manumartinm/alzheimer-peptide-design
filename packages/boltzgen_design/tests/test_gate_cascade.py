from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
WORKBENCH = REPO_ROOT / "packages" / "boltzgen" / "workbench" / "gsk3b_guided"
sys.path.insert(0, str(REPO_ROOT / "packages"))

from boltzgen_design.filtering.gates import GateThresholds, evaluate_gates
from boltzgen_design.filtering.isoform_metrics import compute_isoform_selectivity, load_isoform_map
from boltzgen_design.filtering.struct_metrics import compute_struct_metrics


def test_evaluate_gates_all_pass() -> None:
    row = {
        "hotspot_fraction": 0.8,
        "atp_repulsion": 0.05,
        "bbb_probability": 0.7,
        "iptm": 0.8,
        "plddt": 90.0,
        "closure_rmsd": 1.0,
        "passes_sequence_liability": True,
    }
    flags = evaluate_gates(row, GateThresholds(), require=frozenset({"g3", "g4", "g5"}))
    assert flags["pass_all_gates"] is True


def test_gate_g6_pass() -> None:
    row = {
        "selectivity_margin": 0.7,
        "alpha_contact_fraction": 0.2,
    }
    flags = evaluate_gates(row, GateThresholds(), require=frozenset({"g6"}))
    assert flags["pass_g6_selectivity"] is True
    assert flags["pass_all_gates"] is True


def test_gate_g6_fail_low_margin() -> None:
    row = {
        "selectivity_margin": 0.25,
        "alpha_contact_fraction": 0.1,
    }
    flags = evaluate_gates(row, GateThresholds(), require=frozenset({"g6"}))
    assert flags["pass_g6_selectivity"] is False


def test_gate_g6_fail_high_alpha_contact() -> None:
    row = {
        "selectivity_margin": 0.8,
        "alpha_contact_fraction": 0.5,
    }
    flags = evaluate_gates(row, GateThresholds(), require=frozenset({"g6"}))
    assert flags["pass_g6_selectivity"] is False


def test_compute_isoform_selectivity_on_final_design() -> None:
    cif = WORKBENCH / "final_ranked_designs/final_30_designs/rank021_gsk3b_peptide_design_238_1.cif"
    gsk3a = PACKAGE_ROOT / "targets/gsk3a/gsk3a.cif"
    if not cif.exists() or not gsk3a.exists():
        return
    metrics = compute_isoform_selectivity(cif, gsk3a, load_isoform_map())
    assert metrics["n_isoform_interface"] > 0
    assert 0.0 <= metrics["selectivity_margin"] <= 1.0
    assert 0.0 <= metrics["alpha_contact_fraction"] <= 1.0


def test_compute_struct_metrics_on_final_design(tmp_path: Path | None = None) -> None:
    cif = WORKBENCH / "final_ranked_designs/final_30_designs/rank001_gsk3b_peptide_design_254_0.cif"
    if not cif.exists():
        return
    metrics = compute_struct_metrics(cif)
    assert 0.0 <= metrics["hotspot_fraction"] <= 1.0
    assert metrics["atp_repulsion"] >= 0.0
