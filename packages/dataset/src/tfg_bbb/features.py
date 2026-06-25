from __future__ import annotations

from math import cos, sin, sqrt, radians
from typing import Dict

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
from pyteomics.mass import calculate_mass

HYDROPHOBIC = set("AILMFWVY")
POLAR = set("STNQCY")
BASIC = set("KRH")
ACIDIC = set("DE")
AROMATIC = set("FWYH")

EISENBERG_SCALE = {
    "A": 0.62,
    "C": 0.29,
    "D": -0.90,
    "E": -0.74,
    "F": 1.19,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "K": -1.50,
    "L": 1.06,
    "M": 0.64,
    "N": -0.78,
    "P": 0.12,
    "Q": -0.85,
    "R": -2.53,
    "S": -0.18,
    "T": -0.05,
    "V": 1.08,
    "W": 0.81,
    "Y": 0.26,
}


def _fraction(seq: str, aa_set: set[str]) -> float:
    if not seq:
        return 0.0
    return sum(1 for aa in seq if aa in aa_set) / len(seq)


def _hydrophobic_moment(seq: str, angle_deg: float = 100.0) -> float:
    """Approximate hydrophobic moment for alpha-helix geometry."""
    if not seq:
        return 0.0
    theta = radians(angle_deg)
    x_sum = 0.0
    y_sum = 0.0
    for i, aa in enumerate(seq):
        h = EISENBERG_SCALE.get(aa, 0.0)
        angle = i * theta
        x_sum += h * cos(angle)
        y_sum += h * sin(angle)
    return sqrt(x_sum**2 + y_sum**2) / len(seq)


def compute_features(seq: str) -> Dict[str, float]:
    g = GlobalDescriptor([seq])
    g.calculate_MW(amide=True)
    mw = float(g.descriptor[0, 0])

    g.isoelectric_point()
    pi = float(g.descriptor[0, 0])

    g.calculate_charge(ph=7.0, amide=True)
    net_charge = float(g.descriptor[0, 0])

    g.hydrophobic_ratio()
    hydrophobic_ratio = float(g.descriptor[0, 0])

    g.aliphatic_index()
    aliphatic_index = float(g.descriptor[0, 0])

    g.boman_index()
    boman_index = float(g.descriptor[0, 0])

    p = PeptideDescriptor([seq], "Eisenberg")
    p.calculate_global()
    mean_hydrophobicity = float(p.descriptor[0, 0])

    pa = ProteinAnalysis(seq)
    ext_coef_reduced, ext_coef_oxidized = pa.molar_extinction_coefficient()
    mw_pyteomics = float(calculate_mass(sequence=seq))

    length = len(seq)
    charge_density = net_charge / length if length else 0.0

    return {
        "mw": mw,
        "ext_coef_reduced": float(ext_coef_reduced),
        "ext_coef_oxidized": float(ext_coef_oxidized),
        "hydrophobic_ratio_pct": hydrophobic_ratio * 100.0,
        "pi": pi,
        "net_charge_ph7": net_charge,
        "total_charge": abs(net_charge),
        "mean_hydrophobicity": mean_hydrophobicity,
        "hydrophobicity_ph7": hydrophobic_ratio,
        "hydrophilic_ratio": 1.0 - hydrophobic_ratio,
        "aliphatic_index": aliphatic_index,
        "boman_index": boman_index,
        "aromaticity": pa.aromaticity(),
        "instability_index": pa.instability_index(),
        "gravy": pa.gravy(),
        "charge_density": charge_density,
        "aa_basic_pct": _fraction(seq, BASIC) * 100.0,
        "aa_acidic_pct": _fraction(seq, ACIDIC) * 100.0,
        "aa_aromatic_pct": _fraction(seq, AROMATIC) * 100.0,
        "aa_hydrophobic_pct": _fraction(seq, HYDROPHOBIC) * 100.0,
        "aa_polar_pct": _fraction(seq, POLAR) * 100.0,
        "hydrophobic_moment": _hydrophobic_moment(seq),
        "mw_pyteomics": mw_pyteomics,
        "mw_delta_abs": abs(mw - mw_pyteomics),
    }


def add_feature_columns(df: pd.DataFrame, sequence_col: str = "sequence") -> pd.DataFrame:
    feats = df[sequence_col].map(compute_features).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
