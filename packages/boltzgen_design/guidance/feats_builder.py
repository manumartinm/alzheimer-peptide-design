from __future__ import annotations

from pathlib import Path
from typing import Any


def guidance_config_to_feats(
    guidance: dict[str, Any],
    *,
    target_root: Path | None = None,
) -> dict[str, float | str]:
    """Map campaign guidance YAML/JSON to diffusion `feats` keys."""
    root = target_root or Path(".")
    bbb = guidance.get("bbb", {})
    ckpt = str(bbb.get("ckpt", ""))
    if ckpt and not Path(ckpt).is_absolute():
        ckpt = str((root / ckpt).resolve())

    return {
        "guidance_hotspot_weight": float(guidance.get("hotspot_weight", 0.0)),
        "guidance_atp_weight": float(guidance.get("atp_weight", 0.0)),
        "guidance_alpha": float(guidance.get("alpha", 8.0)),
        "guidance_cutoff_angstrom": float(guidance.get("cutoff_angstrom", 5.0)),
        "guidance_lj_sigma": float(guidance.get("lj_sigma", 3.0)),
        "guidance_max_force": float(guidance.get("max_force", 1.0)),
        "guidance_bbb_weight": float(bbb.get("weight", 0.0)),
        "guidance_membrane_weight": float(bbb.get("membrane_weight", 0.0)),
        "guidance_bbb_ckpt": ckpt,
        "guidance_bbb_sigma_gate": float(bbb.get("sigma_gate", 4.0)),
        "guidance_bbb_hidden": int(bbb.get("hidden", 64)),
        "guidance_bbb_layers": int(bbb.get("layers", 3)),
    }
