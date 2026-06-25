from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from boltzgen_design.guidance.feats_builder import guidance_config_to_feats


def test_guidance_config_to_feats_resolves_ckpt() -> None:
    guidance = {
        "hotspot_weight": 1.0,
        "atp_weight": 0.5,
        "bbb": {"weight": 0.2, "membrane_weight": 0.8, "ckpt": "models/best.ckpt"},
    }
    feats = guidance_config_to_feats(guidance, target_root=Path("/tmp/target"))
    assert feats["guidance_bbb_weight"] == 0.2
    assert feats["guidance_membrane_weight"] == 0.8
    assert feats["guidance_bbb_ckpt"] == str(Path("/tmp/target/models/best.ckpt").resolve())
