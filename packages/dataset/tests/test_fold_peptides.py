from __future__ import annotations

import json
from pathlib import Path

from tfg_bbb.folding import manifest_fields, parse_run_json


def test_parse_run_json_best_sample(tmp_path: Path) -> None:
    run_json = tmp_path / "run.json"
    run_json.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "output": {
                    "best_sample": {
                        "metrics": {
                            "structure_confidence": 0.91,
                            "ptm": 0.92,
                            "iptm": 0.86,
                            "complex_plddt": 0.95,
                            "complex_iplddt": 0.88,
                            "complex_pde": 1.2,
                            "complex_ipde": 1.8,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    metrics = parse_run_json(run_json)
    fields = manifest_fields(metrics)
    assert metrics["complex_plddt"] == 0.95
    assert fields["plddt"] == 95.0
    assert fields["ptm"] == 0.92
    assert fields["structure_confidence"] == 0.91
