from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Iterable


class BBBOracle:
    """Wrapper over TFG/bbb_models prediction script."""

    def __init__(self, bbb_repo_root: Path, run_dir: Path, manifest_path: Path | None = None):
        self.bbb_repo_root = bbb_repo_root
        self.run_dir = run_dir
        self.manifest_path = manifest_path

    def _predict_script(self) -> str:
        meta_path = self.run_dir / "train_metadata.json"
        if meta_path.exists():
            model_type = json.loads(meta_path.read_text(encoding="utf-8"))["exp_cfg"]["model_type"]
            if model_type in {"struct_egnn_geo", "struct_egnn_full"}:
                return "scripts/geo/predict.py"
        return "scripts/classifier/predict.py"

    def _run_predict(self, input_csv: Path, output_csv: Path) -> None:
        command = [
            "python",
            self._predict_script(),
            "--run-dir",
            str(self.run_dir),
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
        ]
        if self.manifest_path is not None:
            command.extend(["--manifest", str(self.manifest_path)])
        subprocess.run(command, cwd=self.bbb_repo_root, check=True)

    def score_sequences(self, sequences: Iterable[str], output_csv: Path) -> Path:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        input_csv = output_csv.with_name(f"{output_csv.stem}.input.csv")

        with input_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["sequence"])
            writer.writeheader()
            for sequence in sequences:
                writer.writerow({"sequence": sequence})

        self._run_predict(input_csv, output_csv)
        return output_csv

    def score_candidates(self, rows: list[dict], output_csv: Path) -> Path:
        """Score candidates with optional coords_path for structural oracle models."""
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        input_csv = output_csv.with_name(f"{output_csv.stem}.input.csv")
        fieldnames = ["sequence"]
        if any("coords_path" in row for row in rows):
            fieldnames.append("coords_path")
        with input_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                payload = {"sequence": row.get("sequence", "")}
                if "coords_path" in row:
                    payload["coords_path"] = row["coords_path"]
                writer.writerow(payload)
        self._run_predict(input_csv, output_csv)
        return output_csv

