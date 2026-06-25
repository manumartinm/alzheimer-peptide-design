from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .enums import ProcessedArtifact


@dataclass
class PathLayout:
    base_dir: Path

    @property
    def raw_dir(self) -> Path:
        return self.base_dir / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.base_dir / "data" / "processed"

    @property
    def interim_dir(self) -> Path:
        return self.base_dir / "data" / "interim"

    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.processed_dir, self.interim_dir):
            path.mkdir(parents=True, exist_ok=True)

    def artifact(self, name: ProcessedArtifact) -> Path:
        return self.processed_dir / name.value

    @staticmethod
    def write_parquet(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    @staticmethod
    def read_parquet(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)
