from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGES_ROOT = REPO_ROOT / "packages"
BBB_MODELS = PACKAGES_ROOT / "bbb_models"
BBB_CLASSIFIER = BBB_MODELS  # backward-compatible alias for scripts/oracle cwd
BOLTZGEN = PACKAGES_ROOT / "boltzgen"
BOLTZGEN_DESIGN = PACKAGES_ROOT / "boltzgen_design"
WORKBENCH = BOLTZGEN / "workbench"
