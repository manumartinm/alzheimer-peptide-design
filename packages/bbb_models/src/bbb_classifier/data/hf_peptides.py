from __future__ import annotations

from pathlib import Path
from typing import Any

from bbb_classifier.utils.io import read_yaml

HF_DATASET_REPO = "manumartinm/bbb-peptides"
HF_ALLOW_PATTERNS = ("peptides.parquet", "structures/**", "stats.json", "README.md")

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = PACKAGE_ROOT / "data" / "bbb-peptides"


def cache_dir_from_config(data_cfg: dict[str, Any]) -> Path:
    root = data_cfg.get("dataset_root")
    if root:
        path = Path(str(root))
        if not path.is_absolute():
            path = (PACKAGE_ROOT / path).resolve()
        return path
    return DEFAULT_CACHE_DIR.resolve()


def resolve_data_paths(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *data_cfg* with dataset paths resolved under the package root."""
    out = dict(data_cfg)
    cache = cache_dir_from_config(out)

    dataset_path = out.get("dataset_path")
    if dataset_path:
        path = Path(str(dataset_path))
        if not path.is_absolute():
            path = (PACKAGE_ROOT / path).resolve()
        out["dataset_path"] = str(path)
    else:
        out["dataset_path"] = str(cache / "peptides.parquet")

    out["dataset_root"] = str(cache)
    manifest = out.get("struct_manifest_path")
    if manifest:
        manifest_path = Path(str(manifest))
        if not manifest_path.is_absolute():
            manifest_path = (PACKAGE_ROOT / manifest_path).resolve()
        out["struct_manifest_path"] = str(manifest_path)
    return out


def sync_hf_dataset(
    cache_dir: Path | None = None,
    *,
    repo_id: str = HF_DATASET_REPO,
    force: bool = False,
) -> Path:
    """Download or refresh the BBB peptides dataset from Hugging Face."""
    from huggingface_hub import snapshot_download

    target = (cache_dir or DEFAULT_CACHE_DIR).resolve()
    peptides = target / "peptides.parquet"
    if peptides.exists() and not force:
        return target

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=list(HF_ALLOW_PATTERNS),
    )
    if not peptides.exists():
        raise FileNotFoundError(
            f"Expected peptides.parquet in {target} after downloading {repo_id}"
        )
    return target


def load_data_config(path: str | Path, *, ensure: bool = False) -> dict[str, Any]:
    cfg = resolve_data_paths(read_yaml(path))
    if ensure and cfg.get("dataset_repo"):
        ensure_dataset(cfg)
    return cfg


def ensure_dataset(data_cfg: dict[str, Any], *, force: bool = False) -> Path:
    """Ensure the configured HF dataset is present locally; download if missing."""
    resolved = resolve_data_paths(data_cfg)
    cache = Path(resolved["dataset_root"])
    peptides = Path(resolved["dataset_path"])
    repo_id = str(data_cfg.get("dataset_repo") or HF_DATASET_REPO)
    if peptides.exists() and not force:
        return cache
    return sync_hf_dataset(cache, repo_id=repo_id, force=force)
