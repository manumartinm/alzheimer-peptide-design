from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bbb_dataset.aa import CANONICAL_AA


@dataclass
class SequenceCleaner:
    min_length: int = 6
    max_length: int = 30
    identity_threshold: float = 0.9
    allowed_aa: set[str] | None = None

    def filter(
        self,
        df: pd.DataFrame,
        sequence_col: str = "sequence",
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        allowed = self.allowed_aa or CANONICAL_AA
        seqs = df[sequence_col].astype(str).str.upper()

        keep_len = seqs.str.len().between(self.min_length, self.max_length)
        keep_alpha = seqs.map(lambda s: set(s).issubset(allowed))
        keep = keep_len & keep_alpha

        stats = {
            "rows_before_filter": len(df),
            "rows_drop_length": int((~keep_len).sum()),
            "rows_drop_noncanonical": int((keep_len & ~keep_alpha).sum()),
            "rows_after_filter": int(keep.sum()),
        }
        out = df.loc[keep].copy()
        out[sequence_col] = seqs.loc[keep]
        return out.reset_index(drop=True), stats

    def resolve_conflicts(
        self,
        df: pd.DataFrame,
        sequence_col: str = "sequence",
        label_col: str = "bbb_label",
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        conflict_mask = (
            df.groupby(sequence_col)[label_col]
            .nunique()
            .reset_index(name="n")
            .query("n > 1")[sequence_col]
        )
        conflict_set = set(conflict_mask.tolist())
        out = df[~df[sequence_col].isin(conflict_set)].copy().reset_index(drop=True)
        stats = {
            "conflict_sequences_removed": len(conflict_set),
            "rows_after_conflict_resolution": len(out),
        }
        return out, stats

    def deduplicate(
        self,
        df: pd.DataFrame,
        sequence_col: str = "sequence",
        label_col: str = "bbb_label",
        keep_cluster_id: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        out = df.copy()
        before = len(out)

        out = out.drop_duplicates(subset=[sequence_col, label_col]).reset_index(drop=True)
        stats = {
            "rows_before_identity_dedup": int(before),
            "rows_after_exact_dedup": len(out),
        }

        seqs = out[sequence_col].astype(str).tolist()
        assignments = _run_cdhit_or_mmseqs(seqs, self.identity_threshold)
        if assignments is None:
            assignments = _cluster_python(seqs, self.identity_threshold)
        out["cluster_id"] = assignments

        out = (
            out.sort_values([label_col], ascending=False)
            .drop_duplicates(subset=["cluster_id"], keep="first")
            .reset_index(drop=True)
        )
        if not keep_cluster_id:
            out = out.drop(columns=["cluster_id"])
        stats["rows_after_identity_dedup"] = len(out)
        return out, stats

    def run(
        self,
        df: pd.DataFrame,
        sequence_col: str = "sequence",
        label_col: str = "bbb_label",
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        cleaned, stats_filter = self.filter(df, sequence_col=sequence_col)
        cleaned, stats_conflicts = self.resolve_conflicts(
            cleaned, sequence_col=sequence_col, label_col=label_col
        )
        cleaned, stats_dedup = self.deduplicate(
            cleaned, sequence_col=sequence_col, label_col=label_col
        )
        stats: dict[str, int] = {}
        stats.update(stats_filter)
        stats.update(stats_conflicts)
        stats.update(stats_dedup)
        return cleaned, stats


def filter_sequences(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    min_length: int = 6,
    max_length: int = 30,
    allowed_aa: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    return SequenceCleaner(
        min_length=min_length, max_length=max_length, allowed_aa=allowed_aa
    ).filter(df, sequence_col=sequence_col)


def resolve_label_conflicts(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    label_col: str = "bbb_label",
) -> tuple[pd.DataFrame, dict[str, int]]:
    return SequenceCleaner().resolve_conflicts(df, sequence_col=sequence_col, label_col=label_col)


def deduplicate_by_identity(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    label_col: str = "bbb_label",
    threshold: float = 0.9,
    keep_cluster_id: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    cleaner = SequenceCleaner(identity_threshold=threshold)
    return cleaner.deduplicate(
        df,
        sequence_col=sequence_col,
        label_col=label_col,
        keep_cluster_id=keep_cluster_id,
    )


def _sequence_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    max_len = max(len(a), len(b))
    min_len = min(len(a), len(b))
    matches = sum(1 for x, y in zip(a[:min_len], b[:min_len], strict=False) if x == y)
    return matches / max_len


def _cluster_python(seqs: list[str], threshold: float) -> list[int]:
    reps: list[str] = []
    assign: list[int] = []
    for seq in seqs:
        found = False
        for cid, rep in enumerate(reps):
            if _sequence_identity(seq, rep) >= threshold:
                assign.append(cid)
                found = True
                break
        if not found:
            reps.append(seq)
            assign.append(len(reps) - 1)
    return assign


def _parse_cdhit_clstr(path: Path, n_rows: int) -> list[int]:
    cluster_id = -1
    assign = [-1] * n_rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                cluster_id += 1
                continue
            marker = ">seq_"
            start = line.find(marker)
            if start < 0:
                continue
            rest = line[start + len(marker) :]
            end = rest.find("...")
            idx = int(rest[:end])
            assign[idx] = cluster_id
    if any(cid < 0 for cid in assign):
        return list(range(n_rows))
    return assign


def _parse_mmseqs_tsv(path: Path, n_rows: int) -> list[int]:
    import csv

    cluster_map: dict[str, int] = {}
    assign = [-1] * n_rows
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            rep, member = row[0], row[1]
            if rep not in cluster_map:
                cluster_map[rep] = len(cluster_map)
            cid = cluster_map[rep]
            if member.startswith("seq_"):
                idx = int(member.split("_", 1)[1])
                assign[idx] = cid
    next_cid = len(cluster_map)
    for i in range(n_rows):
        if assign[i] < 0:
            assign[i] = next_cid
            next_cid += 1
    return assign


def _run_cdhit_or_mmseqs(seqs: list[str], threshold: float) -> list[int] | None:
    with tempfile.TemporaryDirectory(prefix="bbb_cdhit_") as tmp:
        tmp_path = Path(tmp)
        fasta = tmp_path / "input.fa"
        with fasta.open("w", encoding="utf-8") as handle:
            for i, seq in enumerate(seqs):
                handle.write(f">seq_{i}\n{seq}\n")

        cdhit = shutil.which("cd-hit")
        if cdhit:
            out_fa = tmp_path / "out.fa"
            cmd = [
                cdhit,
                "-i",
                str(fasta),
                "-o",
                str(out_fa),
                "-c",
                str(threshold),
                "-n",
                "3",
                "-d",
                "0",
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                clstr = Path(f"{out_fa}.clstr")
                if clstr.exists():
                    return _parse_cdhit_clstr(clstr, len(seqs))
            except subprocess.CalledProcessError:
                pass

        mmseqs = shutil.which("mmseqs")
        if mmseqs:
            db = tmp_path / "db"
            clu = tmp_path / "clu"
            out_tsv = tmp_path / "clu.tsv"
            try:
                subprocess.run(
                    [mmseqs, "createdb", str(fasta), str(db)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    [
                        mmseqs,
                        "cluster",
                        str(db),
                        str(clu),
                        str(tmp_path / "tmp"),
                        "--min-seq-id",
                        str(threshold),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    [mmseqs, "createtsv", str(db), str(db), str(clu), str(out_tsv)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if out_tsv.exists():
                    return _parse_mmseqs_tsv(out_tsv, len(seqs))
            except subprocess.CalledProcessError:
                pass
    return None
