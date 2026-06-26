#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGES_ROOT = REPO_ROOT / "packages"
sys.path.insert(0, str(PACKAGES_ROOT))

from boltzgen_design.filtering.candidates import load_final_designs, normalize_candidate_frame
from boltzgen_design.filtering.gates import (
    ALL_GATES,
    GateThresholds,
    evaluate_gates,
    gate_thresholds_as_dict,
)
from boltzgen_design.filtering.isoform_metrics import compute_isoform_selectivity, load_isoform_map
from boltzgen_design.filtering.pareto import pareto_front
from boltzgen_design.filtering.struct_metrics import compute_struct_metrics, load_guidance
from boltzgen_design.scoring.bbb_oracle import BBBOracle, resolve_bbb_run_dir
from boltzgen_design.utils.paths import BBB_CLASSIFIER, BOLTZGEN_DESIGN, WORKBENCH

DEFAULT_GSK3A_CIF = BOLTZGEN_DESIGN / "targets" / "gsk3a" / "gsk3a.cif"
DEFAULT_ISOFORM_MAP = BOLTZGEN_DESIGN / "targets" / "gsk3a" / "isoform_map.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run G1-G6 cascade on BoltzGen final designs.")
    p.add_argument(
        "--campaign-dir",
        default=str(WORKBENCH / "gsk3b_guided"),
        help="BoltzGen campaign output directory (contains final_ranked_designs/)",
    )
    p.add_argument(
        "--input-dir",
        default="",
        help="Legacy alias: directory with metrics/CIFs (defaults to campaign-dir/final_ranked_designs)",
    )
    p.add_argument(
        "--output-csv",
        default="",
        help="Output CSV (default: <campaign-dir>/gated_final_designs.csv)",
    )
    p.add_argument("--bbb-repo", default=str(BBB_CLASSIFIER))
    p.add_argument(
        "--bbb-run-dir", default="", help="Classifier run dir with checkpoints/best.ckpt"
    )
    p.add_argument(
        "--skip-bbb", action="store_true", help="Skip BBB oracle (G3 will fail unless scores exist)"
    )
    p.add_argument(
        "--download-bbb", action="store_true", help="Download HF model if checkpoint missing"
    )
    p.add_argument("--guidance-json", default="", help="Path to guidance.json for G1/G2")
    p.add_argument(
        "--thresholds-yaml",
        default=str(BOLTZGEN_DESIGN / "configs" / "design_campaign.yaml"),
        help="YAML with thresholds.gate_* overrides",
    )
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument(
        "--require-gates",
        default="",
        help="Required gates, comma-separated: g1,g2,g3,g4,g5,g6 (default: all). Example: g2,g3,g6",
    )
    p.add_argument(
        "--gsk3a-cif",
        default=str(DEFAULT_GSK3A_CIF),
        help="GSK3α reference CIF for G6 isoform selectivity",
    )
    p.add_argument(
        "--isoform-map",
        default=str(DEFAULT_ISOFORM_MAP),
        help="β↔α residue map JSON for G6",
    )
    p.add_argument(
        "--rank-by",
        default="",
        choices=["", "iptm"],
        help="If set, output top-k among pass_all_gates sorted by this metric",
    )
    p.add_argument("--bbb-min", type=float, default=None, help="Override G3 p(BBB) threshold")
    p.add_argument(
        "--skip-g1-g2",
        action="store_true",
        help="Shortcut for --require-gates g3,g4,g5 (deprecated vs --require-gates)",
    )
    return p.parse_args()


def parse_require_gates(args: argparse.Namespace) -> frozenset[str]:
    if args.require_gates.strip():
        parts = {p.strip().lower() for p in args.require_gates.split(",") if p.strip()}
        unknown = parts - ALL_GATES
        if unknown:
            raise ValueError(f"Unknown gates: {unknown}. Use subset of {sorted(ALL_GATES)}")
        return frozenset(parts)
    if args.skip_g1_g2:
        return frozenset({"g3", "g4", "g5"})
    return ALL_GATES


def first_existing(df: pd.DataFrame, names: list[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def load_thresholds(path: Path) -> GateThresholds:
    if not path.exists():
        return GateThresholds()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return GateThresholds.from_mapping(data.get("thresholds", {}))


def normalize_plddt(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return values.map(lambda v: v * 100.0 if v <= 1.5 else v)


def attach_struct_metrics(df: pd.DataFrame, guidance_path: Path | None) -> pd.DataFrame:
    guidance = load_guidance(guidance_path) if guidance_path else load_guidance()
    rows: list[dict] = []
    for _, row in df.iterrows():
        cif = Path(str(row["structure_path"]))
        if not cif.exists():
            rows.append(
                {"hotspot_fraction": 0.0, "atp_repulsion": 1e9, "closure_rmsd": float("nan")}
            )
            continue
        try:
            rows.append(compute_struct_metrics(cif, guidance))
        except Exception as exc:
            print(f"WARN struct metrics failed for {cif.name}: {exc}", file=sys.stderr)
            rows.append(
                {"hotspot_fraction": 0.0, "atp_repulsion": 1e9, "closure_rmsd": float("nan")}
            )
    struct_df = pd.DataFrame(rows)
    for col in struct_df.columns:
        df[col] = struct_df[col]
    return df


def attach_isoform_metrics(
    df: pd.DataFrame,
    gsk3a_cif: Path,
    isoform_map_path: Path,
) -> pd.DataFrame:
    isoform_map = load_isoform_map(isoform_map_path)
    rows: list[dict] = []
    for _, row in df.iterrows():
        cif = Path(str(row["structure_path"]))
        if not cif.exists():
            rows.append(
                {
                    "selectivity_margin": float("nan"),
                    "alpha_contact_fraction": float("nan"),
                    "beta_contact_fraction": float("nan"),
                    "n_isoform_interface": 0.0,
                    "n_beta_favored": 0.0,
                }
            )
            continue
        try:
            rows.append(
                compute_isoform_selectivity(
                    cif,
                    gsk3a_cif,
                    isoform_map,
                )
            )
        except Exception as exc:
            print(f"WARN isoform metrics failed for {cif.name}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "selectivity_margin": float("nan"),
                    "alpha_contact_fraction": float("nan"),
                    "beta_contact_fraction": float("nan"),
                    "n_isoform_interface": 0.0,
                    "n_beta_favored": 0.0,
                }
            )
    iso_df = pd.DataFrame(rows)
    for col in iso_df.columns:
        df[col] = iso_df[col]
    return df


def attach_bbb_scores(df: pd.DataFrame, args: argparse.Namespace, output_csv: Path) -> pd.DataFrame:
    if args.skip_bbb:
        df["bbb_probability"] = float("nan")
        df["bbb_status"] = "skipped"
        return df

    run_dir = resolve_bbb_run_dir(
        Path(args.bbb_run_dir) if args.bbb_run_dir else None,
        download=args.download_bbb,
    )
    if run_dir is None:
        print(
            "WARN BBB checkpoint not found; use --download-bbb or --bbb-run-dir. G3 will fail.",
            file=sys.stderr,
        )
        df["bbb_probability"] = float("nan")
        df["bbb_status"] = "missing_model"
        return df

    sys.path.insert(0, str(PACKAGES_ROOT / "dataset" / "src"))
    from bbb_dataset.features import add_feature_columns

    feat_df = add_feature_columns(df[["sequence"]].copy(), sequence_col="sequence")
    if "length" not in feat_df.columns:
        feat_df["length"] = feat_df["sequence"].str.len()

    bbb_scores_file = output_csv.with_name(f"{output_csv.stem}.bbb.csv")
    bbb_input = bbb_scores_file.with_name(f"{output_csv.stem}.bbb.input.csv")
    feat_df.to_csv(bbb_input, index=False)

    try:
        oracle = BBBOracle(bbb_repo_root=Path(args.bbb_repo), run_dir=run_dir)
        oracle._run_predict(bbb_input, bbb_scores_file)
        bbb_df = pd.read_csv(bbb_scores_file)
    except Exception as exc:
        print(f"WARN BBB scoring failed: {exc}", file=sys.stderr)
        df["bbb_probability"] = float("nan")
        df["bbb_status"] = "error"
        return df

    prob_col = (
        "p_bbb_calibrated"
        if "p_bbb_calibrated" in bbb_df.columns
        else ("p_bbb_raw" if "p_bbb_raw" in bbb_df.columns else "probability")
    )
    df["bbb_probability"] = pd.to_numeric(bbb_df[prob_col], errors="coerce")
    df["bbb_status"] = "ok"
    return df


def liability_pass(row: pd.Series) -> bool:
    seq = str(row.get("sequence", "") or row.get("designed_sequence", ""))
    if not seq:
        return False
    if "KKKK" in seq or "RRRR" in seq:
        return False
    if "liability_num_violations" in row.index:
        violations = pd.to_numeric(row["liability_num_violations"], errors="coerce")
        if pd.notna(violations) and int(violations) > 0:
            return False
    if "liability_high_severity_violations" in row.index:
        high = pd.to_numeric(row["liability_high_severity_violations"], errors="coerce")
        if pd.notna(high) and int(high) > 0:
            return False
    return True


def build_candidate_rows(
    df: pd.DataFrame,
    cfg: GateThresholds,
    *,
    require: frozenset[str] = ALL_GATES,
) -> pd.DataFrame:
    df = df.copy()
    df["iptm"] = first_existing(df, ["design_to_target_iptm", "design_iptm", "iptm"], default=0.0)
    df["plddt"] = normalize_plddt(
        first_existing(
            df, ["complex_plddt", "complex_iplddt", "design_plddt", "plddt"], default=0.0
        )
    )
    if "closure_rmsd" not in df.columns or df["closure_rmsd"].isna().all():
        df["closure_rmsd"] = first_existing(
            df, ["closure_rmsd", "bb_rmsd", "filter_rmsd"], default=1e9
        )
    df["passes_sequence_liability"] = df.apply(liability_pass, axis=1)

    records: list[dict] = []
    for row in df.to_dict(orient="records"):
        flags = evaluate_gates(row, cfg, require=require)
        enriched = {**row, **flags}
        records.append(enriched)
    return pd.DataFrame(records)


def print_gate_summary(df: pd.DataFrame, *, require: frozenset[str]) -> None:
    cols = [
        "pass_g1_hotspot",
        "pass_g2_atp",
        "pass_g3_bbb",
        "pass_g4_quality",
        "pass_g5_liability",
        "pass_g6_selectivity",
        "pass_all_gates",
    ]
    print(f"\nRequired gates: {','.join(sorted(require))}")
    print("Gate pass counts:")
    for col in cols:
        if col in df.columns:
            print(f"  {col}: {int(df[col].sum())}/{len(df)}")
    if "pass_all_gates" in df.columns and df["pass_all_gates"].any():
        print(f"\nCandidates passing required gates ({len(df[df['pass_all_gates']])}):")
        show_cols = [
            "id",
            "final_rank",
            "designed_sequence",
            "hotspot_fraction",
            "atp_repulsion",
            "bbb_probability",
            "selectivity_margin",
            "alpha_contact_fraction",
            "iptm",
            "plddt",
        ]
        show = df[df["pass_all_gates"]].sort_values("iptm", ascending=False)[
            [c for c in show_cols if c in df.columns]
        ]
        print(show.head(15).to_string(index=False))
    else:
        print("\nNo candidates pass the required gates with current thresholds.")


def main() -> None:
    args = parse_args()
    campaign_dir = Path(args.campaign_dir).resolve()
    input_dir = (
        Path(args.input_dir).resolve() if args.input_dir else campaign_dir / "final_ranked_designs"
    )
    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else campaign_dir / "gated_final_designs.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    guidance_path = Path(args.guidance_json).resolve() if args.guidance_json else None
    thresholds = load_thresholds(Path(args.thresholds_yaml))
    if args.bbb_min is not None:
        thresholds.g3_bbb_min = args.bbb_min
    require = parse_require_gates(args)

    try:
        df = load_final_designs(campaign_dir)
    except FileNotFoundError:
        metrics_files = sorted(input_dir.rglob("aggregate_metrics*.csv"))
        if metrics_files:
            df = pd.read_csv(metrics_files[0])
        else:
            raise

    df = normalize_candidate_frame(df, input_dir)
    missing_cif = [p for p in df["structure_path"] if not Path(p).exists()]
    if missing_cif:
        print(
            f"WARN {len(missing_cif)} CIF paths missing (first: {missing_cif[0]})", file=sys.stderr
        )

    if "g1" in require or "g2" in require:
        df = attach_struct_metrics(df, guidance_path)
    elif "closure_rmsd" not in df.columns:
        df["closure_rmsd"] = first_existing(
            df, ["closure_rmsd", "bb_rmsd", "filter_rmsd"], default=float("nan")
        )
        df["hotspot_fraction"] = float("nan")
        df["atp_repulsion"] = float("nan")
    if "g6" in require:
        gsk3a_cif = Path(args.gsk3a_cif).resolve()
        isoform_map_path = Path(args.isoform_map).resolve()
        if not gsk3a_cif.exists():
            raise FileNotFoundError(
                f"GSK3α CIF not found: {gsk3a_cif}. Run boltzgen_design/scripts/build_gsk3a_target.py"
            )
        df = attach_isoform_metrics(df, gsk3a_cif, isoform_map_path)
    df = attach_bbb_scores(df, args, output_csv)
    result = build_candidate_rows(df, thresholds, require=require)

    passed = result[result["pass_all_gates"]].copy()
    wrote_output = False
    if args.rank_by == "iptm":
        if len(passed) > 0:
            selected = passed.sort_values("iptm", ascending=False).head(args.top_k)
            selected.to_csv(output_csv, index=False)
            print(
                f"Saved {len(selected)} candidates (required gates pass, ranked by iPTM) to {output_csv}"
            )
        else:
            print("No candidates pass the required gates.")
            passed.head(0).to_csv(output_csv, index=False)
        wrote_output = True
    elif len(passed) > 0:
        gated = passed
        gated["pareto_hotspot"] = pd.to_numeric(gated["hotspot_fraction"], errors="coerce").fillna(
            0.0
        )
        gated["pareto_bbb"] = pd.to_numeric(gated["bbb_probability"], errors="coerce").fillna(0.0)
        gated["pareto_iptm"] = pd.to_numeric(gated["iptm"], errors="coerce").fillna(0.0)
        gated["pareto_atp_avoidance"] = -pd.to_numeric(
            gated["atp_repulsion"], errors="coerce"
        ).fillna(1e9)
        front = pareto_front(
            gated.to_dict(orient="records"),
            keys=["pareto_hotspot", "pareto_atp_avoidance", "pareto_bbb", "pareto_iptm"],
        )
        pareto_df = pd.DataFrame(front)
        pareto_df.head(args.top_k).to_csv(output_csv, index=False)
        print(f"Saved {min(args.top_k, len(pareto_df))} Pareto candidates to {output_csv}")
        wrote_output = True

    report_cols = [
        "final_rank",
        "id",
        "designed_sequence",
        "structure_path",
        "hotspot_fraction",
        "atp_repulsion",
        "bbb_probability",
        "bbb_status",
        "iptm",
        "plddt",
        "closure_rmsd",
        "passes_sequence_liability",
        "pass_g1_hotspot",
        "pass_g2_atp",
        "pass_g3_bbb",
        "pass_g4_quality",
        "pass_g5_liability",
        "selectivity_margin",
        "alpha_contact_fraction",
        "beta_contact_fraction",
        "n_isoform_interface",
        "pass_g6_selectivity",
        "pass_all_gates",
        "design_to_target_iptm",
        "min_design_to_target_pae",
        "delta_sasa_refolded",
        "plip_hbonds_refolded",
        "liability_num_violations",
        "quality_score",
    ]
    keep = [c for c in report_cols if c in result.columns]
    result.sort_values(["pass_all_gates", "iptm"], ascending=[False, False])[keep].to_csv(
        output_csv.with_name(f"{output_csv.stem}_report.csv"), index=False
    )

    if not wrote_output:
        fallback = result.sort_values(
            by=["pass_g3_bbb", "pass_g2_atp", "pass_g1_hotspot", "iptm"],
            ascending=[False, False, False, False],
        ).head(args.top_k)
        fallback.to_csv(output_csv, index=False)
        print(
            f"No required-gate passers; saved top-{len(fallback)} fallback ranking to {output_csv}"
        )

    print_gate_summary(result, require=require)
    print("\nThresholds:", gate_thresholds_as_dict(thresholds))
    print(f"Full report: {output_csv.with_name(output_csv.stem + '_report.csv')}")


if __name__ == "__main__":
    main()
