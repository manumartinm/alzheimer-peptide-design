#!/usr/bin/env python3
"""Build HF-ready model bundles for BBB classifier and geo EGNN."""

from __future__ import annotations

import argparse

from bbb_classifier.hf_export import export_classifier_hf, export_geo_hf


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export BBB models for Hugging Face upload.")
    parser.add_argument(
        "--kind",
        choices=("classifier", "geo", "both"),
        default="both",
        help="Which model bundle to export",
    )
    parser.add_argument("--base-dir", default=".", help="bbb_models root")
    parser.add_argument("--classifier-run", default=None, help="Run name under artifacts/models/")
    parser.add_argument("--geo-run", default=None, help="Run name under artifacts/models/")
    parser.add_argument("--classifier-out", default=None, help="Output dir for classifier bundle")
    parser.add_argument("--geo-out", default=None, help="Output dir for geo bundle")
    parser.add_argument("--classifier-repo", default="manumartinm/bbb-classifier")
    parser.add_argument("--geo-repo", default="manumartinm/bbb-geo")
    args = parser.parse_args(argv)

    if args.kind in {"classifier", "both"}:
        stats = export_classifier_hf(
            base_dir=args.base_dir,
            run_name=args.classifier_run,
            output_dir=args.classifier_out,
            repo_id=args.classifier_repo,
        )
        print(f"Classifier exported: {stats['run_name']} -> {stats['output_dir']}")
        print(f"  hf upload {args.classifier_repo} {stats['output_dir']} . --repo-type model")

    if args.kind in {"geo", "both"}:
        stats = export_geo_hf(
            base_dir=args.base_dir,
            run_name=args.geo_run,
            output_dir=args.geo_out,
            repo_id=args.geo_repo,
        )
        print(f"Geo exported: {stats['run_name']} -> {stats['output_dir']}")
        print(f"  hf upload {args.geo_repo} {stats['output_dir']} . --repo-type model")


if __name__ == "__main__":
    main()
