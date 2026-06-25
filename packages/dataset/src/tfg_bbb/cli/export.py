#!/usr/bin/env python3
"""Export consolidated BBB peptide dataset for Hugging Face upload."""

from __future__ import annotations

import argparse

from tfg_bbb.export_hf import export_hf_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build HF-ready release: peptides + physicochemical features + Boltz structures."
    )
    parser.add_argument("--base-dir", default=".", help="Dataset root (default: current directory)")
    parser.add_argument(
        "--variant",
        choices=("gold", "full"),
        default="gold",
        help="gold = 452 curated peptides; full = gold + augmented (825 rows)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data/hf_release under base-dir)",
    )
    parser.add_argument(
        "--no-cif",
        action="store_true",
        help="Skip copying Boltz CIF files (coords.npz only)",
    )
    parser.add_argument(
        "--no-copy-structures",
        action="store_true",
        help="Only write parquet with relative paths; do not copy structure files",
    )
    args = parser.parse_args(argv)

    result = export_hf_dataset(
        base_dir=args.base_dir,
        variant=args.variant,
        output_dir=args.output_dir,
        include_cif=not args.no_cif,
        copy_structures=not args.no_copy_structures,
    )
    stats = result["stats"]
    print(f"Exported {stats['rows']} peptides ({stats['with_structure']} with structure)")
    print(f"Output: {stats['output_dir']}")
    print(f"Parquet: {stats['parquet_path']}")
    print("\nUpload to Hugging Face:")
    print("  huggingface-cli upload YOUR_USERNAME/bbb-peptides-tfg data/hf_release .")


if __name__ == "__main__":
    main()
