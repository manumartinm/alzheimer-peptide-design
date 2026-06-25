#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from boltzgen.molecular_dynamics.membrane_simulation import MembraneSimulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OpenMM MD validation for top candidates.")
    p.add_argument("--pdb", required=True, help="Prepared complex PDB path")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=5000000, help="MD steps")
    p.add_argument("--equilibration-steps", type=int, default=500000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sim = MembraneSimulation(system={})
    sim.prepare_openmm(pdb_path=Path(args.pdb))
    result = sim.run_simulation(
        n_steps=args.steps,
        equilibration_steps=args.equilibration_steps,
        output_dir=Path(args.output_dir),
    )
    print(result)


if __name__ == "__main__":
    main()

