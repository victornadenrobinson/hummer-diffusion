#!/usr/bin/env python
"""Build an initial periodic box of methane molecules (pure geometry, no calculator)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mace_ch4.geometry import (  # noqa: E402
    build_methane_box,
    guess_initial_density_kg_m3,
    min_pairwise_distance,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-molecules", type=int, required=True)
    p.add_argument("--pressure-GPa", type=float, required=True, help="Used only for the initial density guess")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    density = guess_initial_density_kg_m3(args.pressure_GPa)
    atoms = build_methane_box(args.n_molecules, density, seed=args.seed)

    min_dist = min_pairwise_distance(atoms)
    print(
        f"Built {args.n_molecules} CH4 molecules ({len(atoms)} atoms), "
        f"box length={atoms.cell.lengths()[0]:.2f} A, "
        f"guessed density={density:.1f} kg/m^3, min pairwise distance={min_dist:.2f} A"
    )
    if min_dist < 0.3:
        raise RuntimeError(
            f"Packing produced near-overlapping atoms (min distance {min_dist:.3f} A); "
            "reduce density or increase box size."
        )

    write(args.output, atoms)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
