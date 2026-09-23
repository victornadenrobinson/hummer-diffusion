#!/usr/bin/env python
"""NPT equilibration of a methane box at a target (T, P) state point.

Uses ase.md.nptberendsen.NPTBerendsen: ASE's own Nose-Hoover/Parrinello-Rahman
NPT (ase.md.npt.NPT / MelchionnaNPT) documents itself as "not recommended due
to stability problems", so we use the weak-coupling Berendsen barostat here
instead. It is not rigorously canonical, but that's fine for equilibrating a
starting density/box before switching to fixed-volume production dynamics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mace_ch4.calculators import get_calculator  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Initial geometry (e.g. from build_initial_config.py)")
    p.add_argument("--output", required=True, help="Where to write the equilibrated structure")
    p.add_argument("--temperature-K", type=float, required=True)
    p.add_argument("--pressure-GPa", type=float, required=True)
    p.add_argument("--model", default="off23-medium")
    p.add_argument("--model-path", default=None, help="Explicit local checkpoint (required for off24-medium)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--timestep-fs", type=float, default=0.5)
    p.add_argument("--taut-fs", type=float, default=100.0, help="Berendsen temperature coupling time")
    p.add_argument("--taup-fs", type=float, default=1000.0, help="Berendsen pressure coupling time")
    p.add_argument("--compressibility-bar-inv", type=float, default=1e-4)
    p.add_argument("--fmax", type=float, default=5.0, help="Pre-relaxation force convergence (eV/A)")
    p.add_argument("--max-relax-steps", type=int, default=500)
    p.add_argument("--log", default=None, help="Path to write per-step diagnostics (npz)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    atoms = read(args.input)
    atoms.calc = get_calculator(
        model=args.model, device=args.device, model_path=args.model_path
    )

    # Remove any bad contacts from the initial packing before starting dynamics.
    relax = FIRE(atoms, logfile=None)
    relax.run(fmax=args.fmax, steps=args.max_relax_steps)
    max_force = np.abs(atoms.get_forces()).max()
    if max_force > args.fmax:
        raise RuntimeError(
            f"Pre-relaxation did not converge: max force {max_force:.2f} eV/A "
            f"> fmax {args.fmax:.2f} eV/A after {args.max_relax_steps} steps. "
            "The initial packing may be too dense/overlapping."
        )
    print(f"Pre-relaxation converged: max force {max_force:.3f} eV/A")

    rng = np.random.default_rng(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K, rng=rng)
    Stationary(atoms)

    pressure_au = args.pressure_GPa * 1e9 * units.Pascal
    compressibility_au = args.compressibility_bar_inv / (1e5 * units.Pascal)

    dyn = NPTBerendsen(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature_K,
        pressure_au=pressure_au,
        taut=args.taut_fs * units.fs,
        taup=args.taup_fs * units.fs,
        compressibility_au=compressibility_au,
    )

    history = {"step": [], "temperature_K": [], "density_kg_m3": [], "volume_A3": []}
    total_mass_amu = atoms.get_masses().sum()

    def record(a=atoms):
        step = dyn.nsteps
        temperature = a.get_temperature()
        volume_a3 = a.get_volume()
        # amu/A^3 -> kg/m^3
        density = (total_mass_amu / volume_a3) * 1.66053906660e3
        history["step"].append(step)
        history["temperature_K"].append(temperature)
        history["density_kg_m3"].append(density)
        history["volume_A3"].append(volume_a3)
        if step % 100 == 0:
            print(f"step={step} T={temperature:.1f}K density={density:.1f}kg/m^3")

    dyn.attach(record, interval=10)
    dyn.run(args.steps)

    # Sanity check: density should have plateaued over the last third of the run.
    n = len(history["density_kg_m3"])
    tail = np.array(history["density_kg_m3"][2 * n // 3 :])
    if tail.size > 1:
        relative_spread = tail.std() / tail.mean()
        if relative_spread > 0.05:
            print(
                f"WARNING: density has not clearly plateaued "
                f"(relative std over final third = {relative_spread:.1%}). "
                "Consider a longer equilibration.",
                file=sys.stderr,
            )

    write(args.output, atoms)
    print(f"Wrote equilibrated structure to {args.output}")

    if args.log:
        np.savez(args.log, **{k: np.array(v) for k, v in history.items()})
        print(f"Wrote diagnostics to {args.log}")


if __name__ == "__main__":
    main()
