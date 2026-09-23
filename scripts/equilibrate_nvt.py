#!/usr/bin/env python
"""NVT re-equilibration at fixed volume, between NPT equilibration and NVE production.

Why this stage exists: switching straight from a fluctuating-cell NPT run to
fixed-volume NVE production skips re-thermalizing at that exact fixed volume,
and throws away a free consistency check. Running NVT at the volume NPT
settled on lets us (a) re-equilibrate velocities at fixed V before the
thermostat-free NVE run used for diffusion, and (b) measure the time-averaged
pressure at that (T, V) -- which should match the NPT target pressure, and
is itself one point of an equation of state (see mace_ch4.eos).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mace_ch4.calculators import get_calculator  # noqa: E402
from mace_ch4.eos import append_eos_point, pressure_GPa  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Output of equilibrate_npt.py")
    p.add_argument("--output", required=True, help="Feed this into run_production.py")
    p.add_argument("--temperature-K", type=float, required=True)
    p.add_argument("--pressure-GPa", type=float, required=True, help="Target pressure this state point was equilibrated at")
    p.add_argument("--model", default="off23-medium")
    p.add_argument("--model-path", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--atoms-per-molecule", type=int, default=5)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--timestep-fs", type=float, default=0.5)
    p.add_argument("--taut-fs", type=float, default=100.0)
    p.add_argument("--sample-interval", type=int, default=10)
    p.add_argument("--burn-in-fraction", type=float, default=0.3, help="Fraction of samples discarded before averaging pressure")
    p.add_argument("--eos-log", default=None, help="Append the measured (T, V, P) point here")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    atoms = read(args.input)
    atoms.calc = get_calculator(model=args.model, device=args.device, model_path=args.model_path)

    rng = np.random.default_rng(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K, rng=rng)
    Stationary(atoms)

    dyn = NVTBerendsen(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature_K,
        taut=args.taut_fs * units.fs,
    )

    pressures_GPa: list[float] = []
    temperatures_K: list[float] = []

    def record(a=atoms):
        pressures_GPa.append(pressure_GPa(a))
        temperatures_K.append(a.get_temperature())

    dyn.attach(record, interval=args.sample_interval)
    dyn.run(args.steps)

    burn_in = int(args.burn_in_fraction * len(pressures_GPa))
    tail_pressure = np.array(pressures_GPa[burn_in:])
    tail_temperature = np.array(temperatures_K[burn_in:])
    if tail_pressure.size < 2:
        raise RuntimeError("Too few samples after burn-in to estimate pressure; run more --steps")

    p_mean = float(tail_pressure.mean())
    p_stderr = float(tail_pressure.std(ddof=1) / np.sqrt(tail_pressure.size))
    t_mean = float(tail_temperature.mean())
    print(f"NVT equilibration done: <T>={t_mean:.1f}K <P>={p_mean:.4f}+-{p_stderr:.4f}GPa (target P={args.pressure_GPa:.4f}GPa)")

    relative_p_error = abs(p_mean - args.pressure_GPa) / max(abs(args.pressure_GPa), 1e-3)
    if relative_p_error > 0.5:
        print(
            f"WARNING: measured pressure differs from the target by {relative_p_error:.0%}. "
            "The NPT-equilibrated volume may not correspond well to the target state point "
            "(e.g. NPT run too short) -- inspect before trusting the production run.",
            file=sys.stderr,
        )

    write(args.output, atoms)
    print(f"Wrote NVT-thermalized structure to {args.output}")

    if args.eos_log:
        volume_A3 = atoms.get_volume()
        density_kg_m3 = (atoms.get_masses().sum() / volume_A3) * 1.66053906660e3
        append_eos_point(
            args.eos_log,
            {
                "n_molecules": len(atoms) // args.atoms_per_molecule,
                "temperature_K_target": args.temperature_K,
                "temperature_K_measured": t_mean,
                "pressure_GPa_target": args.pressure_GPa,
                "pressure_GPa_measured": p_mean,
                "pressure_GPa_stderr": p_stderr,
                "volume_A3": volume_A3,
                "density_kg_m3": density_kg_m3,
            },
        )
        print(f"Appended EOS point to {args.eos_log}")


if __name__ == "__main__":
    main()
