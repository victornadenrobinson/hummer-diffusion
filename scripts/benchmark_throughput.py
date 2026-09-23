#!/usr/bin/env python
"""Time MACE-OFF MD throughput for a given system size and append the result.

Builds a quick methane box in-process (no NPT equilibration -- throughput is
dominated by n_atoms and the calculator, not the exact geometry), runs a
short warmup + timed NVE segment, and appends steps/sec and ns/day to a
growing JSON results file (see scripts/plot_benchmarks.py to visualize it).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from ase.optimize import FIRE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mace_ch4.benchmark import append_benchmark_result, run_benchmark  # noqa: E402
from mace_ch4.calculators import get_calculator  # noqa: E402
from mace_ch4.geometry import build_methane_box, guess_initial_density_kg_m3  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-molecules", type=int, required=True)
    p.add_argument("--pressure-GPa", type=float, default=0.2, help="Only used for the initial density guess")
    p.add_argument("--model", default="off23-medium")
    p.add_argument("--model-path", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--timestep-fs", type=float, default=0.5)
    p.add_argument("--n-warmup-steps", type=int, default=20)
    p.add_argument("--n-timed-steps", type=int, default=100)
    p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads() for CPU core-scaling benchmarks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runner-label", default="local", help="Free-text tag, e.g. 'gh-actions-ubuntu' or a hostname")
    p.add_argument("--out", required=True, help="JSON results file to append to")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads is not None:
        import torch

        torch.set_num_threads(args.threads)

    density = guess_initial_density_kg_m3(args.pressure_GPa)
    atoms = build_methane_box(args.n_molecules, density, seed=args.seed)
    atoms.calc = get_calculator(model=args.model, device=args.device, model_path=args.model_path)

    # Remove bad contacts from the packing so the timed segment doesn't
    # explode/slow down from huge forces; this relaxation itself isn't timed.
    FIRE(atoms, logfile=None).run(fmax=5.0, steps=200)

    result = run_benchmark(
        atoms,
        timestep_fs=args.timestep_fs,
        n_warmup_steps=args.n_warmup_steps,
        n_timed_steps=args.n_timed_steps,
    )
    result.update(
        {
            "n_molecules": args.n_molecules,
            "model": args.model,
            "device": args.device,
            "threads": args.threads,
            "runner_label": args.runner_label,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    print(
        f"n_molecules={args.n_molecules} n_atoms={result['n_atoms']} "
        f"steps/s={result['steps_per_second']:.2f} ns/day={result['ns_per_day']:.4f}"
    )
    append_benchmark_result(args.out, result)
    print(f"Appended benchmark result to {args.out}")


if __name__ == "__main__":
    main()
