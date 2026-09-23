#!/usr/bin/env python
"""NVE production MD with checkpoint/restart, unwrapped COM sampling, and safety guards.

Designed to be invoked repeatedly (e.g. chained across GitHub Actions jobs
capped at 6h each): if --checkpoint already exists, resumes from it; otherwise
starts fresh from --input. Always exits cleanly (checkpoint saved) rather than
leaving a run in an unrecoverable half-written state, whether it stops because
it hit --total-steps, ran out of --max-wall-seconds, or received SIGTERM.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read
from ase.md.velocitydistribution import Stationary
from ase.md.verlet import VelocityVerlet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mace_ch4.calculators import get_calculator  # noqa: E402
from mace_ch4.checkpoint import (  # noqa: E402
    append_com_trajectory,
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)
from mace_ch4.unwrap import UnwrappedCOMTracker, molecule_centers_of_mass  # noqa: E402

_STOP_REQUESTED = False


def _handle_signal(signum, _frame) -> None:
    global _STOP_REQUESTED
    print(f"Received signal {signum}; will checkpoint and stop at the next opportunity.")
    _STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Fixed-volume structure to start from (equilibrate_npt.py -> equilibrate_nvt.py output); used only on a fresh start")
    p.add_argument("--checkpoint", required=True, help="Resumed from if this file already exists")
    p.add_argument("--com-output", required=True, help="Growing file of sampled unwrapped molecule COM positions")
    p.add_argument("--model", default="off23-medium")
    p.add_argument("--model-path", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--atoms-per-molecule", type=int, default=5)
    p.add_argument("--timestep-fs", type=float, default=0.5)
    p.add_argument("--total-steps", type=int, required=True, help="Target step count for the whole (possibly multi-job) run")
    p.add_argument("--sample-interval", type=int, default=100)
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--max-wall-seconds", type=float, default=float("inf"))
    p.add_argument("--energy-drift-threshold", type=float, default=0.05, help="Relative |E-E0|/|E0| abort threshold")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    atoms = read(args.input)
    atoms.calc = get_calculator(model=args.model, device=args.device, model_path=args.model_path)

    dyn = VelocityVerlet(atoms, timestep=args.timestep_fs * units.fs)

    if checkpoint_exists(args.checkpoint):
        state = load_checkpoint(args.checkpoint)
        if state["done"]:
            print("Checkpoint already marked done; nothing to do.")
            return
        atoms.set_positions(state["positions"])
        atoms.set_velocities(state["velocities"])
        atoms.set_cell(state["cell"])
        step = state["step"]
        tracker = UnwrappedCOMTracker.from_state_dict(
            {
                "unwrapped": state["tracker_unwrapped"],
                "last_wrapped": state["tracker_last_wrapped"],
                "cell": state["tracker_cell"],
            }
        )
        initial_energy = state["initial_energy"]
        print(f"Resuming from checkpoint at step {step}")
    else:
        Stationary(atoms)
        step = 0
        wrapped_com = molecule_centers_of_mass(
            atoms.get_positions(wrap=True), atoms.get_masses(), args.atoms_per_molecule
        )
        tracker = UnwrappedCOMTracker(wrapped_com, np.array(atoms.get_cell()))
        initial_energy = atoms.get_potential_energy() + atoms.get_kinetic_energy()
        print(f"Starting fresh production run; E0={initial_energy:.4f} eV")

    pending_times: list[float] = []
    pending_com: list[np.ndarray] = []

    def do_checkpoint(done: bool) -> None:
        nonlocal pending_times, pending_com
        if pending_times:
            append_com_trajectory(
                args.com_output, np.array(pending_times), np.array(pending_com)
            )
            pending_times, pending_com = [], []
        tracker_state = tracker.state_dict()
        save_checkpoint(
            args.checkpoint,
            positions=atoms.get_positions(),
            velocities=atoms.get_velocities(),
            cell=np.array(atoms.get_cell()),
            step=step,
            tracker_unwrapped=tracker_state["unwrapped"],
            tracker_last_wrapped=tracker_state["last_wrapped"],
            tracker_cell=tracker_state["cell"],
            initial_energy=initial_energy,
            done=done,
        )

    steps_remaining = max(0, args.total_steps - step)
    start_wall = time.monotonic()

    for _ in range(steps_remaining):
        dyn.run(1)
        step += 1

        if step % args.sample_interval == 0:
            wrapped_com = molecule_centers_of_mass(
                atoms.get_positions(wrap=True), atoms.get_masses(), args.atoms_per_molecule
            )
            unwrapped = tracker.update(wrapped_com)
            pending_times.append(step * args.timestep_fs)
            pending_com.append(unwrapped.copy())

        if step % args.checkpoint_interval == 0:
            current_energy = atoms.get_potential_energy() + atoms.get_kinetic_energy()
            relative_drift = abs(current_energy - initial_energy) / max(abs(initial_energy), 1e-8)
            if relative_drift > args.energy_drift_threshold:
                do_checkpoint(done=False)
                raise RuntimeError(
                    f"Energy drift {relative_drift:.1%} exceeds threshold "
                    f"{args.energy_drift_threshold:.1%} at step {step}; aborting "
                    "(likely instability -- check timestep/pre-relaxation)."
                )
            do_checkpoint(done=False)
            print(f"step={step}/{args.total_steps} E_drift={relative_drift:.2%}")

        if _STOP_REQUESTED or (time.monotonic() - start_wall) > args.max_wall_seconds:
            do_checkpoint(done=False)
            print(f"Pausing at step {step} (wall-clock limit or signal); checkpoint saved for resume.")
            return

    do_checkpoint(done=True)
    print(f"Reached target step count {args.total_steps}; run complete.")


if __name__ == "__main__":
    main()
