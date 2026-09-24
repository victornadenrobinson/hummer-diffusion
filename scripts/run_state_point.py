#!/usr/bin/env python
"""One (T, P) state point end to end: volume -> NVT -> NVE -> D_PBC.

One job per pressure: pass --pressure-GPa; the starting density follows
from the experimental reference EOS (mace_ch4.eos.reference_density_g_cm3).

Stages, all at --timestep-fs:
  1. volume  the model's own density at the target P (MACE's will differ
             from NIST's), one of two ways:
             * default: build a lattice box at the reference density, short
               fixed-cell FIRE, then NPT (Berendsen T + isotropic P) and
               rescale to the mean volume over the last
               --npt-average-fraction of it, moving whole molecules.
             * --eos-fit FILE: invert a P(rho) fit from scripts/eos_scan.py
               and build the box directly at that density (no NPT).
  3. NVT     Langevin at that fixed volume: re-thermalises after the rescale,
             and its mean pressure checks the volume really gives target P.
  4. NVE     Velocity Verlet, no thermostat -- the trajectory D comes from.
             Full extxyz every --traj-interval steps, unwrapped molecule
             COMs every --com-interval steps (nve_com.npz, the format
             scripts/compute_msd_diffusion.py reads).
  5. D_PBC   multi-origin COM MSD fit + block-average error, in summary.json.
             This is the finite-box value; Yeh-Hummer needs >= 2 box sizes
             (scripts/finite_size_correction.py).

NPT/NVT write only their first and last frames; production_volume.extxyz
is the fixed-volume start for NVT. Every stage writes a
thermo log (step, time, T, U, KE, E_tot, E_tot/molecule, drift, P, V,
density, steps/s) every --thermo-interval steps.

Restarting: finished stages are recorded in summary.json and skipped on
rerun. An interrupted NVE restarts from the start of NVE (its partial
outputs are discarded), so size --nve-steps to fit in one job.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import Stationary
from ase.md.verlet import VelocityVerlet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mace_ch4.calculators import get_calculator  # noqa: E402
from mace_ch4.eos import density_at_pressure, reference_density_g_cm3  # noqa: E402
from mace_ch4.geometry import ATOMS_PER_MOLECULE  # noqa: E402
from mace_ch4.mdtools import (  # noqa: E402
    Summary,
    ThermoLog,
    block_mean_stderr,
    build_relaxed_box,
    log,
    rescale_volume_by_molecule,
    snapshot,
    tail_stats,
)
from mace_ch4.msd import block_average_diffusion, ensemble_msd, fit_diffusion_coefficient  # noqa: E402
from mace_ch4.unwrap import UnwrappedCOMTracker, molecule_centers_of_mass  # noqa: E402

A2_FS_TO_M2_S = 1e-5


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pressure-GPa", type=float, required=True)
    p.add_argument("--temperature-K", type=float, default=450.0)
    p.add_argument("--n-molecules", type=int, default=128)
    p.add_argument("--initial-density-g-cm3", type=float, default=None,
                   help="Starting density for NPT; defaults to the reference (NIST) EOS value")
    p.add_argument("--eos-fit", default=None,
                   help="eos_fit.json from scripts/eos_scan.py: take the volume from it and skip NPT")
    p.add_argument("--outdir", required=True)
    p.add_argument("--model", default="off23-medium")
    p.add_argument("--model-path", default=None, help="Local checkpoint (required for off24-medium)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep-fs", type=float, default=0.5)
    p.add_argument("--npt-steps", type=int, default=100_000, help="Default 50 ps at 0.5 fs")
    p.add_argument("--nvt-steps", type=int, default=40_000, help="Default 20 ps at 0.5 fs")
    p.add_argument("--nve-steps", type=int, default=400_000, help="Default 200 ps at 0.5 fs")
    p.add_argument("--taut-fs", type=float, default=100.0, help="NPT Berendsen thermostat time")
    p.add_argument("--taup-fs", type=float, default=1000.0, help="NPT Berendsen barostat time")
    p.add_argument("--compressibility-per-GPa", type=float, default=1.0,
                   help="Barostat compressibility guess; only sets how fast the box responds")
    p.add_argument("--npt-average-fraction", type=float, default=0.5,
                   help="Final fraction of NPT averaged for the production volume")
    p.add_argument("--langevin-friction-per-fs", type=float, default=0.01)
    p.add_argument("--thermo-interval", type=int, default=100, help="Steps between thermo-log rows")
    p.add_argument("--print-interval", type=int, default=1000, help="Steps between stdout lines")
    p.add_argument("--traj-interval", type=int, default=100, help="Steps between NVE extxyz frames")
    p.add_argument("--com-interval", type=int, default=10, help="Steps between NVE COM samples")
    p.add_argument("--com-flush-interval", type=int, default=50_000, help="Steps between nve_com.npz rewrites")
    p.add_argument("--max-drift-meV-per-mol", type=float, default=25.0,
                   help="Abort NVE if |E - E0| per molecule exceeds this")
    p.add_argument("--msd-fit-start", type=float, default=0.1)
    p.add_argument("--msd-fit-end", type=float, default=0.5)
    p.add_argument("--msd-blocks", type=int, default=5)
    return p.parse_args(argv)


def reference_density(args) -> float:
    density = args.initial_density_g_cm3 or reference_density_g_cm3(args.temperature_K, args.pressure_GPa)
    if density is None:
        raise SystemExit("No reference density for this (T, P) (install CoolProp, or pass --initial-density-g-cm3)")
    return density


def run_volume_from_eos(args, calc, outdir: Path, summary: Summary) -> None:
    fit = json.loads(Path(args.eos_fit).read_text())
    if abs(fit["temperature_K"] - args.temperature_K) > 1e-6:
        raise SystemExit(f"EOS fit is for {fit['temperature_K']} K, not {args.temperature_K} K")
    if fit.get("model") != args.model:
        log(f"[volume] WARNING: EOS fit was made with model {fit.get('model')!r}, running {args.model!r}")
    density = density_at_pressure(np.array(fit["coeffs"]), args.pressure_GPa, tuple(fit["density_range_g_cm3"]))
    log(f"[volume] EOS fit {args.eos_fit}: rho({args.pressure_GPa} GPa) = {density:.5f} g/cm3")
    atoms = build_relaxed_box(args.n_molecules, density, args.temperature_K, calc, seed=args.seed)
    write(outdir / "production_volume.extxyz", snapshot(atoms))
    ref = reference_density_g_cm3(args.temperature_K, args.pressure_GPa)
    summary.set("volume", {
        "source": f"eos_fit:{args.eos_fit}", "density_g_cm3": density, "reference_density_g_cm3": ref,
        "density_vs_reference_percent": 100 * (density / ref - 1) if ref else None,
        "box_length_A": float(atoms.cell.lengths()[0]), "volume_A3": atoms.get_volume(),
    })


def run_npt(args, calc, outdir: Path, summary: Summary, header: dict) -> None:
    atoms = build_relaxed_box(args.n_molecules, reference_density(args), args.temperature_K, calc, seed=args.seed)
    write(outdir / "initial.extxyz", snapshot(atoms))
    steps = args.npt_steps
    log(f"[npt] {steps} steps ({steps * args.timestep_fs / 1e3:.1f} ps) Berendsen, "
        f"T={args.temperature_K} K, P={args.pressure_GPa} GPa, taut={args.taut_fs} fs, taup={args.taup_fs} fs")
    Stationary(atoms)
    dyn = NPTBerendsen(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature_K,
        pressure_au=args.pressure_GPa * 1e9 * units.Pascal,
        taut=args.taut_fs * units.fs,
        taup=args.taup_fs * units.fs,
        compressibility_au=args.compressibility_per_GPa / (1e9 * units.Pascal),
    )
    thermo = ThermoLog(outdir / "thermo_npt.log", "npt", args.n_molecules, args.timestep_fs,
                       args.print_interval, header | {"stage": "NPT (Berendsen)"})
    frame_path = outdir / "npt.extxyz"
    write(frame_path, snapshot(atoms, step=0))
    dyn.attach(lambda: thermo.record(dyn.nsteps, atoms), interval=args.thermo_interval)
    dyn.run(steps)
    write(frame_path, snapshot(atoms, step=steps), append=True)
    thermo.close()

    v_mean, v_std = tail_stats(thermo.column("V_A3"), args.npt_average_fraction)
    rho_mean, rho_std = tail_stats(thermo.column("rho_g_cm3"), args.npt_average_fraction)
    p_mean, p_std = tail_stats(thermo.column("P_GPa"), args.npt_average_fraction)
    t_mean, _ = tail_stats(thermo.column("T_K"), args.npt_average_fraction)
    first_half, _ = tail_stats(thermo.column("rho_g_cm3")[: len(thermo.rows) // 2], 0.5)
    rescale_volume_by_molecule(atoms, v_mean, ATOMS_PER_MOLECULE)
    write(outdir / "production_volume.extxyz", snapshot(atoms))

    ref = reference_density_g_cm3(args.temperature_K, args.pressure_GPa)
    result = {
        "mean_volume_A3": v_mean, "std_volume_A3": v_std,
        "mean_density_g_cm3": rho_mean, "std_density_g_cm3": rho_std,
        "reference_density_g_cm3": ref,
        "density_vs_reference_percent": 100 * (rho_mean / ref - 1) if ref else None,
        "density_change_between_averaging_windows_percent": 100 * (rho_mean / first_half - 1),
        "mean_pressure_GPa": p_mean, "std_pressure_GPa": p_std, "mean_T_K": t_mean,
        "box_length_A": float(atoms.cell.lengths()[0]),
    }
    log(f"[npt] <rho> = {rho_mean:.5f} +/- {rho_std:.5f} g/cm3 (reference {ref if ref is None else round(ref, 5)}), <P> = {p_mean:.4f} GPa, "
        f"box rescaled to L = {result['box_length_A']:.4f} A")
    summary.set("npt", result)
    summary.set("volume", {
        "source": "npt", "density_g_cm3": rho_mean, "reference_density_g_cm3": ref,
        "density_vs_reference_percent": result["density_vs_reference_percent"],
        "box_length_A": result["box_length_A"], "volume_A3": v_mean,
    })


def run_nvt(args, atoms: Atoms, outdir: Path, summary: Summary, header: dict) -> Atoms:
    steps = args.nvt_steps
    log(f"[nvt] {steps} steps ({steps * args.timestep_fs / 1e3:.1f} ps) Langevin, "
        f"T={args.temperature_K} K, friction={args.langevin_friction_per_fs}/fs, V fixed")
    Stationary(atoms)
    dyn = Langevin(atoms, args.timestep_fs * units.fs, temperature_K=args.temperature_K,
                   friction=args.langevin_friction_per_fs / units.fs,
                   rng=np.random.default_rng(args.seed + 1))
    thermo = ThermoLog(outdir / "thermo_nvt.log", "nvt", args.n_molecules, args.timestep_fs,
                       args.print_interval, header | {"stage": "NVT (Langevin)"})
    frame_path = outdir / "nvt.extxyz"
    write(frame_path, snapshot(atoms, step=0))
    dyn.attach(lambda: thermo.record(dyn.nsteps, atoms), interval=args.thermo_interval)
    dyn.run(steps)
    Stationary(atoms)
    write(frame_path, snapshot(atoms, step=steps), append=True)
    thermo.close()

    half = len(thermo.rows) // 2
    p_mean, p_stderr = block_mean_stderr(thermo.column("P_GPa")[half:])
    t_mean, t_std = tail_stats(thermo.column("T_K"), 0.5)
    log(f"[nvt] second half: <T> = {t_mean:.2f} K, <P> = {p_mean:.4f} +/- {p_stderr:.4f} GPa (stderr; "
        f"target {args.pressure_GPa})")
    summary.set("nvt", {"mean_pressure_GPa": p_mean, "stderr_pressure_GPa": p_stderr,
                        "std_pressure_GPa": float(thermo.column("P_GPa")[half:].std()),
                        "mean_T_K": t_mean, "std_T_K": t_std,
                        "density_g_cm3": float(thermo.column("rho_g_cm3")[-1])})
    return atoms


def run_nve(args, atoms: Atoms, outdir: Path, summary: Summary, header: dict) -> None:
    steps = args.nve_steps
    for stale in ("nve.extxyz", "nve_com.npz", "thermo_nve.log"):
        (outdir / stale).unlink(missing_ok=True)
    log(f"[nve] {steps} steps ({steps * args.timestep_fs / 1e3:.1f} ps) Velocity Verlet; "
        f"extxyz every {args.traj_interval}, COM every {args.com_interval}, thermo every {args.thermo_interval}")

    Stationary(atoms)
    cell = np.array(atoms.get_cell())
    masses = atoms.get_masses()
    tracker = UnwrappedCOMTracker(
        molecule_centers_of_mass(atoms.get_positions(), masses, ATOMS_PER_MOLECULE, cell=cell), cell)
    com_times: list[float] = []
    com_frames: list[np.ndarray] = []

    def sample_com():
        com = molecule_centers_of_mass(atoms.get_positions(), masses, ATOMS_PER_MOLECULE, cell=cell)
        com_frames.append(tracker.update(com).copy())
        com_times.append(dyn.nsteps * args.timestep_fs)

    def flush_com():
        tmp = outdir / "nve_com.tmp.npz"
        np.savez(tmp, times_fs=np.array(com_times), unwrapped_com=np.array(com_frames))
        tmp.replace(outdir / "nve_com.npz")

    dyn = VelocityVerlet(atoms, timestep=args.timestep_fs * units.fs)
    thermo = ThermoLog(outdir / "thermo_nve.log", "nve", args.n_molecules, args.timestep_fs,
                       args.print_interval, header | {"stage": "NVE (Velocity Verlet)"})
    traj = open(outdir / "nve.extxyz", "w")

    def write_frame():
        row = thermo.rows[-1] if thermo.rows and thermo.rows[-1]["step"] == dyn.nsteps else None
        info = {"step": dyn.nsteps, "time_fs": dyn.nsteps * args.timestep_fs}
        if row:
            info.update(Etot_eV=row["Etot_eV"], T_K=row["T_K"], P_GPa=row["P_GPa"])
        write(traj, snapshot(atoms, **info), format="extxyz")
        traj.flush()

    def check_drift():
        drift = abs(thermo.rows[-1]["dEtot_per_mol_meV"])
        if drift > args.max_drift_meV_per_mol:
            flush_com()
            raise RuntimeError(f"NVE energy drift {drift:.2f} meV/molecule at step {dyn.nsteps} exceeds "
                               f"{args.max_drift_meV_per_mol} -- check timestep / model stability")

    # Order matters: thermo row first so frames can carry its values.
    dyn.attach(lambda: thermo.record(dyn.nsteps, atoms), interval=args.thermo_interval)
    dyn.attach(check_drift, interval=args.thermo_interval)
    dyn.attach(write_frame, interval=args.traj_interval)
    dyn.attach(sample_com, interval=args.com_interval)
    dyn.attach(flush_com, interval=args.com_flush_interval)
    wall = time.perf_counter()
    dyn.run(steps)
    wall = time.perf_counter() - wall
    traj.close()
    thermo.close()
    flush_com()

    de = thermo.column("dEtot_per_mol_meV")
    t = thermo.column("time_ps")
    drift_slope = float(np.polyfit(t, de, 1)[0]) if len(t) > 1 else float("nan")
    p = thermo.column("P_GPa")
    temps = thermo.column("T_K")
    result = {
        "steps": steps, "time_ps": steps * args.timestep_fs / 1e3,
        "wall_hours": wall / 3600, "mean_steps_per_s": steps / wall,
        "ns_per_day": steps * args.timestep_fs * 1e-6 / wall * 86400,
        "mean_T_K": float(temps.mean()), "std_T_K": float(temps.std()),
        "mean_pressure_GPa": float(p.mean()), "std_pressure_GPa": float(p.std()),
        "max_abs_dE_per_mol_meV": float(np.abs(de).max()),
        "dE_drift_meV_per_mol_per_ps": drift_slope,
    }
    log(f"[nve] done in {result['wall_hours']:.2f} h ({result['mean_steps_per_s']:.1f} steps/s): "
        f"<T> = {result['mean_T_K']:.2f} K, <P> = {result['mean_pressure_GPa']:.4f} GPa, "
        f"max|dE|/mol = {result['max_abs_dE_per_mol_meV']:.3f} meV, drift {drift_slope:+.4f} meV/mol/ps")
    summary.set("nve", result)


def run_diffusion(args, outdir: Path, summary: Summary) -> None:
    with np.load(outdir / "nve_com.npz") as data:
        times_fs, com = data["times_fs"], data["unwrapped_com"]
    times_fs = times_fs - times_fs[0]
    com = com - com[0]
    window = (args.msd_fit_start, args.msd_fit_end)
    msd = ensemble_msd(com)
    fit = fit_diffusion_coefficient(times_fs, msd, fit_fraction=window)
    blocks = block_average_diffusion(times_fs, com, n_blocks=args.msd_blocks, fit_fraction=window)
    np.savez(outdir / "nve_msd.npz", times_fs=times_fs, msd_A2=msd)

    result = {
        "D_PBC_m2_s": fit["D"] * A2_FS_TO_M2_S,
        "block_D_mean_m2_s": blocks["D_mean"] * A2_FS_TO_M2_S,
        "block_D_stderr_m2_s": blocks["D_stderr"] * A2_FS_TO_M2_S,
        "n_blocks": args.msd_blocks,
        "fit_window_fraction": list(window),
        "loglog_slope": fit["loglog_slope"],
        "is_diffusive": fit["is_diffusive"],
        "box_length_A": summary.data["volume"]["box_length_A"],
        "note": "Finite-box D_PBC (no Yeh-Hummer correction)",
    }
    log(f"[msd] D_PBC = {result['D_PBC_m2_s']:.4e} m^2/s, blocks {result['block_D_mean_m2_s']:.4e} "
        f"+/- {result['block_D_stderr_m2_s']:.1e} (stderr, {args.msd_blocks} blocks), "
        f"log-log slope {fit['loglog_slope']:.3f}")
    if not fit["is_diffusive"]:
        log("[msd] WARNING: MSD not diffusive in the fit window -- run longer or change the window")
    summary.set("diffusion", result)


def main(argv=None) -> None:
    args = parse_args(argv)
    for name in ("thermo_interval", "traj_interval", "com_interval", "com_flush_interval", "print_interval"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.print_interval % args.thermo_interval:
        raise SystemExit("--print-interval must be a multiple of --thermo-interval")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = Summary(outdir / "summary.json")
    summary.set("config", vars(args) | {"host": platform.node()})

    header = {
        "model": args.model + (f" ({args.model_path})" if args.model_path else ""),
        "dtype": args.dtype, "device": args.device,
        "state point": f"T={args.temperature_K} K, P={args.pressure_GPa} GPa, {args.n_molecules} CH4",
        "timestep_fs": args.timestep_fs,
        "units": "energies in eV (U = potential), P = virial + kinetic, T from 3N-3 dof",
    }
    log(f"host {platform.node()} | {header['state point']} | model {header['model']} {args.dtype} on {args.device}")

    calc = get_calculator(model=args.model, device=args.device, model_path=args.model_path,
                          default_dtype=args.dtype)

    def load(name: str) -> Atoms:
        atoms = read(outdir / name, index=-1)
        atoms.calc = calc
        return atoms

    if summary.done("volume"):
        log(f"[volume] already done ({summary.data['volume']['source']}), skipping")
    elif args.eos_fit:
        run_volume_from_eos(args, calc, outdir, summary)
    else:
        run_npt(args, calc, outdir, summary, header)

    if summary.done("nvt"):
        log("[nvt] already done, skipping")
    else:
        run_nvt(args, load("production_volume.extxyz"), outdir, summary, header)

    if summary.done("nve"):
        log("[nve] already done, skipping")
    else:
        run_nve(args, load("nvt.extxyz"), outdir, summary, header)

    run_diffusion(args, outdir, summary)
    log(f"all done -> {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
