#!/usr/bin/env python
"""P(rho) equation of state along one isotherm from a chained fixed-volume NVT scan.

Why NVT at fixed densities rather than NPT at fixed pressures: the mean
pressure at fixed V converges fast (it decorrelates in well under a ps),
while <V> under a barostat converges slowly (volume relaxes on the
barostat timescale and fluctuates collectively). So each point needs only
a few ps, and it has a clean block-averaged error bar.

The densities are visited from high to low, each starting from the
previous point's final state rescaled to the new volume (whole molecules
move, bonds don't stretch). Only the first point starts from a lattice and
needs a long equilibration; the rest start from an equilibrated fluid a
few percent away, so they need only --equil-steps.

Outputs in --outdir:
  thermo_rho<rho>.log   thermo log for each density (equilibration + sampling)
  rho<rho>.extxyz       final frame of each density (chaining and resume)
  eos_points.json       one record per density (scripts/plot_eos.py format)
  eos_fit.json          weighted polynomial P(rho), and rho at --target-pressures;
                        pass it to run_state_point.py --eos-fit to skip NPT
  eos_fit.png           points, fit and the reference (NIST) isotherm

Rerunning skips densities already in eos_points.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mace_ch4.calculators import get_calculator  # noqa: E402
from mace_ch4.eos import (  # noqa: E402
    append_eos_point,
    density_at_pressure,
    fit_pressure_vs_density,
    load_eos_points,
    reference_density_g_cm3,
)
from mace_ch4.geometry import ATOMS_PER_MOLECULE  # noqa: E402
from mace_ch4.mdtools import (  # noqa: E402
    ThermoLog,
    block_mean_stderr,
    build_relaxed_box,
    log,
    rescale_volume_by_molecule,
    snapshot,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", required=True)
    p.add_argument("--temperature-K", type=float, default=450.0)
    p.add_argument("--n-molecules", type=int, default=128)
    p.add_argument("--densities", type=float, nargs="+", default=None,
                   help="Densities in g/cm3 (default: 0.200 to 0.500 in steps of 0.025)")
    p.add_argument("--target-pressures", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5],
                   help="Pressures (GPa) to report the fitted density at")
    p.add_argument("--model", default="off23-medium")
    p.add_argument("--model-path", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep-fs", type=float, default=0.5)
    p.add_argument("--first-equil-steps", type=int, default=20_000,
                   help="Equilibration for the first density, which starts from a lattice (10 ps)")
    p.add_argument("--equil-steps", type=int, default=4_000, help="Equilibration after each rescale (2 ps)")
    p.add_argument("--sample-steps", type=int, default=20_000, help="Sampling steps per density (10 ps)")
    p.add_argument("--langevin-friction-per-fs", type=float, default=0.01)
    p.add_argument("--thermo-interval", type=int, default=10, help="Steps between P samples / log rows")
    p.add_argument("--print-interval", type=int, default=2000)
    p.add_argument("--n-blocks", type=int, default=10, help="Blocks for the pressure standard error")
    p.add_argument("--fit-degree", type=int, default=3)
    return p.parse_args(argv)


def tag(density: float) -> str:
    return f"rho{density:.4f}"


def run_point(args, atoms, density: float, equil_steps: int, outdir: Path, header: dict) -> dict:
    rescale_volume_by_molecule(atoms, atoms.get_masses().sum() / (density / 1.66053906660), ATOMS_PER_MOLECULE)
    Stationary(atoms)
    total = equil_steps + args.sample_steps
    log(f"[{tag(density)}] {equil_steps} equilibration + {args.sample_steps} sampling steps, "
        f"L={atoms.cell.lengths()[0]:.4f} A")
    dyn = Langevin(atoms, args.timestep_fs * units.fs, temperature_K=args.temperature_K,
                   friction=args.langevin_friction_per_fs / units.fs,
                   rng=np.random.default_rng(args.seed + int(round(density * 1e4))))
    thermo = ThermoLog(outdir / f"thermo_{tag(density)}.log", tag(density), args.n_molecules,
                       args.timestep_fs, args.print_interval,
                       header | {"stage": f"NVT (Langevin) at {density} g/cm3; rows with step > "
                                          f"{equil_steps} are the sampling window"})
    dyn.attach(lambda: thermo.record(dyn.nsteps, atoms), interval=args.thermo_interval)
    dyn.run(total)
    thermo.close()
    write(outdir / f"{tag(density)}.extxyz", snapshot(atoms, step=total))

    steps = thermo.column("step")
    window = steps > equil_steps
    p_mean, p_stderr = block_mean_stderr(thermo.column("P_GPa")[window], args.n_blocks)
    t_mean = float(thermo.column("T_K")[window].mean())
    point = {
        "model": args.model,
        "n_molecules": args.n_molecules,
        "temperature_K_target": args.temperature_K,
        "temperature_K_measured": t_mean,
        "density_g_cm3": density,
        "density_kg_m3": density * 1e3,
        "volume_A3": atoms.get_volume(),
        "pressure_GPa_measured": p_mean,
        "pressure_GPa_stderr": p_stderr,
        "pressure_GPa_std": float(thermo.column("P_GPa")[window].std()),
        "sampling_ps": args.sample_steps * args.timestep_fs / 1e3,
    }
    log(f"[{tag(density)}] <P> = {p_mean:.4f} +/- {p_stderr:.4f} GPa (stderr), <T> = {t_mean:.2f} K")
    return point


def fit_and_report(args, points: list[dict], outdir: Path) -> dict:
    rho = np.array([pt["density_g_cm3"] for pt in points])
    p = np.array([pt["pressure_GPa_measured"] for pt in points])
    err = np.array([pt["pressure_GPa_stderr"] for pt in points])
    order = np.argsort(rho)
    rho, p, err = rho[order], p[order], err[order]
    coeffs = fit_pressure_vs_density(rho, p, err, degree=args.fit_degree)
    normalized_residuals = (p - np.polyval(coeffs, rho)) / err

    targets = {}
    for target in args.target_pressures:
        try:
            density = density_at_pressure(coeffs, target, (rho[0], rho[-1]))
        except ValueError as exc:
            log(f"[fit] {exc}")
            continue
        ref = reference_density_g_cm3(args.temperature_K, target)
        targets[str(target)] = {
            "density_g_cm3": density,
            "reference_density_g_cm3": ref,
            "density_vs_reference_percent": 100 * (density / ref - 1) if ref else None,
        }
        log(f"[fit] P = {target} GPa -> rho = {density:.5f} g/cm3"
            + (f" (reference {ref:.5f}, {100 * (density / ref - 1):+.2f}%)" if ref else ""))

    fit = {
        "model": args.model,
        "n_molecules": args.n_molecules,
        "temperature_K": args.temperature_K,
        "form": "P_GPa = polyval(coeffs, rho_g_cm3)",
        "degree": args.fit_degree,
        "coeffs": coeffs.tolist(),
        "density_range_g_cm3": [float(rho[0]), float(rho[-1])],
        "reduced_chi2": float(np.sum(normalized_residuals ** 2) / max(len(rho) - args.fit_degree - 1, 1)),
        "targets": targets,
    }
    (outdir / "eos_fit.json").write_text(json.dumps(fit, indent=2))
    log(f"[fit] degree-{args.fit_degree} fit over {len(rho)} points, reduced chi2 = {fit['reduced_chi2']:.2f}"
        " (>> 1 means the polynomial is too stiff or error bars too small)")
    plot_fit(args, rho, p, err, coeffs, outdir / "eos_fit.png")
    return fit


def plot_fit(args, rho, p, err, coeffs, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(rho[0], rho[-1], 200)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(rho, p, yerr=err, fmt="o", capsize=3, label=f"{args.model} NVT")
    ax.plot(grid, np.polyval(coeffs, grid), "-", label=f"degree-{args.fit_degree} fit")
    ref_p = np.linspace(0.02, max(p.max(), 0.05) * 1.1, 60)
    ref_rho = [reference_density_g_cm3(args.temperature_K, x) for x in ref_p]
    if all(r is not None for r in ref_rho):
        ax.plot(ref_rho, ref_p, "k--", label="reference (NIST) EOS")
    ax.set_xlabel(r"density (g/cm$^3$)")
    ax.set_ylabel("pressure (GPa)")
    ax.set_title(f"CH$_4$ {args.temperature_K:g} K, {args.n_molecules} molecules")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    densities = args.densities or list(np.round(np.arange(0.200, 0.5001, 0.025), 4))
    densities = sorted(densities, reverse=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    points_path = outdir / "eos_points.json"
    points = load_eos_points(points_path) if points_path.is_file() else []
    done = {round(pt["density_g_cm3"], 4) for pt in points}

    header = {
        "model": args.model + (f" ({args.model_path})" if args.model_path else ""),
        "dtype": args.dtype, "device": args.device,
        "state point": f"T={args.temperature_K} K, {args.n_molecules} CH4, fixed volume",
        "timestep_fs": args.timestep_fs,
        "units": "energies in eV (U = potential), P = virial + kinetic, T from 3N-3 dof",
    }
    log(f"host {platform.node()} | EOS scan at {args.temperature_K} K over {len(densities)} densities "
        f"{densities[0]}..{densities[-1]} g/cm3 | model {header['model']} {args.dtype} on {args.device}")
    calc = get_calculator(model=args.model, device=args.device, model_path=args.model_path,
                          default_dtype=args.dtype)

    atoms = None
    for density in densities:
        if round(density, 4) in done:
            atoms = read(outdir / f"{tag(density)}.extxyz", index=-1)
            log(f"[{tag(density)}] already done, skipping")
            continue
        if atoms is None:
            atoms = build_relaxed_box(args.n_molecules, density, args.temperature_K, calc, seed=args.seed)
            equil = args.first_equil_steps
        else:
            equil = args.equil_steps
        atoms.calc = calc
        point = run_point(args, atoms, density, equil, outdir, header)
        append_eos_point(points_path, point)
        points.append(point)

    if len(points) > args.fit_degree:
        fit_and_report(args, points, outdir)
    else:
        log(f"[fit] only {len(points)} points; need more than {args.fit_degree} to fit")


if __name__ == "__main__":
    main()
