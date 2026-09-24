#!/usr/bin/env python3
"""
run_statepoint.py

One CH4 state point with MACE-OFF: NPT -> NVT -> NVE -> self-diffusion D.

    python run_statepoint.py --pressure 0.2                       # MACE-OFF23, 450 K
    python run_statepoint.py --pressure 0.3 --model mace_off24
    python run_statepoint.py --pressure 0.2 --npt-steps 2000 --nvt-steps 1000 --nve-steps 2000   # quick test

Per run:
  0. lattice box of whole CH4 molecules at the NIST density for (450 K, P)
  1. MELT  Langevin NVT at that density, discarded (lattice -> fluid)
  2. NPT   Berendsen T + isotropic P; box then set to the MEAN volume of the
           last half (molecules moved rigidly with their COM)
  3. NVT   Langevin at that fixed volume: re-thermalise; its mean P checks
           that the volume really gives the target P
  4. NVE   Velocity Verlet: full trajectory + COM trajectory, D from the
           multi-origin COM MSD (block-averaged error)

Outputs in OUTDIR (default statepoint_<model>_450K_<P>GPa/):
  thermo_{melt,npt,nvt,nve}.log   every THERMO_EVERY steps: step, t, T, U, KE,
                                  Etot, Etot/mol, dEtot/mol, P, V, rho, steps/s
  {melt,npt,nvt}.extxyz           first + last frame only
  nve.extxyz                      every TRAJ_EVERY steps (positions + momenta)
  nve_com.npz                     unwrapped molecule COMs every COM_EVERY steps
  nve_msd.npz, summary.json       MSD(t); densities, pressures, drift, D

Conventions as compare_4way_450K_02GPa.py: T from 3N-3 dof, P = virial +
kinetic (include_ideal_gas=True), energy drift per molecule in meV.
D here is the finite-box D_PBC (no Yeh-Hummer correction).

Calculators come from shared_potentials.py in SHARED_POTENTIALS_DIR, as in
compare_4way_450K_02GPa.py. No restart: size the run to fit one job.
"""

import argparse
import functools
import json
import os
import platform
import sys
import time

import numpy as np
import torch

torch.load = functools.partial(torch.load, weights_only=False)  # MACE checkpoint fix

from ase import Atoms, units
from ase.build import molecule
from ase.io import write as ase_write
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from scipy.spatial.transform import Rotation

SHARED_POTENTIALS_DIR = os.environ.get(
    "SHARED_POTENTIALS_DIR",
    "/lustre/scratch/mmm0037/methane/first-test-md-x12t-005",
)
sys.path.insert(0, SHARED_POTENTIALS_DIR)

import shared_potentials as sp  # noqa: E402

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:9.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# CONFIG (defaults; most can be overridden on the command line)
# ---------------------------------------------------------------------------
N_MOLECULES = 128
TEMPERATURE_K = 450.0
DT_FS = 0.5
MELT_STEPS = 2_000     # 1 ps Langevin from the lattice, discarded
NPT_STEPS = 100_000    # 50 ps
NVT_STEPS = 40_000     # 20 ps
NVE_STEPS = 400_000    # 200 ps
TAUT_FS = 100.0        # Berendsen thermostat
TAUP_FS = 1000.0       # Berendsen barostat
COMPRESSIBILITY_PER_GPA = 1.0   # only sets how fast the box responds
LANGEVIN_FRICTION = 0.01        # 1/fs
THERMO_EVERY = 100     # steps between thermo-log rows
PRINT_EVERY = 1_000    # steps between console lines (multiple of THERMO_EVERY)
TRAJ_EVERY = 100       # steps between nve.extxyz frames (50 fs)
COM_EVERY = 10         # steps between COM samples for the MSD (5 fs)
MSD_FIT = (0.1, 0.5)   # fit window, as fractions of the NVE length
MSD_BLOCKS = 5
SEED = 42
DEVICE = "cuda"

APM = 5  # atoms per molecule (C, H, H, H, H)
EV_A3_TO_GPA = 160.21766208
AMU_A3_TO_G_CM3 = 1.66053906660
A2_FS_TO_M2_S = 1e-5

# NIST WebBook (Setzmann-Wagner EOS) CH4 density at 450 K, g/cm3, by P in GPa.
# Starting density only: NPT finds the model's own density.
NIST_450K = {
    0.05: 0.17992, 0.1: 0.26473, 0.15: 0.31269, 0.2: 0.34586, 0.25: 0.37132,
    0.3: 0.39208, 0.35: 0.40967, 0.4: 0.42499, 0.45: 0.43858, 0.5: 0.45084,
    0.55: 0.46201, 0.6: 0.47229, 0.65: 0.48183, 0.7: 0.49074, 0.75: 0.49910,
    0.8: 0.50698, 0.85: 0.51445, 0.9: 0.52154, 0.95: 0.52830, 1.0: 0.53477,
}
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pressure", type=float, required=True, help="Target pressure, GPa")
    p.add_argument("--model", default="mace_off23", choices=["mace_off23", "mace_off24"])
    p.add_argument("--density", type=float, default=None,
                   help="Starting density, g/cm3 (default: NIST at 450 K, interpolated)")
    p.add_argument("--temperature", type=float, default=TEMPERATURE_K)
    p.add_argument("--n-molecules", type=int, default=N_MOLECULES)
    p.add_argument("--melt-steps", type=int, default=MELT_STEPS)
    p.add_argument("--npt-steps", type=int, default=NPT_STEPS)
    p.add_argument("--nvt-steps", type=int, default=NVT_STEPS)
    p.add_argument("--nve-steps", type=int, default=NVE_STEPS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--outdir", default=None)
    return p.parse_args()


def get_calc(model, device):
    if model == "mace_off23":
        return sp.get_mace23_calc(device)
    return sp.get_mace24_calc(device)


# ---------------------------------------------------------------- box + helpers
def build_box(n_mol, density_g_cm3, seed):
    """Whole CH4 molecules, randomly rotated, on a random subset of a cubic grid.

    Atoms are contiguous per molecule (C first) and never wrapped, so
    molecules stay whole and COMs are continuous for the whole run.
    """
    rng = np.random.default_rng(seed)
    template = molecule("CH4")
    template.positions -= template.get_center_of_mass()
    mass_amu = n_mol * template.get_masses().sum()
    L = (mass_amu * AMU_A3_TO_G_CM3 / density_g_cm3) ** (1 / 3)
    n_side = int(np.ceil(n_mol ** (1 / 3)))
    grid = np.array([(i, j, k) for i in range(n_side) for j in range(n_side) for k in range(n_side)], float)
    sites = (grid[np.sort(rng.choice(len(grid), n_mol, replace=False))] + 0.5) * (L / n_side)
    positions = [Rotation.random(random_state=rng.integers(1 << 31)).apply(template.positions) + s for s in sites]
    return Atoms(numbers=np.tile(template.numbers, n_mol), positions=np.concatenate(positions),
                 cell=[L, L, L], pbc=True)


def ndof(atoms):
    return 3 * len(atoms) - 3


def temperature(atoms):
    return atoms.get_kinetic_energy() / (0.5 * ndof(atoms) * units.kB)


def pressure_GPa(atoms):
    return -np.mean(atoms.get_stress(voigt=True, include_ideal_gas=True)[:3]) * EV_A3_TO_GPA


def density(atoms):
    return atoms.get_masses().sum() / atoms.get_volume() * AMU_A3_TO_G_CM3


def mol_coms(atoms):
    pos = atoms.get_positions().reshape(-1, APM, 3)
    m = atoms.get_masses().reshape(-1, APM, 1)
    return (pos * m).sum(axis=1) / m.sum(axis=1)


def set_volume_rigid_molecules(atoms, volume):
    """Scale the box to `volume`, moving each molecule rigidly with its COM."""
    s = (volume / atoms.get_volume()) ** (1 / 3)
    shift = np.repeat((s - 1) * mol_coms(atoms), APM, axis=0)
    atoms.set_cell(atoms.cell * s, scale_atoms=False)
    atoms.positions += shift


def frame(atoms, **info):
    """Calculator-free copy (positions, momenta, cell) for extxyz output."""
    f = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=True,
              momenta=atoms.get_momenta())
    f.info.update(info)
    return f


class Thermo:
    """Thermo log: one row every THERMO_EVERY steps, console line every PRINT_EVERY."""

    HEAD = ("step", "time_ps", "T_K", "U_eV", "KE_eV", "Etot_eV", "Etot_per_mol_eV",
            "dEtot_per_mol_meV", "P_GPa", "V_A3", "rho_g_cm3", "steps_per_s")
    FMT = "{:>9d} {:>10.4f} {:>8.2f} {:>16.6f} {:>11.5f} {:>16.6f} {:>16.6f} {:>17.4f} {:>8.4f} {:>11.3f} {:>9.5f} {:>11.2f}\n"

    def __init__(self, path, stage, atoms, n_mol, header):
        self.stage, self.atoms, self.n_mol = stage, atoms, n_mol
        self.rows, self.E0 = [], None
        self.t_last, self.step_last = None, 0
        self.fh = open(path, "w")
        for k, v in header.items():
            self.fh.write(f"# {k}: {v}\n")
        self.fh.write(f"# stage: {stage}; dEtot_per_mol_meV relative to this stage's first row\n")
        self.fh.write("# " + " ".join(self.HEAD) + "\n")

    def __call__(self, step):
        a = self.atoms
        now = time.time()
        rate = (step - self.step_last) / (now - self.t_last) if self.t_last and step > self.step_last else float("nan")
        self.t_last, self.step_last = now, step
        U, KE = a.get_potential_energy(), a.get_kinetic_energy()
        E = U + KE
        if self.E0 is None:
            self.E0 = E
        row = (step, step * DT_FS / 1000, temperature(a), U, KE, E, E / self.n_mol,
               1000 * (E - self.E0) / self.n_mol, pressure_GPa(a), a.get_volume(), density(a), rate)
        self.rows.append(row)
        self.fh.write(self.FMT.format(*row))
        self.fh.flush()
        if step % PRINT_EVERY == 0:
            log(f"[{self.stage}] {step:>8d}  t={row[1]:8.3f} ps  T={row[2]:7.2f} K  P={row[8]:7.4f} GPa  "
                f"rho={row[10]:.5f}  dE/mol={row[7]:+8.3f} meV  {rate:7.2f} steps/s")

    def col(self, name, second_half=False):
        c = np.array([r[self.HEAD.index(name)] for r in self.rows])
        return c[len(c) // 2:] if second_half else c

    def close(self):
        self.fh.close()


def run_stage(stage, dyn, atoms, steps, outdir, header, keep_frames=True):
    thermo = Thermo(os.path.join(outdir, f"thermo_{stage}.log"), stage, atoms, len(atoms) // APM, header)
    if keep_frames:
        ase_write(os.path.join(outdir, f"{stage}.extxyz"), frame(atoms, step=0))
    dyn.attach(lambda: thermo(dyn.nsteps), interval=THERMO_EVERY)
    dyn.run(steps)
    thermo.close()
    if keep_frames:
        ase_write(os.path.join(outdir, f"{stage}.extxyz"), frame(atoms, step=steps), append=True)
    return thermo


def block_stderr(x, n_blocks=10):
    n_blocks = min(n_blocks, len(x) // 2)
    if n_blocks < 2:
        return float("nan")
    b = len(x) // n_blocks
    means = np.asarray(x[: b * n_blocks]).reshape(n_blocks, b).mean(axis=1)
    return means.std(ddof=1) / np.sqrt(n_blocks)


# ---------------------------------------------------------------- MSD / D
def msd_multi_origin(com):
    """MSD(t) averaged over molecules and all time origins (FFT, O(N log N)).
    com: (n_frames, n_mol, 3), unwrapped."""
    n = com.shape[0]
    sq = np.sum(com ** 2, axis=2)                                    # (n, mol)
    f = np.fft.rfft(com, n=2 * n, axis=0)
    s2 = np.fft.irfft(np.sum(f * f.conj(), axis=2), axis=0)[:n]      # sum_d autocorr, (n, mol)
    s2 /= (n - np.arange(n))[:, None]
    sq_pad = np.vstack([sq, np.zeros((1, sq.shape[1]))])
    q = 2 * sq.sum(axis=0)
    s1 = np.empty_like(s2)
    for m in range(n):
        q = q - sq_pad[m - 1] - sq_pad[n - m]
        s1[m] = q / (n - m)
    return (s1 - 2 * s2).mean(axis=1)


def fit_D(t_fs, msd, window):
    i0, i1 = int(window[0] * len(t_fs)), int(window[1] * len(t_fs))
    slope = np.polyfit(t_fs[i0:i1], msd[i0:i1], 1)[0]
    loglog = np.polyfit(np.log(t_fs[i0:i1]), np.log(msd[i0:i1]), 1)[0]
    return slope / 6 * A2_FS_TO_M2_S, loglog


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    global TEMPERATURE_K
    TEMPERATURE_K = args.temperature
    assert PRINT_EVERY % THERMO_EVERY == 0

    nist = float(np.interp(args.pressure, list(NIST_450K), list(NIST_450K.values()))) \
        if TEMPERATURE_K == 450.0 and 0.05 <= args.pressure <= 1.0 else None
    rho0 = args.density or nist
    if rho0 is None:
        sys.exit("No NIST density for this (T, P): pass --density")
    outdir = args.outdir or f"statepoint_{args.model}_{TEMPERATURE_K:g}K_{args.pressure:g}GPa"
    os.makedirs(outdir, exist_ok=True)

    header = {"model": args.model, "T_K": TEMPERATURE_K, "P_target_GPa": args.pressure,
              "n_molecules": args.n_molecules, "dt_fs": DT_FS,
              "units": "eV (U = potential), GPa (virial + kinetic), T from 3N-3 dof"}
    log(f"python {platform.python_version()} on {platform.node()} | torch {torch.__version__} | "
        f"cuda {torch.cuda.is_available()}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    log(f"{args.model}, {args.n_molecules} CH4, T={TEMPERATURE_K} K, P={args.pressure} GPa, start rho={rho0:.5f} "
        f"| melt/npt/nvt/nve = {args.melt_steps}/{args.npt_steps}/{args.nvt_steps}/{args.nve_steps} steps x {DT_FS} fs "
        f"| out: {outdir}/")
    summary = {"config": vars(args) | {"host": platform.node(), "nist_density_g_cm3": nist}}

    def save_summary():
        with open(os.path.join(outdir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

    # ---- build + melt
    atoms = build_box(args.n_molecules, rho0, args.seed)
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    log(f"[build] {len(atoms)} atoms, L={atoms.cell.lengths()[0]:.3f} A, min distance {d.min():.3f} A")
    atoms.calc = get_calc(args.model, args.device)
    log(f"[build] E={atoms.get_potential_energy():.4f} eV, P={pressure_GPa(atoms):.3f} GPa (lattice, no KE)")
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=np.random.default_rng(args.seed))
    Stationary(atoms)

    def langevin(seed_offset):
        return Langevin(atoms, DT_FS * units.fs, temperature_K=TEMPERATURE_K,
                        friction=LANGEVIN_FRICTION / units.fs, rng=np.random.default_rng(args.seed + seed_offset))

    log(f"[melt] {args.melt_steps} steps Langevin at rho={rho0:.5f}")
    run_stage("melt", langevin(1), atoms, args.melt_steps, outdir, header)

    # ---- NPT
    Stationary(atoms)
    log(f"[npt] {args.npt_steps} steps Berendsen, taut={TAUT_FS} fs, taup={TAUP_FS} fs")
    npt = NPTBerendsen(atoms, timestep=DT_FS * units.fs, temperature_K=TEMPERATURE_K,
                       pressure_au=args.pressure * units.GPa, taut=TAUT_FS * units.fs, taup=TAUP_FS * units.fs,
                       compressibility_au=COMPRESSIBILITY_PER_GPA / units.GPa)
    th = run_stage("npt", npt, atoms, args.npt_steps, outdir, header)
    V_mean = th.col("V_A3", True).mean()
    rho_mean, rho_std = th.col("rho_g_cm3", True).mean(), th.col("rho_g_cm3", True).std()
    set_volume_rigid_molecules(atoms, V_mean)
    summary["npt"] = {"mean_density_g_cm3": rho_mean, "std_density_g_cm3": rho_std,
                      "density_vs_nist_percent": 100 * (rho_mean / nist - 1) if nist else None,
                      "mean_P_GPa": th.col("P_GPa", True).mean(), "mean_T_K": th.col("T_K", True).mean(),
                      "mean_volume_A3": V_mean, "box_length_A": atoms.cell.lengths()[0]}
    save_summary()
    log(f"[npt] last half: <rho> = {rho_mean:.5f} +/- {rho_std:.5f} g/cm3"
        + (f" (NIST {nist:.5f}, {summary['npt']['density_vs_nist_percent']:+.2f}%)" if nist else "")
        + f" -> box set to L = {atoms.cell.lengths()[0]:.4f} A")

    # ---- NVT
    Stationary(atoms)
    log(f"[nvt] {args.nvt_steps} steps Langevin at fixed V")
    th = run_stage("nvt", langevin(2), atoms, args.nvt_steps, outdir, header)
    P = th.col("P_GPa", True)
    summary["nvt"] = {"mean_P_GPa": P.mean(), "stderr_P_GPa": block_stderr(P), "std_P_GPa": P.std(),
                      "mean_T_K": th.col("T_K", True).mean(), "density_g_cm3": density(atoms)}
    save_summary()
    log(f"[nvt] last half: <P> = {P.mean():.4f} +/- {block_stderr(P):.4f} GPa (target {args.pressure}), "
        f"<T> = {summary['nvt']['mean_T_K']:.2f} K")

    # ---- NVE
    Stationary(atoms)
    nve = VelocityVerlet(atoms, timestep=DT_FS * units.fs)
    traj_path = os.path.join(outdir, "nve.extxyz")
    if os.path.exists(traj_path):
        os.remove(traj_path)
    com_t, com = [], []
    nve.attach(lambda: (com_t.append(nve.nsteps * DT_FS), com.append(mol_coms(atoms))), interval=COM_EVERY)
    nve.attach(lambda: ase_write(traj_path, frame(atoms, step=nve.nsteps, time_fs=nve.nsteps * DT_FS), append=True),
               interval=TRAJ_EVERY)
    log(f"[nve] {args.nve_steps} steps ({args.nve_steps * DT_FS / 1000:.1f} ps) Velocity Verlet; "
        f"extxyz every {TRAJ_EVERY}, COM every {COM_EVERY}")
    wall = time.time()
    th = run_stage("nve", nve, atoms, args.nve_steps, outdir, header, keep_frames=False)
    wall = time.time() - wall
    com_t, com = np.array(com_t), np.array(com)
    np.savez(os.path.join(outdir, "nve_com.npz"), times_fs=com_t, unwrapped_com=com)

    dE = th.col("dEtot_per_mol_meV")
    tp = th.col("time_ps")
    summary["nve"] = {"mean_T_K": th.col("T_K").mean(), "mean_P_GPa": th.col("P_GPa").mean(),
                      "std_P_GPa": th.col("P_GPa").std(), "max_abs_dE_per_mol_meV": np.abs(dE).max(),
                      "drift_meV_per_mol_per_ps": np.polyfit(tp, dE, 1)[0],
                      "steps_per_s": args.nve_steps / wall, "wall_h": wall / 3600}
    save_summary()
    log(f"[nve] done: <T> = {summary['nve']['mean_T_K']:.2f} K, <P> = {summary['nve']['mean_P_GPa']:.4f} GPa, "
        f"max|dE|/mol = {summary['nve']['max_abs_dE_per_mol_meV']:.3f} meV, "
        f"drift {summary['nve']['drift_meV_per_mol_per_ps']:+.4f} meV/mol/ps, {summary['nve']['steps_per_s']:.1f} steps/s")

    # ---- D from COM MSD
    t = com_t - com_t[0]
    msd = msd_multi_origin(com - com[0])
    np.savez(os.path.join(outdir, "nve_msd.npz"), times_fs=t, msd_A2=msd)
    D, loglog = fit_D(t, msd, MSD_FIT)
    blocks = []
    for blk in np.array_split(np.arange(len(t)), MSD_BLOCKS):
        blocks.append(fit_D(t[blk] - t[blk[0]], msd_multi_origin(com[blk] - com[blk[0]]), MSD_FIT)[0])
    summary["diffusion"] = {"D_PBC_m2_s": D, "block_mean_m2_s": np.mean(blocks),
                            "block_stderr_m2_s": np.std(blocks, ddof=1) / np.sqrt(MSD_BLOCKS),
                            "loglog_slope_in_fit_window": loglog, "fit_window": MSD_FIT,
                            "box_length_A": atoms.cell.lengths()[0]}
    save_summary()
    log(f"[D] D_PBC = {D:.4e} m^2/s, blocks {np.mean(blocks):.4e} +/- "
        f"{summary['diffusion']['block_stderr_m2_s']:.1e} (stderr, {MSD_BLOCKS}), log-log slope {loglog:.3f}"
        + ("" if 0.9 < loglog < 1.1 else "  <-- not diffusive yet: run NVE longer"))
    log(f"=== done. output in {outdir}/ ===")


if __name__ == "__main__":
    main()
