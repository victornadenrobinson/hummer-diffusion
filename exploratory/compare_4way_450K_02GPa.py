#!/usr/bin/env python3
"""
compare_4way_450K_02GPa.py

4-way comparison at 450 K, 0.2 GPa (rho = 0.34586 g/cm3, NIST WebBook /
Setzmann-Wagner EOS):
  - mace_off23 : MACE-OFF23 medium (flexible, ML)
  - mace_off24 : MACE-OFF24 medium (flexible, ML)
  - opls_aa : FLEXIBLE all-atom OPLS-AA-style LJ + harmonic bonds/angles
                 (flexible_lj.FlexibleCH4Calculator -- no FixBondLengths)
  - ua         : TraPPE united-atom LJ (1 site/molecule)

Per model:
  1. build lattice box at target density, preflight
  2. Maxwell-Boltzmann velocities at 450 K
  3. NVT equilibration: N_EQUIL_STEPS of Langevin at 450 K (discarded)
  4. NVE production: N_STEPS of VelocityVerlet, recorded every LOG_EVERY
  5. RDF (COM), MSD (COM, Einstein), energy trace, pressure trace

Why equilibrate then NVE: the lattice start is far from equilibrium, and
pure NVE from it drifts to whatever T the relaxation lands at (MACE went
~455 -> ~350 K, rigid LJ ~534 -> >1000 K). Langevin brings every model
to 450 K first; NVE then gives unthermostatted dynamics (needed for
clean MSD) and a real energy-conservation check.

Energy drift is reported as |dE| PER MOLECULE in eV, not relative to E0:
MACE's E0 includes absolute atomic energies (~-141,000 eV) while LJ's is
~tens of eV, so relative drift is not comparable across models.

Pressure uses get_stress(include_ideal_gas=True): virial + kinetic term.
Without the kinetic term (~0.4 GPa here with per-atom KE) P is badly
underestimated. Instantaneous P fluctuates by several tenths of a GPa
in a 128-molecule box -- compare the NVE MEAN against the 0.2 GPa target.

MSD caveat: 500 fs of NVE is too short for a converged D. This is a
qualitative cross-model check; raise N_STEPS for real diffusion work.

Files: flexible_lj.py must sit next to this script. shared_potentials.py
is imported from SHARED_POTENTIALS_DIR (box building, COM trajectory,
RDF, MACE/UA calculators). rigidify_ch4 is no longer used.

HARD CONSTRAINT reminder: runs on Young-ng only; I have no execution
access -- run it yourself and report the output back.
"""

import functools
import os
import platform
import sys
import time

import numpy as np
import torch

torch.load = functools.partial(torch.load, weights_only=False)  # MACE checkpoint fix

from ase import units
from ase.io import write as ase_write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.neighborlist import neighbor_list

SHARED_POTENTIALS_DIR = os.environ.get(
    "SHARED_POTENTIALS_DIR",
    "/lustre/scratch/mmm0037/methane/first-test-md-x12t-005",
)
sys.path.insert(0, SHARED_POTENTIALS_DIR)

import shared_potentials as sp  # noqa: E402
from flexible_lj import FlexibleCH4Calculator  # noqa: E402

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:8.2f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
N_MOLECULES = 128
DENSITY_G_CM3 = 0.34586  # NIST @ 450 K / 0.2 GPa
TEMPERATURE_K = 450.0
DT_FS = 0.5
N_EQUIL_STEPS = 2000  # Langevin NVT, discarded (1 ps) -- lattice melt needs time
LANGEVIN_FRICTION = 0.01  # 1/fs -- strong coupling is fine, equilibration only
N_STEPS = 1000  # NVE production (0.5 ps)
LOG_EVERY = 5  # steps between data samples (frames, E/T/P, extxyz write)
PRINT_EVERY = 50  # steps between console lines (multiple of LOG_EVERY)
SEED = 42
DEVICE = "cuda"
OUTDIR = "compare_4way_450K_02GPa_out"

assert PRINT_EVERY % LOG_EVERY == 0, "PRINT_EVERY must be a multiple of LOG_EVERY"

EV_A3_TO_GPA = 160.21766208

UNITED_ATOM_MODELS = {"ua"}
ATOMS_PER_MOL = {"mace_off23": 5, "mace_off24": 5, "opls_aa": 5, "ua": 1}
MODEL_NAMES = ["mace_off23", "mace_off24", "opls_aa", "ua"]

# ---------------------------------------------------------------------------


def ndof(atoms):
    """No constraints anywhere now; -3 for COM momentum removal."""
    return 3 * len(atoms) - 3


def temperature_K(atoms):
    return atoms.get_kinetic_energy() / (0.5 * ndof(atoms) * units.kB)


def build_box(model_name):
    if model_name in UNITED_ATOM_MODELS:
        return sp.build_box_ua(N_MOLECULES, DENSITY_G_CM3, seed=SEED)
    return sp.build_box_atomistic(N_MOLECULES, DENSITY_G_CM3, seed=SEED)


def get_calc(model_name, mol_id, device):
    if model_name == "mace_off23":
        return sp.get_mace23_calc(device)
    if model_name == "mace_off24":
        return sp.get_mace24_calc(device)
    if model_name == "opls_aa":
        return FlexibleCH4Calculator(mol_id)
    if model_name == "ua":
        return sp.get_ua_calc()
    raise ValueError(f"unknown model_name: {model_name}")


def preflight(atoms, model_name):
    pos = atoms.get_positions()
    assert np.all(np.isfinite(pos)), f"[{model_name}] non-finite positions after box build"
    expected = N_MOLECULES * ATOMS_PER_MOL[model_name]
    assert len(atoms) == expected, f"[{model_name}] expected {expected} atoms, got {len(atoms)}"

    _, _, d = neighbor_list("ijd", atoms, 2.0)
    min_d = d.min() if len(d) else np.inf
    L = atoms.cell.lengths()[0]
    log(
        f"[{model_name}] preflight: {len(atoms)} atoms, L={L:.3f} A, "
        f"min pairwise dist (<2.0 A cutoff) = {min_d:.3f} A"
        + ("  [C-H bonds expected ~1.09 A]" if ATOMS_PER_MOL[model_name] > 1 else "")
    )
    log(f"[{model_name}] preflight OK: rho={DENSITY_G_CM3} g/cm3, NDOF={ndof(atoms)}")


def get_pressure_GPa(atoms):
    """Virial + ideal-gas (kinetic) pressure. None if calculator has no stress."""
    try:
        stress = atoms.get_stress(voigt=True, include_ideal_gas=True)
        return -np.mean(stress[:3]) * EV_A3_TO_GPA
    except Exception:
        return None


def smoke_test_all():
    """Build + preflight + calculator + one energy eval for every model
    before any long run, so a broken model fails in seconds."""
    log("=== smoke test: build + preflight + calculator + single energy eval, all models ===")
    for name in MODEL_NAMES:
        atoms, mol_id = build_box(name)
        preflight(atoms, name)
        atoms.calc = get_calc(name, mol_id, DEVICE)
        log(f"[{name}] smoke test OK: E={atoms.get_potential_energy():.4f} eV")
    log("=== smoke test passed for all models ===")


class RateTracker:
    def __init__(self):
        self.t, self.step = None, 0

    def __call__(self, step):
        now = time.time()
        if self.t is None or step == self.step:
            s = "   n/a steps/s"
        else:
            s = f"{(step - self.step) / (now - self.t):6.1f} steps/s"
        self.t, self.step = now, step
        return s


def run_one(model_name):
    log(f"=== {model_name}: building box ===")
    atoms, mol_id = build_box(model_name)
    preflight(atoms, model_name)

    log(f"[{model_name}] attaching calculator...")
    atoms.calc = get_calc(model_name, mol_id, DEVICE)

    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K,
                                 rng=np.random.default_rng(SEED))
    Stationary(atoms)

    # ---------------- NVT equilibration ----------------
    log(
        f"[{model_name}] NVT equilibration: {N_EQUIL_STEPS} steps Langevin @ "
        f"{TEMPERATURE_K} K, friction={LANGEVIN_FRICTION}/fs "
        f"({N_EQUIL_STEPS * DT_FS:.0f} fs, discarded)"
    )
    eq = Langevin(atoms, DT_FS * units.fs, temperature_K=TEMPERATURE_K,
                  friction=LANGEVIN_FRICTION / units.fs,
                  rng=np.random.default_rng(SEED + 1))
    eq_rate = RateTracker()

    def eq_log():
        step = eq.nsteps
        p = get_pressure_GPa(atoms)
        pstr = f"{p:7.3f} GPa" if p is not None else "n/a"
        log(f"[{model_name}] equil {step:5d}/{N_EQUIL_STEPS}  T={temperature_K(atoms):7.2f} K  "
            f"P={pstr}  {eq_rate(step)}")

    eq.attach(eq_log, interval=PRINT_EVERY)
    eq.run(N_EQUIL_STEPS)
    Stationary(atoms)

    # ---------------- NVE production ----------------
    dyn = VelocityVerlet(atoms, timestep=DT_FS * units.fs)
    E0 = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    log(
        f"[{model_name}] equilibrated: E0={E0:.4f} eV, T={temperature_K(atoms):.1f} K -- "
        f"starting NVE, {N_STEPS} steps x {DT_FS} fs = {N_STEPS * DT_FS:.1f} fs"
    )

    traj_path = os.path.join(OUTDIR, f"{model_name}.extxyz")
    if os.path.exists(traj_path):
        os.remove(traj_path)

    frames, energies, temps, pressures, times_fs = [], [], [], [], []
    nve_rate = RateTracker()

    def log_step():
        step = dyn.nsteps
        etot = atoms.get_potential_energy() + atoms.get_kinetic_energy()
        temp = temperature_K(atoms)
        p = get_pressure_GPa(atoms)

        energies.append(etot)
        temps.append(temp)
        pressures.append(p)
        times_fs.append(step * DT_FS)
        frames.append(atoms.copy())
        ase_write(traj_path, atoms, append=True)

        if step % PRINT_EVERY == 0:
            pstr = f"{p:7.3f} GPa" if p is not None else "n/a"
            log(
                f"[{model_name}] NVE {step:5d}/{N_STEPS}  t={step * DT_FS:7.1f} fs  "
                f"dE/mol={(etot - E0) / N_MOLECULES:+.2e} eV  T={temp:7.2f} K  "
                f"P={pstr}  {nve_rate(step)}"
            )

    dyn.attach(log_step, interval=LOG_EVERY)  # dyn.run() calls it at step 0 too
    dyn.run(N_STEPS)

    E = np.array(energies)
    P = np.array([p if p is not None else np.nan for p in pressures])
    T = np.array(temps)
    max_dE = np.max(np.abs(E - E0)) / N_MOLECULES
    log(f"[{model_name}] NVE done. max|dE|/mol={max_dE:.2e} eV, "
        f"final dE/mol={(E[-1] - E0) / N_MOLECULES:+.2e} eV")

    log(f"[{model_name}] computing COM trajectory from {len(frames)} frames...")
    com_images = sp.com_trajectory(frames, N_MOLECULES, ATOMS_PER_MOL[model_name])

    np.savez(os.path.join(OUTDIR, f"{model_name}_trace.npz"),
             times_fs=np.array(times_fs), energies=E, temps=T, pressures=P)
    log(f"[{model_name}] wrote {model_name}_trace.npz ({len(E)} rows)")

    return {"com_images": com_images, "times_fs": np.array(times_fs),
            "energies": E, "E0": E0, "temps": T, "pressures": P,
            "L": atoms.cell.lengths()[0]}


def compute_msd(com_images):
    """Einstein MSD from COM pseudo-Atoms, frame 0 as reference.
    Positions are never wrapped, so no unwrapping is needed."""
    coms = np.array([im.get_positions() for im in com_images])
    disp = coms - coms[0]
    return np.mean(np.sum(disp ** 2, axis=-1), axis=-1)


def main():
    log(f"python {platform.python_version()} on {platform.node()}")
    log(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(
        f"n_mol={N_MOLECULES}, rho={DENSITY_G_CM3} g/cm3, T={TEMPERATURE_K} K, dt={DT_FS} fs, "
        f"equil={N_EQUIL_STEPS} (Langevin), nve={N_STEPS}, log_every={LOG_EVERY}, "
        f"print_every={PRINT_EVERY}, models={MODEL_NAMES}"
    )

    os.makedirs(OUTDIR, exist_ok=True)
    smoke_test_all()

    results = {name: run_one(name) for name in MODEL_NAMES}

    log("=== computing RDFs ===")
    for name, r in results.items():
        rmax = min(10.0, r["L"] / 2 - 0.1)
        r_bins, g_r = sp.compute_rdf(r["com_images"], rmax)
        np.savez(os.path.join(OUTDIR, f"{name}_rdf.npz"), r=r_bins, g_r=g_r)
        log(f"[{name}] wrote {name}_rdf.npz ({len(r_bins)} bins, rmax={rmax:.2f} A)")

    log("=== computing MSDs ===")
    for name, r in results.items():
        msd = compute_msd(r["com_images"])
        np.savez(os.path.join(OUTDIR, f"{name}_msd.npz"), times_fs=r["times_fs"], msd=msd)
        log(f"[{name}] wrote {name}_msd.npz (MSD at {r['times_fs'][-1]:.0f} fs = {msd[-1]:.3f} A^2)")

    log("=== summary (NVE production window) ===")
    for name, r in results.items():
        max_dE = np.max(np.abs(r["energies"] - r["E0"])) / N_MOLECULES
        P = r["pressures"]
        pstr = (f"P={np.nanmean(P):6.3f} +/- {np.nanstd(P):.3f} GPa"
                if np.any(np.isfinite(P)) else "P=n/a")
        log(f"{name:12s}  max|dE|/mol={max_dE:.2e} eV  "
            f"<T>={np.mean(r['temps']):6.1f} K  {pstr}  (target P=0.2 GPa)")

    log(f"=== done. output in {OUTDIR}/ ===")


if __name__ == "__main__":
    main()
