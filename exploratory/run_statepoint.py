#!/usr/bin/env python3
"""
run_statepoint.py (v4)

One CH4 state point with MACE-OFF or GRACE-OFF: melt -> NPT -> NVT -> NVE -> self-diffusion D
(Einstein MSD + Green-Kubo VACF). NVE also records the full pressure tensor for
shear viscosity (Green-Kubo / Einstein-Helfand, done by the analysis script).
Restartable.

    python run_statepoint.py --pressure 0.2                                   # MACE-OFF23, 450 K
    python run_statepoint.py --pressure 0.3 --model mace_off24 --temperature 300
    python run_statepoint.py --pressure 0.2 --npt-steps 2000 --nvt-steps 1000 --nve-steps 2000   # quick test

v4 additions:
  --model grace_off_s|m|l   GRACE-OFF 2-layer small/medium/large (the published b_off models from
                            github.com/heid-lab/grace-off, in --grace-dir, default $GRACE_OFF_DIR or
                            ~/grace-off). --dtype float32 (default for GRACE; like the MACE runs) or
                            float64. Needs tensorpotential (~/grace-venv), not torch/MACE: torch and
                            shared_potentials.py are only imported for the MACE models.
  --fixed-density           NVT -> NVE at --density (g/cm3): no NPT. --pressure is then optional
                            and only a label; the reference (S-W EOS) pressure at (T, rho) is
                            recorded as P_target. Default outdir statepoint_<model>_<T>K_rho<rho>.
  --nve-steps 0             stop after NVT: an EOS point (thermo_nvt.log, summary.json 'nvt').
                            Re-running the same outdir with --nve-steps N > 0 adds the NVE later.
  --init-from DIR           start from DIR/restart.npz (a finished NVT or NVE state, any density),
                            rescaled to this run's density by moving whole molecules, keeping the
                            velocities -- instead of a lattice. For chained EOS scans: step the
                            density a few % at a time and use a short --melt-steps (e.g. 2000).

    # EOS-only chain at 300 K, high -> low density, GRACE-OFF 2L-M float32:
    prev=""
    for rho in 0.55 0.50 0.45 0.40 0.35 0.30 0.25 0.20; do
      python run_statepoint.py --model grace_off_m --temperature 300 --fixed-density --density $rho \
          --nvt-steps 30000 --nve-steps 0 ${prev:+--init-from $prev --melt-steps 2000}
      prev=statepoint_grace_off_m_300K_rho$rho
    done

Per run:
  0. lattice box of whole CH4 molecules at the reference density for (T, P):
     reference_density() = built-in Setzmann-Wagner table (200/300/400/450 K,
     0.01-1.0 GPa) -> CoolProp if installed -> nearest-table guess. Override
     with --density. It is only a start: NPT finds the model's own density.
     --show-density prints the reference density for (T, P) and exits.
  1. MELT  Langevin NVT at that density, discarded (lattice -> fluid)
  2. NPT   Berendsen T + isotropic P; box then set to the MEAN volume of the
           last half (molecules moved rigidly with their COM). The barostat's
           compressibility is the reference (EOS) value at (T, P), so the volume
           relaxes on ~TAUP_FS at every pressure (a fixed 1/GPa made it ~0.2 ps at
           1 GPa and ~30 ps at 0.03 GPa); override with --compressibility
  3. NVT   Langevin at that fixed volume: re-thermalise; its mean P checks
           that the volume really gives the target P
  3b. NVE START  the NVE run conserves whatever total energy it starts with, so a
           single NVT end frame gives an NVE temperature off by up to ~2 sigma (~30 K
           for 128 CH4). Instead: continue Langevin (up to --nve-select-steps) until
           the potential energy is within NVE_SELECT_TOL sigma of its NVT mean, remove
           the total momentum, then rescale the molecular translation, rotation and
           vibration velocities SEPARATELY, each to its equipartition share, so that the
           total energy equals the NVT mean total energy (energy exchange between these
           motions is slow in dense CH4: a one-frame imbalance persists for 100+ ps). Logs T_trans/T_rot/T_vib (molecular COM
           translation, rigid rotation, remainder = vibration) before and after.
  4. NVE   Velocity Verlet: full trajectory + COM positions/velocities +
           pressure tensor (virial and kinetic parts, 6 Voigt components)
           every --stress-every steps
           T_trans/T_rot/T_vib recorded every THERMO_EVERY steps (nve_modes.npz) and
           summarised at each checkpoint: in equilibrium all three equal T.
           --nve-thermostat-ps TAU (default 0 = off): weak Langevin (friction 1/TAU)
           instead of pure NVE, to keep the three in equipartition if energy exchange
           between them is slow; perturbs D by ~ tau_v/TAU (tau_v ~ 0.15 ps). CAUTION: it
           does not conserve momentum, so it also damps the hydrodynamic modes behind the
           Yeh-Hummer finite-size effect (momentum crosses a 20 A box in ~20 ps): do not use
           it for the N-series or viscosity runs.
  5. D     Einstein: multi-origin COM MSD slope; the full run and the MSD_BLOCKS blocks
           are fitted over the SAME lag window (MSD_FIT x block length), so the
           block error describes the same regime as the main value
           Green-Kubo: running integral of the COM VACF, averaged over a
           plateau window of lag times (GK_PLATEAU_PS)

Restarts
  OUTDIR/restart.npz is written at the end of melt, NPT and NVT, and every
  --ckpt-every NVE steps (default 20000 = 10 ps). Re-running the SAME command
  with the same OUTDIR resumes from it: finished stages are skipped and NVE
  continues from the exact step (positions + momenta restored). thermo_nve.log
  and nve.extxyz are truncated back to the checkpoint first, so a hard kill
  leaves no duplicate frames. A stage killed part-way is re-run from the
  previous checkpoint.
  --max-wall-hours H: don't start a stage / NVE chunk that would not finish
  within H hours of this job's start (from measured steps/s); exit with code 3
  instead. job_statepoint.sh resubmits itself on exit code 3.
  Raising --nve-steps on a finished run extends its NVE. Checkpoints written by
  the previous version (no pressure tensor) resume fine: the tensor record then
  starts at the step the extension starts (nve_stress.npz says where).
  model / pressure / temperature / n-molecules must match the checkpoint, and
  --stress-every must match an existing tensor record; to start from scratch
  use a new --outdir (or delete the old one).

Outputs in OUTDIR (default statepoint_<model>_<T>K_<P>GPa/):
  thermo_{melt,npt,nvt,nve}.log   every THERMO_EVERY steps: step, t, T, U, KE,
                                  Etot, Etot/mol, dEtot/mol, P, V, rho, steps/s
  {melt,npt,nvt}.extxyz           first + last frame only
  nve.extxyz                      every TRAJ_EVERY steps (positions + momenta)
  nve_com.npz                     unwrapped COMs + COM velocities every COM_EVERY steps
  nve_stress.npz                  pressure tensor P_ab (GPa, Voigt xx yy zz yz xz xy),
                                  virial and kinetic parts separately (total = sum),
                                  every --stress-every steps; P = -stress (ASE sign)
  nve_msd.npz, nve_vacf.npz       MSD(t); VACF(t) and running Green-Kubo D(t)
  summary.json                    densities, pressures, drift, D (both), job history
  restart.npz                     checkpoint

Conventions: T from 3N-3 dof, P = virial + kinetic (include_ideal_gas=True),
energy drift per molecule in meV. D here is the finite-box D_PBC
(Yeh-Hummer correction is done in the analysis script, which needs eta).

Calculators come from shared_potentials.py in SHARED_POTENTIALS_DIR.
"""

import argparse
import functools
import json
import os
import platform
import sys
import time
import warnings

import numpy as np
from ase import Atoms, units  # noqa: E402
from ase.build import molecule  # noqa: E402
from ase.io import write as ase_write  # noqa: E402
from ase.md.langevin import Langevin  # noqa: E402
from ase.md.nptberendsen import NPTBerendsen  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary  # noqa: E402
from ase.md.verlet import VelocityVerlet  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

SHARED_POTENTIALS_DIR = os.environ.get(
    "SHARED_POTENTIALS_DIR",
    "/lustre/scratch/mmm0037/methane/first-test-md-x12t-005",
)
GRACE_OFF_DIR = os.environ.get("GRACE_OFF_DIR", os.path.expanduser("~/grace-off"))
GRACE_SIZES = {"grace_off_s": "small", "grace_off_m": "medium", "grace_off_l": "large"}
MODELS = ("mace_off23", "mace_off24", *GRACE_SIZES)

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
COMPRESSIBILITY_PER_GPA = 1.0   # fallback only: normally the EOS value at (T, P), see reference_compressibility
LANGEVIN_FRICTION = 0.01        # 1/fs
THERMO_EVERY = 100     # steps between thermo-log rows
PRINT_EVERY = 500      # steps between console lines (multiple of THERMO_EVERY)
TRAJ_EVERY = 100       # steps between nve.extxyz frames (50 fs)
COM_EVERY = 10         # steps between COM position/velocity samples (5 fs)
STRESS_EVERY = 2       # steps between pressure-tensor samples (1 fs; C-H stretch period ~11 fs)
CKPT_EVERY = 20_000    # NVE steps between checkpoints (10 ps)
MSD_FIT = (0.1, 0.5)   # fit window, as fractions of the BLOCK length (NVE length / MSD_BLOCKS)
MSD_BLOCKS = 5
GK_PLATEAU_PS = (2.0, 10.0)   # lag window (ps) over which the running GK integral is averaged
NVE_SELECT_STEPS = 4000   # max extra Langevin steps to find an NVE start with typical U
NVE_SELECT_TOL = 0.5      # |U - <U>_NVT| <= this many NVT standard deviations
SEED = 42
DEVICE = "cuda"
EXIT_RESUBMIT = 3      # exit code: stopped cleanly for wall time, re-run to continue

APM = 5  # atoms per molecule (C, H, H, H, H)
EV_A3_TO_GPA = 160.21766208
AMU_A3_TO_G_CM3 = 1.66053906660
A2_FS_TO_M2_S = 1e-5
STAGES = ("melt", "npt", "nvt", "nve")
VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------- reference densities
# Fluid CH4 densities (g/cm3) from the Setzmann & Wagner (1991) EOS -- the reference EOS
# behind the NIST WebBook methane data -- generated with CoolProp 'HEOS::Methane'.
# Reproduces the old NIST_450K table exactly (450 K, 0.2/0.3 GPa -> 0.34586/0.39208).
# Grid: 0.01-0.09 GPa every 0.01, 0.10-1.00 GPa every 0.05 (PCHIP in between); tables stop
# below the EOS melting pressure. Start density only: NPT finds the model's own density.
# reference_density(T, P) -> (rho, source, note): table -> CoolProp (if installed) -> nearest guess.
# T (K) -> {P (GPa): rho (g/cm3)}
DENSITY_TABLE = {
    200: {  # EOS melting line at 0.589 GPa
        0.01: 0.26619,
        0.02: 0.31391,
        0.03: 0.33750,
        0.04: 0.35399,
        0.05: 0.36692,
        0.06: 0.37765,
        0.07: 0.38689,
        0.08: 0.39505,
        0.09: 0.40237,
        0.10: 0.40903,
        0.15: 0.43578,
        0.20: 0.45601,
        0.25: 0.47254,
        0.30: 0.48665,
        0.35: 0.49903,
        0.40: 0.51012,
        0.45: 0.52019,
        0.50: 0.52946,
        0.55: 0.53804,
    },
    300: {  # EOS melting line at 1.378 GPa
        0.01: 0.07518,
        0.02: 0.15528,
        0.03: 0.21076,
        0.04: 0.24646,
        0.05: 0.27169,
        0.06: 0.29101,
        0.07: 0.30665,
        0.08: 0.31980,
        0.09: 0.33115,
        0.10: 0.34116,
        0.15: 0.37892,
        0.20: 0.40552,
        0.25: 0.42635,
        0.30: 0.44362,
        0.35: 0.45846,
        0.40: 0.47153,
        0.45: 0.48325,
        0.50: 0.49390,
        0.55: 0.50368,
        0.60: 0.51274,
        0.65: 0.52119,
        0.70: 0.52912,
        0.75: 0.53660,
        0.80: 0.54369,
        0.85: 0.55042,
        0.90: 0.55685,
        0.95: 0.56299,
        1.00: 0.56889,
    },
    400: {  # EOS melting line at 2.377 GPa
        0.01: 0.04974,
        0.02: 0.09854,
        0.03: 0.14108,
        0.04: 0.17567,
        0.05: 0.20343,
        0.06: 0.22601,
        0.07: 0.24481,
        0.08: 0.26079,
        0.09: 0.27465,
        0.10: 0.28686,
        0.15: 0.33243,
        0.20: 0.36387,
        0.25: 0.38805,
        0.30: 0.40782,
        0.35: 0.42462,
        0.40: 0.43928,
        0.45: 0.45233,
        0.50: 0.46411,
        0.55: 0.47487,
        0.60: 0.48478,
        0.65: 0.49400,
        0.70: 0.50261,
        0.75: 0.51070,
        0.80: 0.51835,
        0.85: 0.52559,
        0.90: 0.53248,
        0.95: 0.53906,
        1.00: 0.54535,
    },
    450: {  # EOS melting line at 2.949 GPa
        0.01: 0.04318,
        0.02: 0.08471,
        0.03: 0.12180,
        0.04: 0.15341,
        0.05: 0.17992,
        0.06: 0.20223,
        0.07: 0.22123,
        0.08: 0.23763,
        0.09: 0.25199,
        0.10: 0.26473,
        0.15: 0.31269,
        0.20: 0.34586,
        0.25: 0.37132,
        0.30: 0.39208,
        0.35: 0.40967,
        0.40: 0.42499,
        0.45: 0.43858,
        0.50: 0.45084,
        0.55: 0.46201,
        0.60: 0.47229,
        0.65: 0.48183,
        0.70: 0.49074,
        0.75: 0.49910,
        0.80: 0.50698,
        0.85: 0.51445,
        0.90: 0.52154,
        0.95: 0.52830,
        1.00: 0.53477,
    },
}

# melting pressure (GPa) from the Setzmann-Wagner melting equation (extrapolated at high T)
MELTING_P_GPA = {
    200: 0.5886,
    300: 1.3778,
    400: 2.3766,
    450: 2.9488,
}

TABLE_TEMPS = tuple(sorted(DENSITY_TABLE))


def _table(T):
    tab = DENSITY_TABLE[T]
    P = np.array(sorted(tab))
    return P, np.array([tab[p] for p in P])


def melting_pressure_GPa(temperature_K):
    T = int(round(temperature_K))
    if T not in MELTING_P_GPA:
        raise KeyError(f"no melting pressure tabulated for T={temperature_K} K (have {TABLE_TEMPS})")
    return MELTING_P_GPA[T]


def nist_density(temperature_K, pressure_GPa):
    """Strict table lookup (g/cm3). Raises ValueError outside the table."""
    T = int(round(temperature_K))
    if abs(temperature_K - T) > 1e-6 or T not in DENSITY_TABLE:
        raise ValueError(f"T={temperature_K} K not tabulated (have {TABLE_TEMPS} K)")
    P, rho = _table(T)
    if not (P[0] - 1e-9 <= pressure_GPa <= P[-1] + 1e-9):
        why = (f"above EOS melting pressure {MELTING_P_GPA[T]:.3f} GPa (likely solid)"
               if pressure_GPa >= MELTING_P_GPA[T] else "outside table/EOS range")
        raise ValueError(f"P={pressure_GPa} GPa at {T} K: {why}; "
                         f"table covers {P[0]:.2f}-{P[-1]:.2f} GPa")
    hit = np.isclose(P, pressure_GPa, atol=1e-9)
    if hit.any():
        return float(rho[hit][0])
    from scipy.interpolate import PchipInterpolator
    return float(PchipInterpolator(P, rho)(pressure_GPa))


def _coolprop_density(T, P_GPa):
    import CoolProp.CoolProp as CP  # ImportError -> caller falls through
    AS = CP.AbstractState("HEOS", "Methane")
    if T <= 625 and P_GPa >= AS.melting_line(CP.iP, CP.iT, T) / 1e9:
        raise ValueError("above EOS melting line")
    return CP.PropsSI("D", "T", float(T), "P", P_GPa * 1e9, "Methane") / 1000.0


def reference_density(temperature_K, pressure_GPa):
    """Best available start density: (rho_g_cm3, source, note). See module doc."""
    if pressure_GPa <= 0:
        raise ValueError(f"pressure must be > 0, got {pressure_GPa}")
    try:
        return nist_density(temperature_K, pressure_GPa), "table", "Setzmann-Wagner table"
    except ValueError as e:
        why = str(e)
    try:
        rho = _coolprop_density(temperature_K, pressure_GPa)
        return rho, "coolprop", f"table miss ({why}); exact S-W EOS via CoolProp"
    except Exception as e2:
        why2 = f"{type(e2).__name__}: {e2}"
    T_near = min(TABLE_TEMPS, key=lambda t: abs(t - temperature_K))
    P, rho = _table(T_near)
    P_use = float(np.clip(pressure_GPa, P[0], P[-1]))
    rho_use = nist_density(T_near, P_use)
    note = (f"table miss ({why}); CoolProp unavailable/refused ({why2}); "
            f"GUESS from {T_near} K at {P_use:.2f} GPa -- NPT must fix the density")
    return rho_use, "nearest", note


def reference_compressibility(temperature_K, pressure_GPa):
    """Isothermal compressibility beta_T = (1/rho) d rho/dP, 1/GPa: (value, source)."""
    def from_table(T, P):
        tab = DENSITY_TABLE[T]
        Ps = np.array(sorted(tab))
        h = 0.005
        lo, hi = max(Ps[0], P - h), min(Ps[-1], P + h)
        if hi - lo < 1e-6:
            raise ValueError("no table width")
        return (nist_density(T, hi) - nist_density(T, lo)) / (hi - lo) / nist_density(T, min(max(P, Ps[0]), Ps[-1]))
    T = int(round(temperature_K))
    try:
        if abs(temperature_K - T) < 1e-6 and T in DENSITY_TABLE:
            Ps = sorted(DENSITY_TABLE[T])
            if Ps[0] <= pressure_GPa <= Ps[-1]:
                return float(from_table(T, pressure_GPa)), "table"
    except Exception:
        pass
    try:
        import CoolProp.CoolProp as CP
        P0, h = pressure_GPa * 1e9, max(1e6, pressure_GPa * 1e9 * 0.01)
        r1 = CP.PropsSI("Dmass", "T", float(temperature_K), "P", P0 - h, "Methane")
        r2 = CP.PropsSI("Dmass", "T", float(temperature_K), "P", P0 + h, "Methane")
        return float((r2 - r1) / (2 * h) / (0.5 * (r1 + r2)) * 1e9), "coolprop"
    except Exception:
        pass
    T_near = min(TABLE_TEMPS, key=lambda t: abs(t - temperature_K))
    Ps = sorted(DENSITY_TABLE[T_near])
    P_use = float(np.clip(pressure_GPa, Ps[0], Ps[-1]))
    return float(from_table(T_near, P_use)), f"nearest ({T_near} K, {P_use:.2f} GPa)"


def mode_temperatures(atoms):
    """(T_trans, T_rot, T_vib) in K from one frame: molecular COM translation (3 dof/mol),
    rigid-body rotation from the angular momentum about the COM (3 dof/mol), and the rest of
    the internal kinetic energy (9 dof/mol). Equal in equilibrium."""
    m = atoms.get_masses().reshape(-1, APM)
    M = m.sum(axis=1)
    x = atoms.get_positions().reshape(-1, APM, 3)
    v = (atoms.get_momenta() / atoms.get_masses()[:, None]).reshape(-1, APM, 3)
    R = (x * m[..., None]).sum(axis=1) / M[:, None]
    V = (v * m[..., None]).sum(axis=1) / M[:, None]
    d, u = x - R[:, None], v - V[:, None]
    L = np.einsum("na,naj->nj", m, np.cross(d, u))
    eye = np.eye(3)
    Iten = np.einsum("na,naij->nij", m, (d ** 2).sum(-1)[..., None, None] * eye - d[..., :, None] * d[..., None, :])
    w = np.linalg.solve(Iten, L[..., None])[..., 0]
    n = len(M)
    Kt = 0.5 * (M * (V ** 2).sum(axis=1)).sum()
    Kr = 0.5 * (w * L).sum()
    Kint = 0.5 * (m[..., None] * u ** 2).sum()
    k = units.kB
    return 2 * Kt / ((3 * n - 3) * k), 2 * Kr / (3 * n * k), 2 * (Kint - Kr) / ((3 * APM - 6) * n * k)


def rescale_modes(atoms, KE_target):
    """Set the translational, rotational and vibrational kinetic energies each to its equipartition
    share of KE_target (dof 3n-3 : 3n : 9n; total 3N-3). The three velocity parts per molecule
    (COM velocity, omega x d, remainder) are mutually orthogonal in the kinetic-energy metric, so
    this is exact. Assumes zero total momentum. Returns the three scale factors."""
    m = atoms.get_masses().reshape(-1, APM)
    M = m.sum(axis=1)
    n = len(M)
    x = atoms.get_positions().reshape(-1, APM, 3)
    v = (atoms.get_momenta() / atoms.get_masses()[:, None]).reshape(-1, APM, 3)
    R = (x * m[..., None]).sum(axis=1) / M[:, None]
    V = (v * m[..., None]).sum(axis=1) / M[:, None]
    d, u = x - R[:, None], v - V[:, None]
    L = np.einsum("na,naj->nj", m, np.cross(d, u))
    Iten = np.einsum("na,naij->nij", m, (d ** 2).sum(-1)[..., None, None] * np.eye(3) - d[..., :, None] * d[..., None, :])
    w = np.linalg.solve(Iten, L[..., None])[..., 0]
    v_rot = np.cross(w[:, None, :], d)
    v_vib = u - v_rot
    K = [0.5 * (M * (V ** 2).sum(1)).sum(), 0.5 * (m[..., None] * v_rot ** 2).sum(), 0.5 * (m[..., None] * v_vib ** 2).sum()]
    dof = np.array([3 * n - 3, 3 * n, (3 * APM - 6) * n], float)
    target = KE_target * dof / dof.sum()
    f = [float(np.sqrt(t / k)) if k > 0 else 1.0 for t, k in zip(target, K)]
    v_new = f[0] * V[:, None] + f[1] * v_rot + f[2] * v_vib
    atoms.set_momenta((v_new * m[..., None]).reshape(-1, 3))
    return f


def fmt_modes(tm):
    return f"T_trans {tm[0]:6.1f}  T_rot {tm[1]:6.1f}  T_vib {tm[2]:6.1f} K"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pressure", type=float, default=None,
                   help="Target pressure, GPa (required unless --fixed-density)")
    p.add_argument("--model", default="mace_off23", choices=MODELS)
    p.add_argument("--dtype", default=None, choices=["float32", "float64"],
                   help="GRACE precision (default float32); MACE precision is set in shared_potentials.py")
    p.add_argument("--grace-dir", default=GRACE_OFF_DIR, help="clone of github.com/heid-lab/grace-off")
    p.add_argument("--fixed-density", action="store_true",
                   help="no NPT: melt -> NVT -> NVE at --density (g/cm3)")
    p.add_argument("--init-from", default=None, metavar="DIR",
                   help="start from DIR/restart.npz (after NVT or NVE), rescaled to this density, instead of a lattice")
    p.add_argument("--density", type=float, default=None,
                   help="Starting density, g/cm3 (default: reference_density(T, P), built in)")
    p.add_argument("--temperature", type=float, default=TEMPERATURE_K, help="K (default 450)")
    p.add_argument("--n-molecules", type=int, default=N_MOLECULES)
    p.add_argument("--melt-steps", type=int, default=MELT_STEPS)
    p.add_argument("--npt-steps", type=int, default=NPT_STEPS)
    p.add_argument("--nvt-steps", type=int, default=NVT_STEPS)
    p.add_argument("--nve-steps", type=int, default=NVE_STEPS)
    p.add_argument("--stress-every", type=int, default=STRESS_EVERY,
                   help="NVE steps between pressure-tensor samples (0 = off)")
    p.add_argument("--ckpt-every", type=int, default=CKPT_EVERY,
                   help="NVE steps between checkpoints (multiple of 100)")
    p.add_argument("--print-every", type=int, default=PRINT_EVERY,
                   help="steps between console lines (multiple of 100)")
    p.add_argument("--max-wall-hours", type=float, default=None,
                   help="stop cleanly (exit 3) before work that would overrun this many hours")
    p.add_argument("--nve-select-steps", type=int, default=NVE_SELECT_STEPS,
                   help="max extra Langevin steps to find an NVE start frame with typical U (0 = last NVT frame)")
    p.add_argument("--nve-thermostat-ps", type=float, default=0.0,
                   help="weak Langevin time constant (ps) during the production run; 0 = pure NVE (default)")
    p.add_argument("--compressibility", type=float, default=None,
                   help="NPT barostat compressibility, 1/GPa (default: EOS value at (T, P))")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--outdir", default=None)
    p.add_argument("--show-density", action="store_true",
                   help="print the reference density for (--temperature, --pressure) and exit")
    return p.parse_args()


def get_calc(model, device, dtype=None, grace_dir=GRACE_OFF_DIR):
    if model in GRACE_SIZES:
        from tensorpotential.calculator import TPCalculator  # TensorFlow picks the GPU itself

        dtype = dtype or "float32"
        path = os.path.join(grace_dir, "models", "2l", f"b_off_{GRACE_SIZES[model]}", "seed", "1",
                            "saved_model" if dtype == "float64" else "casted_model")
        if not os.path.isdir(path):
            sys.exit(f"GRACE-OFF model not found: {path} (clone github.com/heid-lab/grace-off to --grace-dir)")
        calc = TPCalculator(model=path)
        if dtype == "float32":
            # tensorpotential 0.6 always builds float64 bond vectors; the float32 export wants float32
            calc.geom_data_builder.float_dtype = np.float32
        return calc
    import torch

    torch.load = functools.partial(torch.load, weights_only=False)  # MACE checkpoint fix
    sys.path.insert(0, SHARED_POTENTIALS_DIR)
    import shared_potentials as sp

    if model == "mace_off23":
        return sp.get_mace23_calc(device)
    return sp.get_mace24_calc(device)


def device_info(model):
    if model in GRACE_SIZES:
        import tensorpotential  # noqa: F401  (before tensorflow: sets TF_USE_LEGACY_KERAS)
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        return f"tensorflow {tf.__version__} | GPUs {[g.name for g in gpus] or 'none (CPU)'}"
    import torch

    return f"torch {torch.__version__} | cuda {torch.cuda.is_available()}" + (
        f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")


def reference_pressure(temperature_K, rho_g_cm3):
    """Setzmann-Wagner pressure (GPa) at (T, rho): CoolProp, else the density table inverted; nan if neither."""
    try:
        import CoolProp.CoolProp as CP
        return CP.PropsSI("P", "T", float(temperature_K), "Dmass", rho_g_cm3 * 1000, "Methane") / 1e9
    except Exception:
        pass
    T = int(round(temperature_K))
    if abs(temperature_K - T) < 1e-6 and T in DENSITY_TABLE:
        P, rho = _table(T)
        if rho[0] <= rho_g_cm3 <= rho[-1]:
            from scipy.interpolate import PchipInterpolator
            return float(PchipInterpolator(rho, P)(rho_g_cm3))
    return float("nan")


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
    # float(): MACE runs in float32, and numpy float32 values can't go into summary.json
    return float(-np.mean(atoms.get_stress(voigt=True, include_ideal_gas=True)[:3]) * EV_A3_TO_GPA)


def pressure_tensor_GPa(atoms):
    """(P_virial, P_kinetic), each 6 Voigt comps (xx yy zz yz xz xy) in GPa, float64.

    P = -stress (ASE sign). P_kinetic = sum_i p_i p_i^T / (m_i V), which is exactly
    what ASE adds for include_ideal_gas=True, so P_virial + P_kinetic is the total.
    The virial part is the calculator's cached stress (no extra force call).
    """
    p_vir = -np.asarray(atoms.get_stress(voigt=True, include_ideal_gas=False), dtype=np.float64)
    p = atoms.get_momenta()
    K = (p / atoms.get_masses()[:, None]).T @ p / atoms.get_volume()
    p_kin = np.array([K[0, 0], K[1, 1], K[2, 2], K[1, 2], K[0, 2], K[0, 1]])
    return p_vir * EV_A3_TO_GPA, p_kin * EV_A3_TO_GPA


def density(atoms):
    return atoms.get_masses().sum() / atoms.get_volume() * AMU_A3_TO_G_CM3


def mol_coms(atoms):
    pos = atoms.get_positions().reshape(-1, APM, 3)
    m = atoms.get_masses().reshape(-1, APM, 1)
    return (pos * m).sum(axis=1) / m.sum(axis=1)


def mol_com_vels(atoms):
    """COM velocity of each molecule, Angstrom/fs."""
    p = atoms.get_momenta().reshape(-1, APM, 3).sum(axis=1)
    m = atoms.get_masses().reshape(-1, APM).sum(axis=1)[:, None]
    return p / m * units.fs


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


def json_default(o):
    """Let json.dump write numpy scalars/arrays (e.g. float32 values from MACE)."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def fmt_eta(seconds):
    if not np.isfinite(seconds):
        return "--"
    s = int(seconds)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


# ---------------------------------------------------------------- checkpoints
def save_checkpoint(path, atoms, stage, **extra):
    """Atomic write (tmp file + rename): a kill mid-write never corrupts the checkpoint."""
    tmp = path[:-len(".npz")] + ".tmp.npz"
    np.savez(tmp, stage=stage, numbers=atoms.numbers, positions=atoms.get_positions(),
             momenta=atoms.get_momenta(), cell=np.array(atoms.cell), **extra)
    os.replace(tmp, path)


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        ck = {k: z[k] for k in z.files}
    ck["stage"] = str(ck["stage"])
    return ck


def atoms_from_checkpoint(ck):
    atoms = Atoms(numbers=ck["numbers"], positions=ck["positions"], cell=ck["cell"], pbc=True)
    atoms.set_momenta(ck["momenta"])
    return atoms


def read_thermo_rows(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = np.loadtxt(path, comments="#", ndmin=2)
    return [tuple(r) for r in data]


# ---------------------------------------------------------------- thermo log
class Thermo:
    """Thermo log: one row every THERMO_EVERY steps, console line every PRINT_EVERY.

    Calls for a step already logged are ignored, so observers can safely fire
    again at a chunk/restart boundary.
    """

    HEAD = ("step", "time_ps", "T_K", "U_eV", "KE_eV", "Etot_eV", "Etot_per_mol_eV",
            "dEtot_per_mol_meV", "P_GPa", "V_A3", "rho_g_cm3", "steps_per_s")
    FMT = "{:>9d} {:>10.4f} {:>8.2f} {:>16.6f} {:>11.5f} {:>16.6f} {:>16.6f} {:>17.4f} {:>8.4f} {:>11.3f} {:>9.5f} {:>11.2f}\n"

    def __init__(self, path, stage, atoms, n_mol, header, total_steps, resume_rows=None, E0=None):
        self.stage, self.atoms, self.n_mol, self.total = stage, atoms, n_mol, total_steps
        self.t_last = self.t_start = None
        self.step_last = self.step_start = 0
        if resume_rows is None:
            self.rows, self.E0 = [], None
            self.fh = open(path, "w")
            for k, v in header.items():
                self.fh.write(f"# {k}: {v}\n")
            self.fh.write(f"# stage: {stage}; dEtot_per_mol_meV relative to this stage's first row\n")
            self.fh.write("# " + " ".join(self.HEAD) + "\n")
        else:
            self.rows, self.E0 = list(resume_rows), E0
            self.fh = open(path, "a")
        self.last_logged = int(self.rows[-1][0]) if self.rows else -1
        if self.rows:  # resumed: start the rate/ETA clocks at the last logged step (no nan in the first new row)
            self.t_last = self.t_start = time.time()
            self.step_last = self.step_start = self.last_logged

    def __call__(self, step):
        if step <= self.last_logged:
            return
        self.last_logged = step
        a = self.atoms
        now = time.time()
        rate = (step - self.step_last) / (now - self.t_last) if self.t_last else float("nan")
        if self.t_start is None:
            self.t_start, self.step_start = now, step
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
            avg = (step - self.step_start) / (now - self.t_start) if now > self.t_start else float("nan")
            eta = (self.total - step) / avg if avg > 0 else float("nan")
            log(f"[{self.stage}] {step:>8d}/{self.total}  t={row[1]:8.3f} ps  T={row[2]:7.2f} K  "
                f"P={row[8]:7.4f} GPa  rho={row[10]:.5f}  dE/mol={row[7]:+8.3f} meV  "
                f"{rate:7.2f} steps/s  ETA {fmt_eta(eta)}")

    def col(self, name, second_half=False):
        c = np.array([r[self.HEAD.index(name)] for r in self.rows])
        return c[len(c) // 2:] if second_half else c

    def close(self):
        self.fh.close()


def run_stage(stage, dyn, atoms, steps, outdir, header, keep_frames=True):
    thermo = Thermo(os.path.join(outdir, f"thermo_{stage}.log"), stage, atoms, len(atoms) // APM,
                    header, steps)
    if keep_frames:
        ase_write(os.path.join(outdir, f"{stage}.extxyz"), frame(atoms, step=0))
    dyn.attach(lambda: thermo(dyn.nsteps), interval=THERMO_EVERY)
    wall = time.time()
    dyn.run(steps)
    thermo.rate = steps / (time.time() - wall)
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


# ---------------------------------------------------------------- MSD / VACF / D
def msd_multi_origin(com):
    """MSD(t) averaged over molecules and all time origins (FFT, O(N log N)).
    com: (n_frames, n_mol, 3), unwrapped."""
    n = com.shape[0]
    sq = np.sum(com ** 2, axis=2)                                    # (n, mol)
    f = np.fft.rfft(com, n=2 * n, axis=0)
    s2 = np.fft.irfft(np.sum(f * f.conj(), axis=2), n=2 * n, axis=0)[:n]   # sum_d autocorr, (n, mol)
    s2 /= (n - np.arange(n))[:, None]
    sq_pad = np.vstack([sq, np.zeros((1, sq.shape[1]))])
    q = 2 * sq.sum(axis=0)
    s1 = np.empty_like(s2)
    for m in range(n):
        q = q - sq_pad[m - 1] - sq_pad[n - m]
        s1[m] = q / (n - m)
    return (s1 - 2 * s2).mean(axis=1)


def vacf_multi_origin(v):
    """<v(0).v(t)> averaged over molecules and all time origins (FFT).
    v: (n_frames, n_mol, 3)."""
    n = v.shape[0]
    f = np.fft.rfft(v, n=2 * n, axis=0)
    ac = np.fft.irfft(np.sum(f * f.conj(), axis=2), n=2 * n, axis=0)[:n]
    ac /= (n - np.arange(n))[:, None]
    return ac.mean(axis=1)


def fit_D(t_fs, msd, window_fs):
    """D from the MSD slope over lag times window_fs = (lo, hi) in fs; also the log-log slope."""
    m = (t_fs >= window_fs[0]) & (t_fs <= window_fs[1]) & (t_fs > 0)
    if m.sum() < 3:
        return float("nan"), float("nan")
    slope = np.polyfit(t_fs[m], msd[m], 1)[0]
    loglog = np.polyfit(np.log(t_fs[m]), np.log(msd[m]), 1)[0]
    return slope / 6 * A2_FS_TO_M2_S, loglog


# ---------------------------------------------------------------- main
def main():
    global TEMPERATURE_K, PRINT_EVERY
    args = parse_args()
    TEMPERATURE_K = args.temperature
    PRINT_EVERY = args.print_every
    assert PRINT_EVERY % THERMO_EVERY == 0, f"--print-every must be a multiple of {THERMO_EVERY}"
    assert all(args.ckpt_every % k == 0 for k in (COM_EVERY, TRAJ_EVERY, THERMO_EVERY)), \
        f"--ckpt-every must be a multiple of {THERMO_EVERY}"
    assert args.stress_every >= 0 and (args.stress_every == 0 or args.ckpt_every % args.stress_every == 0), \
        "--stress-every must be >= 0 and divide --ckpt-every"

    if args.fixed_density:
        if args.density is None:
            sys.exit("--fixed-density needs --density (g/cm3)")
        ref_P = reference_pressure(TEMPERATURE_K, args.density)
        if args.pressure is None:  # a label only; recorded as P_target
            args.pressure = round(ref_P, 5) if np.isfinite(ref_P) else float("nan")
        ref_rho, ref_src, ref_note = args.density, "fixed", (
            f"fixed density; reference (S-W) P at ({TEMPERATURE_K:g} K, {args.density:g} g/cm3) = {ref_P:.5f} GPa")
    elif args.pressure is None:
        sys.exit("--pressure is required (or use --fixed-density --density RHO)")
    else:
        # reference density: table -> CoolProp -> nearest-table guess (never fails for P > 0)
        ref_rho, ref_src, ref_note = reference_density(TEMPERATURE_K, args.pressure)
    if args.show_density:
        print(f"T={TEMPERATURE_K:g} K, P={args.pressure:g} GPa: {ref_rho:.5f} g/cm3 [{ref_src}] {ref_note}")
        return
    rho0 = args.density if args.density is not None else ref_rho
    outdir = args.outdir or (f"statepoint_{args.model}_{TEMPERATURE_K:g}K_rho{args.density:g}" if args.fixed_density
                             else f"statepoint_{args.model}_{TEMPERATURE_K:g}K_{args.pressure:g}GPa")
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, "restart.npz")
    summary_path = os.path.join(outdir, "summary.json")

    header = {"model": args.model + (f" ({args.dtype or 'float32'})" if args.model in GRACE_SIZES else ""),
              "T_K": TEMPERATURE_K, "P_target_GPa": args.pressure,
              "n_molecules": args.n_molecules, "dt_fs": DT_FS,
              "units": "eV (U = potential), GPa (virial + kinetic), T from 3N-3 dof"}
    log(f"python {platform.python_version()} on {platform.node()} | {device_info(args.model)}")
    log(f"[density] reference {ref_rho:.5f} g/cm3 [{ref_src}]: {ref_note}")
    if args.fixed_density:
        log(f"[density] fixed at {args.density:.5f} g/cm3: no NPT")
    elif args.density:
        log(f"[density] --density {args.density:.5f} g/cm3 overrides it as the start density")
    elif ref_src == "nearest":
        log("[density] WARNING: start density is a guess; check the NPT density converges (thermo_npt.log)")
    log(f"{args.model}, {args.n_molecules} CH4, T={TEMPERATURE_K} K, P={args.pressure} GPa, start rho={rho0:.5f} "
        f"| melt/npt/nvt/nve = {args.melt_steps}/{args.npt_steps}/{args.nvt_steps}/{args.nve_steps} steps x {DT_FS} fs "
        f"| ckpt every {args.ckpt_every} NVE steps | P tensor every "
        f"{args.stress_every if args.stress_every else 'off'} | max wall "
        f"{'%.2f h' % args.max_wall_hours if args.max_wall_hours else 'none'} | out: {outdir}/")

    # ---- fresh start or resume
    ck = load_checkpoint(ckpt_path)
    if ck is None:
        summary = {"config": vars(args) | {"host": platform.node(), "nist_density_g_cm3": ref_rho,
                                           "ref_density_source": ref_src, "ref_density_note": ref_note},
                   "jobs": []}
    else:
        if not os.path.exists(summary_path):
            sys.exit(f"{ckpt_path} exists but {summary_path} is missing: cannot resume safely")
        with open(summary_path) as fh:
            summary = json.load(fh)
        cfg = summary["config"]
        for key in ("model", "pressure", "temperature", "n_molecules", "fixed_density", "dtype"):
            if cfg.get(key, getattr(args, key)) != getattr(args, key) and not (
                    key == "pressure" and cfg[key] != cfg[key] and args.pressure != args.pressure):  # nan labels
                sys.exit(f"checkpoint in {outdir}/ was made with {key}={cfg[key]}, not {getattr(args, key)}: "
                         f"use a different --outdir")
        if args.fixed_density and abs(cfg.get("density", args.density) - args.density) > 1e-9:
            sys.exit(f"checkpoint in {outdir}/ was made at density {cfg['density']}: use a different --outdir")
        if float(cfg.get("nve_thermostat_ps", 0.0)) != args.nve_thermostat_ps:
            sys.exit(f"checkpoint in {outdir}/ was made with --nve-thermostat-ps {cfg.get('nve_thermostat_ps', 0.0)}: "
                     f"re-run with the same value, or use a different --outdir")
        cfg["nve_steps"] = args.nve_steps
        cfg["stress_every"] = args.stress_every
    # density comparisons use the reference recorded at the start of this run
    nist = summary["config"].get("nist_density_g_cm3")
    summary["jobs"].append({"host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                            "start": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "resumed_after": ck["stage"] if ck else None})

    def save_summary():
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2, default=json_default)

    save_summary()
    def _rate(key):
        v = float(ck[key]) if ck is not None and key in ck else float("nan")
        return v if np.isfinite(v) and v > 0 else None
    # measured steps/s, kept in the checkpoint so a resubmitted job can still judge its wall time;
    # NPT (stress + rescaling) runs at ~half the MD rate
    state = {"done": ck["stage"] if ck else None, "md_rate": _rate("md_rate"), "npt_rate": _rate("npt_rate")}

    def completed(stage):
        return state["done"] is not None and STAGES.index(state["done"]) >= STAGES.index(stage)

    def checkpoint(stage, **extra):
        save_checkpoint(ckpt_path, atoms, stage, md_rate=state["md_rate"] or np.nan,
                        npt_rate=state["npt_rate"] or np.nan, **extra)
        state["done"] = stage
        save_summary()

    def out_of_time(steps, what, kind="md"):
        """True if `steps` more steps (at the measured rate, +20%) would overrun --max-wall-hours."""
        if args.max_wall_hours is None:
            return False
        md, npt_ = state["md_rate"], state["npt_rate"]
        rate = (md or (2 * npt_ if npt_ else None)) if kind == "md" else (npt_ or (0.5 * md if md else None))
        if not rate:
            return False
        need = 1.2 * steps / rate
        left = args.max_wall_hours * 3600 - (time.time() - t0)
        if need <= left:
            return False
        log(f"=== stopping for wall time: {what} needs ~{fmt_eta(need)}, {fmt_eta(left)} left. "
            f"Checkpoint is after '{state['done']}'; re-run the same command to continue (exit {EXIT_RESUBMIT}) ===")
        return True

    if ck is None and args.init_from:
        src = args.init_from if args.init_from.endswith(".npz") else os.path.join(args.init_from, "restart.npz")
        ck0 = load_checkpoint(src)
        if ck0 is None or ck0["stage"] not in ("nvt", "nve"):
            sys.exit(f"--init-from: {src} missing or not after NVT/NVE (stage {ck0 and ck0['stage']})")
        atoms = atoms_from_checkpoint(ck0)
        if len(atoms) != args.n_molecules * APM:
            sys.exit(f"--init-from: {src} has {len(atoms) // APM} molecules, not {args.n_molecules}")
        rho_src = density(atoms)
        set_volume_rigid_molecules(atoms, atoms.get_masses().sum() * AMU_A3_TO_G_CM3 / rho0)
        Stationary(atoms)
        atoms.calc = get_calc(args.model, args.device, args.dtype, args.grace_dir)
        summary["config"]["init_from_density_g_cm3"] = rho_src
        log(f"[init] from {src} (stage {ck0['stage']}): rho {rho_src:.5f} -> {density(atoms):.5f} g/cm3 "
            f"({100 * (rho0 / rho_src - 1):+.1f} %), L={atoms.cell.lengths()[0]:.3f} A, velocities kept, "
            f"T={temperature(atoms):.1f} K, P={pressure_GPa(atoms):.3f} GPa")
    elif ck is None:
        atoms = build_box(args.n_molecules, rho0, args.seed)
        d = atoms.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        log(f"[build] {len(atoms)} atoms, L={atoms.cell.lengths()[0]:.3f} A, min distance {d.min():.3f} A")
        atoms.calc = get_calc(args.model, args.device, args.dtype, args.grace_dir)
        log(f"[build] E={atoms.get_potential_energy():.4f} eV, P={pressure_GPa(atoms):.3f} GPa (lattice, no KE)")
        MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=np.random.default_rng(args.seed))
        Stationary(atoms)
    else:
        atoms = atoms_from_checkpoint(ck)
        atoms.calc = get_calc(args.model, args.device, args.dtype, args.grace_dir)
        where = f" (NVE step {int(ck['nve_step'])}/{args.nve_steps})" if ck["stage"] == "nve" else ""
        log(f"[resume] from checkpoint after '{ck['stage']}'{where}: {len(atoms)} atoms, "
            f"L={atoms.cell.lengths()[0]:.4f} A, rho={density(atoms):.5f}, T={temperature(atoms):.1f} K")

    n_mol = len(atoms) // APM

    def langevin(seed_offset):
        return Langevin(atoms, DT_FS * units.fs, temperature_K=TEMPERATURE_K,
                        friction=LANGEVIN_FRICTION / units.fs, rng=np.random.default_rng(args.seed + seed_offset))

    def prepare_nve_start(E_nvt, U_nvt, select):
        """Pick a frame with typical U (optional extra Langevin), zero the total momentum, and rescale
        velocities so E_tot = <E_tot>_NVT. Returns a summary dict."""
        E_target, U_mean, U_std = float(np.mean(E_nvt)), float(np.mean(U_nvt)), float(np.std(U_nvt))
        info = {"E_target_eV": E_target, "E_target_per_mol_eV": E_target / n_mol, "U_mean_eV": U_mean,
                "U_std_eV": U_std, "E_std_per_mol_meV": 1000 * float(np.std(E_nvt)) / n_mol}
        log(f"[nve-start] NVT last half: <Etot>/mol = {E_target / n_mol:.5f} eV (std "
            f"{info['E_std_per_mol_meV']:.2f} meV), <U> = {U_mean:.3f} +/- {U_std:.3f} eV")
        log(f"[nve-start] modes before: {fmt_modes(mode_temperatures(atoms))} (one frame: +/- ~"
            f"{TEMPERATURE_K * np.sqrt(2 / (3 * n_mol)):.0f} / {TEMPERATURE_K * np.sqrt(2 / (3 * n_mol)):.0f} / "
            f"{TEMPERATURE_K * np.sqrt(2 / (9 * n_mol)):.0f} K noise)")
        extra = 0
        if select and args.nve_select_steps > 0 and U_std > 0:
            dyn = langevin(3)
            while True:
                dev = (atoms.get_potential_energy() - U_mean) / U_std
                if abs(dev) <= NVE_SELECT_TOL or extra >= args.nve_select_steps:
                    break
                dyn.run(100)
                extra += 100
            ok = abs(dev) <= NVE_SELECT_TOL
            log(f"[nve-start] {'selected' if ok else 'no frame within tolerance; using'} frame after {extra} extra "
                f"Langevin steps: U - <U> = {dev:+.2f} sigma (tolerance {NVE_SELECT_TOL})")
            info.update(select_extra_steps=extra, select_U_dev_sigma=float(dev), select_ok=bool(ok))
        Stationary(atoms)
        U0, KE0, T0 = atoms.get_potential_energy(), atoms.get_kinetic_energy(), temperature(atoms)
        KE_t = E_target - U0
        T_t = KE_t / (0.5 * ndof(atoms) * units.kB)
        if KE_t > 0 and abs(T_t / TEMPERATURE_K - 1) < 0.10:
            # each motion rescaled separately: energy exchange between translation/rotation and the
            # internal vibrations can be slow (100s of ps), so a one-frame imbalance would persist in NVE
            f = rescale_modes(atoms, KE_t)
            log(f"[nve-start] rescaled translation/rotation/vibration x{f[0]:.4f}/{f[1]:.4f}/{f[2]:.4f}: "
                f"T {T0:.1f} -> {temperature(atoms):.1f} K, Etot/mol {(U0 + KE0) / n_mol:.5f} -> "
                f"{(U0 + atoms.get_kinetic_energy()) / n_mol:.5f} eV (target)")
            info.update(rescale_factors_trans_rot_vib=f, T_before_K=T0, T_after_K=temperature(atoms))
        else:
            log(f"[nve-start] WARNING: target KE implies T = {T_t:.1f} K (>10% from {TEMPERATURE_K} K): "
                f"not rescaling; NVE starts at T = {T0:.1f} K")
            info.update(rescale_factor=None, T_before_K=T0, T_after_K=T0)
        tm = mode_temperatures(atoms)
        log(f"[nve-start] modes after : {fmt_modes(tm)}")
        info["modes_after_K"] = list(tm)
        return info

    # ---- melt
    if not completed("melt"):
        log(f"[melt] {args.melt_steps} steps Langevin at rho={rho0:.5f}")
        th = run_stage("melt", langevin(1), atoms, args.melt_steps, outdir, header)
        state["md_rate"] = th.rate
        checkpoint("melt")
        log(f"[melt] done ({th.rate:.1f} steps/s), checkpoint written")

    # ---- NPT (skipped at fixed density)
    if not completed("npt") and args.fixed_density:
        summary["npt"] = {"skipped": "fixed density", "density_g_cm3": density(atoms),
                          "box_length_A": atoms.cell.lengths()[0]}
        checkpoint("npt")
        log(f"[npt] skipped (--fixed-density): rho = {density(atoms):.5f} g/cm3, L = {atoms.cell.lengths()[0]:.4f} A")
    if not completed("npt"):
        if out_of_time(args.npt_steps, "NPT", kind="npt"):
            sys.exit(EXIT_RESUBMIT)
        Stationary(atoms)
        if args.compressibility is not None:
            beta, beta_src = args.compressibility, "--compressibility"
        else:
            try:
                beta, beta_src = reference_compressibility(TEMPERATURE_K, args.pressure)
            except Exception as e:
                beta, beta_src = COMPRESSIBILITY_PER_GPA, f"fallback constant ({type(e).__name__}: {e})"
            beta = float(np.clip(beta, 0.05, 100.0))
        log(f"[npt] {args.npt_steps} steps Berendsen, taut={TAUT_FS} fs, taup={TAUP_FS} fs, "
            f"compressibility {beta:.4g} /GPa [{beta_src}]")
        npt = NPTBerendsen(atoms, timestep=DT_FS * units.fs, temperature_K=TEMPERATURE_K,
                           pressure_au=args.pressure * units.GPa, taut=TAUT_FS * units.fs, taup=TAUP_FS * units.fs,
                           compressibility_au=beta / units.GPa)
        th = run_stage("npt", npt, atoms, args.npt_steps, outdir, header)
        state["npt_rate"] = th.rate
        V_mean = th.col("V_A3", True).mean()
        rho_mean, rho_std = th.col("rho_g_cm3", True).mean(), th.col("rho_g_cm3", True).std()
        set_volume_rigid_molecules(atoms, V_mean)
        summary["npt"] = {"mean_density_g_cm3": rho_mean, "std_density_g_cm3": rho_std,
                          "density_vs_nist_percent": 100 * (rho_mean / nist - 1) if nist else None,
                          "mean_P_GPa": th.col("P_GPa", True).mean(), "mean_T_K": th.col("T_K", True).mean(),
                          "mean_volume_A3": V_mean, "box_length_A": atoms.cell.lengths()[0],
                          "barostat_compressibility_per_GPa": beta, "compressibility_source": beta_src}
        checkpoint("npt")
        log(f"[npt] last half: <rho> = {rho_mean:.5f} +/- {rho_std:.5f} g/cm3"
            + (f" (ref {nist:.5f}, {summary['npt']['density_vs_nist_percent']:+.2f}%)" if nist else "")
            + f" -> box set to L = {atoms.cell.lengths()[0]:.4f} A; checkpoint written")

    # ---- NVT
    if not completed("nvt"):
        if out_of_time(args.nvt_steps + args.nve_select_steps, "NVT"):
            sys.exit(EXIT_RESUBMIT)
        Stationary(atoms)
        log(f"[nvt] {args.nvt_steps} steps Langevin at fixed V")
        th = run_stage("nvt", langevin(2), atoms, args.nvt_steps, outdir, header)
        state["md_rate"] = th.rate
        P = th.col("P_GPa", True)
        summary["nvt"] = {"mean_P_GPa": P.mean(), "stderr_P_GPa": block_stderr(P), "std_P_GPa": P.std(),
                          "mean_T_K": th.col("T_K", True).mean(), "density_g_cm3": density(atoms),
                          "reference_P_GPa_at_density": reference_pressure(TEMPERATURE_K, density(atoms))}
        log(f"[nvt] last half: <P> = {P.mean():.4f} +/- {block_stderr(P):.4f} GPa (target {args.pressure}; "
            f"S-W EOS at this density {summary['nvt']['reference_P_GPa_at_density']:.4f}), "
            f"<T> = {summary['nvt']['mean_T_K']:.2f} K, rho = {density(atoms):.5f} g/cm3")
        if args.nve_steps > 0:
            summary["nvt"]["nve_start"] = prepare_nve_start(th.col("Etot_eV", True), th.col("U_eV", True),
                                                            select=True)
        checkpoint("nvt")
        log("[nvt] " + ("NVE start prepared; " if args.nve_steps > 0 else "") + "checkpoint written")

    if args.nve_steps == 0:
        log(f"=== --nve-steps 0: stopping after NVT (EOS point in {summary_path}). "
            f"Re-run with --nve-steps N to add the NVE. ===")
        return

    # ---- NVE (checkpointed in chunks)
    traj_path = os.path.join(outdir, "nve.extxyz")
    thermo_nve_path = os.path.join(outdir, "thermo_nve.log")
    if state["done"] == "nve":
        nve_step = int(ck["nve_step"])
        for path, key in ((traj_path, "traj_bytes"), (thermo_nve_path, "thermo_bytes")):
            if os.path.exists(path) and os.path.getsize(path) > int(ck[key]):
                os.truncate(path, int(ck[key]))
        com_t, com, com_v = list(ck["com_t"]), list(ck["com"]), list(ck["com_v"])
        E0, nve_wall, traj_last = float(ck["nve_E0"]), float(ck["nve_wall_s"]), int(ck["traj_last_step"])
        rows = read_thermo_rows(thermo_nve_path)
        state["md_rate"] = nve_step / nve_wall if nve_wall > 0 else state["md_rate"]
        # pressure tensor record (absent in checkpoints from the previous version)
        if "stress_t" in ck and len(ck["stress_t"]):
            ck_every = int(ck["stress_every"])
            if ck_every != args.stress_every:
                sys.exit(f"checkpoint has a pressure-tensor record every {ck_every} steps; "
                         f"re-run with --stress-every {ck_every}")
            st_t, st_vir, st_kin = list(ck["stress_t"]), list(ck["stress_vir"]), list(ck["stress_kin"])
            log(f"[nve] pressure-tensor record restored: {len(st_t)} samples from "
                f"{st_t[0] / 1000:.3f} to {st_t[-1] / 1000:.3f} ps")
        else:
            st_t, st_vir, st_kin = [], [], []
            if args.stress_every:
                log(f"[nve] checkpoint has no pressure-tensor record (older version): "
                    f"recording starts at NVE step {nve_step} ({nve_step * DT_FS / 1000:.2f} ps)")
    else:
        nve_step, com_t, com, com_v, E0, nve_wall, traj_last, rows = 0, [], [], [], None, 0.0, -1, None
        st_t, st_vir, st_kin = [], [], []
        if "nve_start" not in summary.get("nvt", {}):
            # checkpoint from an older version: NVE start was not prepared; do it now from thermo_nvt.log
            nvt_rows = read_thermo_rows(os.path.join(outdir, "thermo_nvt.log"))
            h = len(nvt_rows) // 2
            log("[nve-start] older checkpoint: preparing the NVE start from thermo_nvt.log (no frame selection)")
            summary.setdefault("nvt", {})["nve_start"] = prepare_nve_start(
                [r[Thermo.HEAD.index("Etot_eV")] for r in nvt_rows[h:]],
                [r[Thermo.HEAD.index("U_eV")] for r in nvt_rows[h:]], select=False)
            save_summary()
        if os.path.exists(traj_path):
            os.remove(traj_path)

    modes = ([list(r) for r in ck["modes"]]
             if ck is not None and ck["stage"] == "nve" and state["done"] == "nve" and "modes" in ck else [])
    th = Thermo(thermo_nve_path, "nve", atoms, n_mol, header, args.nve_steps, resume_rows=rows, E0=E0)
    if args.nve_thermostat_ps > 0:
        nve = Langevin(atoms, DT_FS * units.fs, temperature_K=TEMPERATURE_K,
                       friction=1.0 / (args.nve_thermostat_ps * 1000 * units.fs),
                       rng=np.random.default_rng(args.seed + 1000 + nve_step))
        log(f"[nve] production with weak Langevin, tau = {args.nve_thermostat_ps:g} ps "
            f"(dEtot/mol in the log is then not a conservation check)")
    else:
        nve = VelocityVerlet(atoms, timestep=DT_FS * units.fs)
    nve.nsteps = nve_step
    last_frame = [traj_last]

    def record_com():
        s = nve.nsteps
        if com_t and s * DT_FS <= com_t[-1]:
            return
        com_t.append(s * DT_FS)
        com.append(mol_coms(atoms))
        com_v.append(mol_com_vels(atoms))

    def record_frame():
        s = nve.nsteps
        if s <= last_frame[0]:
            return
        last_frame[0] = s
        ase_write(traj_path, frame(atoms, step=s, time_fs=s * DT_FS), append=True)

    def record_stress():
        s = nve.nsteps
        if st_t and s * DT_FS <= st_t[-1]:
            return
        p_vir, p_kin = pressure_tensor_GPa(atoms)
        if not st_t:  # one-off consistency check against the scalar P used everywhere else
            P_tensor = float((p_vir[:3] + p_kin[:3]).mean())
            log(f"[nve] pressure-tensor check at step {s}: tr/3 = {P_tensor:.6f} GPa vs "
                f"pressure_GPa() = {pressure_GPa(atoms):.6f} GPa; kinetic part tr/3 = {p_kin[:3].mean():.4f} GPa")
        st_t.append(s * DT_FS)
        st_vir.append(p_vir)
        st_kin.append(p_kin)

    def record_modes():
        s = nve.nsteps
        if modes and s * DT_FS <= modes[-1][0]:
            return
        modes.append([s * DT_FS, *mode_temperatures(atoms)])

    nve.attach(record_com, interval=COM_EVERY)
    nve.attach(record_modes, interval=THERMO_EVERY)
    nve.attach(record_frame, interval=TRAJ_EVERY)
    if args.stress_every:
        nve.attach(record_stress, interval=args.stress_every)
    nve.attach(lambda: th(nve.nsteps), interval=THERMO_EVERY)

    def save_stress_npz():
        if not st_t:
            return
        np.savez(os.path.join(outdir, "nve_stress.npz"), times_fs=np.array(st_t),
                 P_virial_GPa=np.array(st_vir), P_kinetic_GPa=np.array(st_kin),
                 voigt_order=np.array(VOIGT), volume_A3=atoms.get_volume(),
                 T_target_K=TEMPERATURE_K, stress_every_steps=args.stress_every, dt_fs=DT_FS,
                 note="P = -stress (ASE sign); total = virial + kinetic; atomic (not molecular) tensor")

    if nve_step < args.nve_steps:
        log(f"[nve] {'resuming at step %d of ' % nve_step if nve_step else ''}{args.nve_steps} steps "
            f"({args.nve_steps * DT_FS / 1000:.1f} ps) "
            f"{'weak Langevin (tau %g ps)' % args.nve_thermostat_ps if args.nve_thermostat_ps > 0 else 'Velocity Verlet'}; "
            f"extxyz every {TRAJ_EVERY}, "
            f"COM every {COM_EVERY}, P tensor every {args.stress_every if args.stress_every else '-'}, "
            f"checkpoint every {args.ckpt_every}")
    while nve_step < args.nve_steps:
        n = min(args.ckpt_every - nve_step % args.ckpt_every, args.nve_steps - nve_step)
        if out_of_time(n, f"NVE steps {nve_step}-{nve_step + n}"):
            th.close()
            sys.exit(EXIT_RESUBMIT)
        tc = time.time()
        nve.run(n)
        dt = time.time() - tc
        if nve.nsteps != nve_step + n:
            sys.exit(f"internal error: expected NVE step {nve_step + n}, ASE reports {nve.nsteps}")
        nve_step, nve_wall = nve.nsteps, nve_wall + dt
        state["md_rate"] = n / dt
        checkpoint("nve", nve_step=nve_step, nve_E0=th.E0, nve_wall_s=nve_wall,
                   modes=np.array(modes).reshape(-1, 4),
                   com_t=np.array(com_t), com=np.array(com), com_v=np.array(com_v),
                   stress_t=np.array(st_t), stress_vir=np.array(st_vir).reshape(-1, 6),
                   stress_kin=np.array(st_kin).reshape(-1, 6), stress_every=args.stress_every,
                   traj_last_step=last_frame[0], traj_bytes=os.path.getsize(traj_path),
                   thermo_bytes=os.path.getsize(thermo_nve_path))
        chunk = np.array([r[1:] for r in modes if r[0] > (nve_step - n) * DT_FS])
        if len(chunk):
            log(f"[nve] last {n * DT_FS / 1000:.1f} ps: {fmt_modes(chunk.mean(axis=0))}")
        log(f"[nve] checkpoint at step {nve_step}/{args.nve_steps} ({nve_step * DT_FS / 1000:.2f} ps)"
            + (f", {len(st_t)} P-tensor samples" if st_t else ""))
    th.close()
    if nve_step > args.nve_steps:
        log(f"[nve] note: checkpoint already has {nve_step} NVE steps (> --nve-steps); analysing all of them")

    modes_arr = np.array(modes).reshape(-1, 4)
    np.savez(os.path.join(outdir, "nve_modes.npz"), times_fs=modes_arr[:, 0], T_trans_K=modes_arr[:, 1],
             T_rot_K=modes_arr[:, 2], T_vib_K=modes_arr[:, 3],
             note="molecular translation / rigid rotation / remaining internal (vibration) kinetic temperatures")
    if len(modes_arr):
        q = max(1, len(modes_arr) // 5)
        first, last, allm = modes_arr[:q, 1:].mean(0), modes_arr[-q:, 1:].mean(0), modes_arr[:, 1:].mean(0)
        log(f"[nve] modes, whole run : {fmt_modes(allm)}")
        log(f"[nve] modes, first 20% : {fmt_modes(first)}")
        log(f"[nve] modes, last 20%  : {fmt_modes(last)}")
        spread = (allm.max() - allm.min()) / allm.mean()
        if spread > 0.03:
            log(f"[nve] WARNING: translation/rotation/vibration temperatures differ by {100 * spread:.1f} %: "
                f"not in equipartition (slow energy exchange?); D reflects T_trans = {allm[0]:.1f} K")
    com_t, com, com_v = np.array(com_t), np.array(com), np.array(com_v, dtype=float)
    np.savez(os.path.join(outdir, "nve_com.npz"), times_fs=com_t, unwrapped_com=com, com_vel_A_fs=com_v)
    save_stress_npz()
    if st_t:
        log(f"[nve] wrote nve_stress.npz: {len(st_t)} samples every {args.stress_every} steps, "
            f"{st_t[0] / 1000:.3f}-{st_t[-1] / 1000:.3f} ps ({(st_t[-1] - st_t[0]) / 1000:.1f} ps)")

    dE = th.col("dEtot_per_mol_meV")
    tp = th.col("time_ps")
    T_nve = th.col("T_K").mean()
    summary["nve"] = {"n_steps": nve_step, "mean_T_K": T_nve, "mean_P_GPa": th.col("P_GPa").mean(),
                      "std_P_GPa": th.col("P_GPa").std(), "max_abs_dE_per_mol_meV": np.abs(dE).max(),
                      "drift_meV_per_mol_per_ps": np.polyfit(tp, dE, 1)[0],
                      "steps_per_s": nve_step / nve_wall if nve_wall else float("nan"), "wall_h": nve_wall / 3600,
                      "mode_temperatures_K": ({"T_trans": modes_arr[:, 1].mean(), "T_rot": modes_arr[:, 2].mean(),
                                               "T_vib": modes_arr[:, 3].mean()} if len(modes_arr) else None),
                      "thermostat_tau_ps": args.nve_thermostat_ps or None,
                      "stress_record_ps": [st_t[0] / 1000, st_t[-1] / 1000] if st_t else None}
    save_summary()
    log(f"[nve] done: <T> = {T_nve:.2f} K, <P> = {summary['nve']['mean_P_GPa']:.4f} GPa, "
        f"max|dE|/mol = {summary['nve']['max_abs_dE_per_mol_meV']:.3f} meV, "
        f"drift {summary['nve']['drift_meV_per_mol_per_ps']:+.4f} meV/mol/ps, "
        f"{summary['nve']['steps_per_s']:.1f} steps/s over {summary['nve']['wall_h']:.2f} h of NVE")

    # ---- D from COM MSD (Einstein)
    t = com_t - com_t[0]
    msd = msd_multi_origin(com - com[0])
    np.savez(os.path.join(outdir, "nve_msd.npz"), times_fs=t, msd_A2=msd)
    T_blk = t[-1] / MSD_BLOCKS
    win_fs = (MSD_FIT[0] * T_blk, MSD_FIT[1] * T_blk)   # same lag window for the full run and the blocks
    D, loglog = fit_D(t, msd, win_fs)
    blocks = []
    for blk in np.array_split(np.arange(len(t)), MSD_BLOCKS):
        blocks.append(fit_D(t[blk] - t[blk[0]], msd_multi_origin(com[blk] - com[blk[0]]), win_fs)[0])
    summary["diffusion"] = {"D_PBC_m2_s": D, "block_mean_m2_s": np.mean(blocks),
                            "block_stderr_m2_s": np.std(blocks, ddof=1) / np.sqrt(MSD_BLOCKS),
                            "loglog_slope_in_fit_window": loglog, "fit_window_ps": [w / 1000 for w in win_fs],
                            "box_length_A": atoms.cell.lengths()[0]}
    save_summary()
    log(f"[D] D_PBC = {D:.4e} m^2/s, blocks {np.mean(blocks):.4e} +/- "
        f"{summary['diffusion']['block_stderr_m2_s']:.1e} (stderr, {MSD_BLOCKS}); fit {win_fs[0] / 1000:.3g}-"
        f"{win_fs[1] / 1000:.3g} ps, log-log slope {loglog:.3f}"
        + ("" if 0.9 < loglog < 1.1 else "  <-- not diffusive yet: run NVE longer"))

    # ---- D from COM VACF (Green-Kubo)
    vacf = vacf_multi_origin(com_v)
    D_run = np.concatenate([[0.0], np.cumsum(0.5 * (vacf[1:] + vacf[:-1]) * np.diff(t))]) / 3 * A2_FS_TO_M2_S
    np.savez(os.path.join(outdir, "nve_vacf.npz"), times_fs=t, vacf_A2_fs2=vacf, D_running_m2_s=D_run)
    win = (t / 1000 >= GK_PLATEAU_PS[0]) & (t / 1000 <= GK_PLATEAU_PS[1])
    D_gk = float(D_run[win].mean()) if win.sum() >= 2 else float("nan")
    m_mol = atoms.get_masses()[:APM].sum()
    v2_expected = 3 * units.kB * T_nve / m_mol * units.fs ** 2
    summary["green_kubo"] = {"D_GK_m2_s": D_gk, "plateau_window_ps": GK_PLATEAU_PS,
                             "vacf0_A2_fs2": vacf[0], "vacf0_expected_3kT_over_m": v2_expected,
                             "D_GK_over_D_MSD": D_gk / D if np.isfinite(D_gk) else None}
    save_summary()
    log(f"[GK] <v(0)^2> = {vacf[0]:.4e} A^2/fs^2 (3kT/m at <T>: {v2_expected:.4e})")
    if np.isfinite(D_gk):
        log(f"[GK] D_GK = {D_gk:.4e} m^2/s (running integral averaged over {GK_PLATEAU_PS[0]:g}-"
            f"{GK_PLATEAU_PS[1]:g} ps lag); D_GK / D_MSD = {D_gk / D:.3f}")
    else:
        log(f"[GK] NVE too short for the {GK_PLATEAU_PS[0]:g}-{GK_PLATEAU_PS[1]:g} ps plateau window: no D_GK")
    log(f"=== done. output in {outdir}/ ===")


if __name__ == "__main__":
    main()
