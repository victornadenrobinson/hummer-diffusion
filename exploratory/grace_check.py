#!/usr/bin/env python3
"""
grace_check.py -- sanity + speed check of GRACE-OFF on a CH4 box before production.

    source ~/grace-venv/bin/activate
    python grace_check.py                              # 2L-S, 2L-M, 2L-L; float64 and float32
    python grace_check.py --sizes medium --nve-steps 4000

Checks, per model size:
  * float64: forces = -dE/dx and stress = dE/d(strain)/V by central finite
    differences on 32 CH4 at 0.35 g/cm3 (stress is what the pressure/EOS rests on)
  * float32 vs float64 on the same frame: energy/molecule, forces, stress
Then for --n-molecules CH4 (default 128 = 640 atoms), per size and precision:
  * steps/s of Velocity Verlet at 0.5 fs (after warm-up; the first calls
    compile the model), and |dE|/molecule over --nve-steps of NVE after a
    short Langevin melt at 450 K.

Models: GRACE-OFF 2-layer (the published GRACE-OFF, trained on SPICE 2.0 with
the MACE-OFF element/charge filter) from github.com/heid-lab/grace-off:
models/2l/b_off_<size>/seed/1/saved_model (float64) or casted_model (float32).
"""

import argparse
import os
import sys
import time

import numpy as np
from ase import Atoms, units
from ase.build import molecule
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from scipy.spatial.transform import Rotation

AMU_A3_TO_G_CM3 = 1.66053906660


def grace_model_path(models_dir, size, dtype):
    sub = "saved_model" if dtype == "float64" else "casted_model"
    return os.path.join(models_dir, "models", "2l", f"b_off_{size}", "seed", "1", sub)


def get_grace_calc(models_dir, size="medium", dtype="float64"):
    """GRACE-OFF 2L ASE calculator. TensorFlow uses the GPU automatically when visible.
    (tensorpotential >= 0.6 has no device/float_dtype arguments: precision is set by
    which exported model is loaded, and the input builder's dtype must match it.)"""
    import numpy as np
    from tensorpotential.calculator import TPCalculator

    calc = TPCalculator(model=grace_model_path(models_dir, size, dtype))
    if dtype == "float32":
        # tensorpotential 0.6 always builds float64 bond vectors; the float32 export wants float32
        calc.geom_data_builder.float_dtype = np.float32
    return calc


def build_box(n_mol, rho, seed=1):
    rng = np.random.default_rng(seed)
    t = molecule("CH4")
    t.positions -= t.get_center_of_mass()
    L = (n_mol * t.get_masses().sum() * AMU_A3_TO_G_CM3 / rho) ** (1 / 3)
    n = int(np.ceil(n_mol ** (1 / 3)))
    grid = np.array([(i, j, k) for i in range(n) for j in range(n) for k in range(n)], float)
    sites = (grid[np.sort(rng.choice(len(grid), n_mol, replace=False))] + 0.5) * (L / n)
    pos = [Rotation.random(random_state=rng.integers(1 << 31)).apply(t.positions) + s for s in sites]
    return Atoms(numbers=np.tile(t.numbers, n_mol), positions=np.concatenate(pos), cell=[L] * 3, pbc=True)


def fd_checks(calc):
    a = build_box(32, 0.35)
    a.calc = calc
    f, s, V = a.get_forces(), a.get_stress(voigt=False), a.get_volume()
    h, worst_f, worst_s = 1e-4, 0.0, 0.0
    for i, c in [(0, 0), (7, 1), (100, 2)]:
        b = a.copy(); b.calc = calc
        b.positions[i, c] += h; ep = b.get_potential_energy()
        b.positions[i, c] -= 2 * h; em = b.get_potential_energy()
        worst_f = max(worst_f, abs(f[i, c] + (ep - em) / (2 * h)))
    for x, y in [(0, 0), (1, 1), (2, 2), (0, 1)]:
        es = []
        for e in (h, -h):
            b = a.copy(); b.calc = calc
            m = np.eye(3)
            if x == y:
                m[x, x] += e
            else:
                m[x, y] += e / 2; m[y, x] += e / 2
            b.set_cell(np.array(a.cell) @ m, scale_atoms=True)
            es.append(b.get_potential_energy())
        worst_s = max(worst_s, abs(s[x, y] - (es[0] - es[1]) / (2 * h) / V))
    print(f"  FD check (32 CH4): max |F - (-dE/dx)| = {worst_f:.2e} eV/A, "
          f"max |stress - dE/de/V| = {worst_s * 160.2177:.2e} GPa  (both should be ~1e-4 or less)")


def precision_check(calc64, calc32):
    a = build_box(32, 0.35)
    out = []
    for c in (calc64, calc32):
        b = a.copy(); b.calc = c
        out.append((b.get_potential_energy(), b.get_forces(), b.get_stress()))
    print(f"  float32 vs float64 (32 CH4): dE/mol = {1e3 * (out[1][0] - out[0][0]) / 32:+.4f} meV, "
          f"max|dF| = {np.abs(out[1][1] - out[0][1]).max():.1e} eV/A, "
          f"max|d stress| = {160.2177 * np.abs(out[1][2] - out[0][2]).max():.1e} GPa")


def speed_and_nve(calc, n_mol, nve_steps):
    a = build_box(n_mol, 0.35)
    a.calc = calc
    MaxwellBoltzmannDistribution(a, temperature_K=450, rng=np.random.default_rng(0))
    Stationary(a)
    Langevin(a, 0.5 * units.fs, temperature_K=450, friction=0.02 / units.fs, rng=np.random.default_rng(1)).run(400)
    dyn = VelocityVerlet(a, 0.5 * units.fs)
    dyn.run(20)  # warm-up / compile
    E0 = a.get_potential_energy() + a.get_kinetic_energy()
    dE, t0 = [], time.time()
    for _ in range(nve_steps // 50):
        dyn.run(50)
        dE.append(a.get_potential_energy() + a.get_kinetic_energy() - E0)
    rate = nve_steps / (time.time() - t0)
    P = -np.trace(a.get_stress(voigt=False, include_ideal_gas=True)) / 3 / units.GPa
    print(f"  {n_mol} CH4 ({len(a)} atoms): {rate:.1f} steps/s = {rate * 0.5e-6 * 86400:.2f} ns/day at 0.5 fs; "
          f"max|dE|/mol over {nve_steps} NVE steps = {1e3 * np.abs(dE).max() / n_mol:.3f} meV; "
          f"final drift {1e3 * dE[-1] / n_mol:+.3f} meV/mol; P ~ {P:.3f} GPa (450 K, 0.35 g/cm3, 0.2 ps melt)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", default=os.path.expanduser("~/grace-off"))
    p.add_argument("--sizes", nargs="+", default=["small", "medium", "large"], choices=["small", "medium", "large"])
    p.add_argument("--dtypes", nargs="+", default=["float64", "float32"])
    p.add_argument("--n-molecules", type=int, default=128)
    p.add_argument("--nve-steps", type=int, default=2000)
    args = p.parse_args()

    import tensorpotential  # noqa: F401  (must come before tensorflow: sets TF_USE_LEGACY_KERAS)
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print(f"tensorflow {tf.__version__}; GPUs visible: {[g.name for g in gpus] or 'NONE (running on CPU)'}")
    for size in args.sizes:
        calcs = {}
        for dtype in args.dtypes:
            path = grace_model_path(args.models_dir, size, dtype)
            if not os.path.isdir(path):
                sys.exit(f"model not found: {path} (clone github.com/heid-lab/grace-off to --models-dir)")
            print(f"GRACE-OFF 2L-{size[0].upper()} {dtype}: {path}")
            calcs[dtype] = get_grace_calc(args.models_dir, size, dtype)
            if dtype == "float64":
                fd_checks(calcs[dtype])
            elif "float64" in calcs:
                precision_check(calcs["float64"], calcs[dtype])
            speed_and_nve(calcs[dtype], args.n_molecules, args.nve_steps)


if __name__ == "__main__":
    main()
