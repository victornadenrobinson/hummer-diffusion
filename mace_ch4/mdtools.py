"""Shared MD-driver helpers: thermo logging, snapshots, volume rescaling, run summaries."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms, units

from mace_ch4.eos import pressure_GPa
from mace_ch4.unwrap import make_molecules_whole, molecule_centers_of_mass

AMU_A3_TO_G_CM3 = 1.66053906660

def ndof(atoms: Atoms) -> int:
    """3N - 3: total momentum is zeroed (Stationary) before every stage."""
    return 3 * len(atoms) - 3


def temperature_K(atoms: Atoms) -> float:
    return atoms.get_kinetic_energy() / (0.5 * ndof(atoms) * units.kB)


def density_g_cm3(atoms: Atoms) -> float:
    return atoms.get_masses().sum() / atoms.get_volume() * AMU_A3_TO_G_CM3


def snapshot(atoms: Atoms, **info) -> Atoms:
    """Calculator-free copy (positions, momenta, cell) for extxyz output.

    Keeps files small (no per-atom forces) and makes reading a frame back
    restore velocities exactly.
    """
    snap = Atoms(numbers=atoms.get_atomic_numbers(), positions=atoms.get_positions(),
                 cell=atoms.get_cell(), pbc=atoms.pbc)
    snap.set_momenta(atoms.get_momenta())
    snap.info.update(info)
    return snap


def rescale_volume_by_molecule(atoms: Atoms, target_volume: float, atoms_per_molecule: int) -> None:
    """Isotropically set the cell volume, moving molecules rigidly with their COM."""
    factor = (target_volume / atoms.get_volume()) ** (1.0 / 3.0)
    cell = np.array(atoms.get_cell())
    positions = make_molecules_whole(atoms.get_positions(), cell, atoms_per_molecule)
    com = molecule_centers_of_mass(positions, atoms.get_masses(), atoms_per_molecule)
    shift = np.repeat((factor - 1.0) * com, atoms_per_molecule, axis=0)
    atoms.set_cell(cell * factor, scale_atoms=False)
    atoms.set_positions(positions + shift)


class ThermoLog:
    """Fixed-width thermo log, one row per call, with throughput since the last row."""

    COLUMNS = [
        ("step", "{:>9d}"), ("time_ps", "{:>10.4f}"), ("T_K", "{:>8.2f}"),
        ("U_eV", "{:>16.6f}"), ("KE_eV", "{:>11.5f}"), ("Etot_eV", "{:>16.6f}"),
        ("Etot_per_mol_eV", "{:>16.6f}"), ("dEtot_per_mol_meV", "{:>17.4f}"),
        ("P_GPa", "{:>8.4f}"), ("V_A3", "{:>11.3f}"), ("rho_g_cm3", "{:>9.5f}"),
        ("steps_per_s", "{:>11.2f}"),
    ]

    def __init__(self, path: Path, stage: str, n_molecules: int, timestep_fs: float,
                 print_interval: int, header: dict):
        self.stage = stage
        self.n_molecules = n_molecules
        self.timestep_fs = timestep_fs
        self.print_interval = print_interval
        self.e_ref: float | None = None
        self.rows: list[dict] = []
        self._last_wall: float | None = None
        self._last_step = 0
        self._fh = open(path, "w")
        for key, value in header.items():
            self._fh.write(f"# {key}: {value}\n")
        self._fh.write("# dEtot_per_mol_meV is relative to the first row of this stage "
                       "(only a conservation check in NVE)\n")
        self._fh.write("# " + " ".join(name for name, _ in self.COLUMNS) + "\n")
        self._fh.flush()

    def record(self, step: int, atoms: Atoms) -> dict:
        now = time.perf_counter()
        rate = (step - self._last_step) / (now - self._last_wall) if self._last_wall and step > self._last_step else float("nan")
        self._last_wall, self._last_step = now, step

        u = atoms.get_potential_energy()
        ke = atoms.get_kinetic_energy()
        etot = u + ke
        if self.e_ref is None:
            self.e_ref = etot
        row = {
            "step": step,
            "time_ps": step * self.timestep_fs * 1e-3,
            "T_K": temperature_K(atoms),
            "U_eV": u,
            "KE_eV": ke,
            "Etot_eV": etot,
            "Etot_per_mol_eV": etot / self.n_molecules,
            "dEtot_per_mol_meV": 1e3 * (etot - self.e_ref) / self.n_molecules,
            "P_GPa": pressure_GPa(atoms),
            "V_A3": atoms.get_volume(),
            "rho_g_cm3": density_g_cm3(atoms),
            "steps_per_s": rate,
        }
        self.rows.append(row)
        self._fh.write("  " + " ".join(fmt.format(row[name]) for name, fmt in self.COLUMNS) + "\n")
        self._fh.flush()
        if step % self.print_interval == 0:
            log(f"[{self.stage}] step {step:>8d}  t={row['time_ps']:8.3f} ps  T={row['T_K']:7.2f} K  "
                f"P={row['P_GPa']:7.4f} GPa  rho={row['rho_g_cm3']:.5f}  "
                f"dE/mol={row['dEtot_per_mol_meV']:+8.3f} meV  {rate:7.2f} steps/s")
        return row

    def column(self, name: str) -> np.ndarray:
        return np.array([r[name] for r in self.rows])

    def close(self) -> None:
        self._fh.close()


_T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - _T0:9.1f}s] {msg}", flush=True)


def tail_stats(values: np.ndarray, fraction: float) -> tuple[float, float]:
    tail = values[int((1.0 - fraction) * len(values)):]
    return float(np.mean(tail)), float(np.std(tail))


class Summary:
    """summary.json: per-stage results; a stage present here is finished."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.is_file() else {}

    def done(self, stage: str) -> bool:
        return stage in self.data

    def set(self, key: str, value) -> None:
        self.data[key] = value
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)



def block_mean_stderr(values: np.ndarray, n_blocks: int = 10) -> tuple[float, float]:
    """Mean and its standard error from n_blocks contiguous block means.

    Correct for correlated samples when each block is much longer than the
    correlation time (for pressure in a dense fluid, well under 1 ps).
    """
    values = np.asarray(values, dtype=float)
    block_len = len(values) // n_blocks
    if block_len < 1:
        raise ValueError(f"{len(values)} samples is too few for {n_blocks} blocks")
    blocks = values[: block_len * n_blocks].reshape(n_blocks, block_len).mean(axis=1)
    return float(values.mean()), float(blocks.std(ddof=1) / np.sqrt(n_blocks))


def build_relaxed_box(n_molecules: int, density: float, temperature: float, calc, seed: int = 0) -> Atoms:
    """Lattice box of whole CH4 molecules at density (g/cm^3), FIRE-relaxed at fixed cell,
    with Maxwell-Boltzmann velocities at temperature (K) and zero total momentum."""
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from ase.optimize import FIRE

    from mace_ch4.geometry import ATOMS_PER_MOLECULE, build_methane_box, min_pairwise_distance

    atoms = build_methane_box(n_molecules, density * 1e3, seed=seed)
    atoms.set_positions(make_molecules_whole(atoms.get_positions(), np.array(atoms.get_cell()), ATOMS_PER_MOLECULE))
    log(f"[build] {n_molecules} CH4, {len(atoms)} atoms, L={atoms.cell.lengths()[0]:.3f} A, "
        f"rho={density:.5f} g/cm3, min distance {min_pairwise_distance(atoms):.3f} A")
    atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=1.0, steps=500)
    fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    log(f"[build] fixed-cell FIRE done: max |F| = {fmax:.3f} eV/A, min distance {min_pairwise_distance(atoms):.3f} A")
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=np.random.default_rng(seed))
    Stationary(atoms)
    atoms.info["fire_max_force_eV_A"] = fmax
    return atoms
