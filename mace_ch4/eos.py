"""Instantaneous pressure from the virial stress, and EOS-point persistence.

Recording pressure during the fixed-volume NVT re-equilibration stage (see
scripts/equilibrate_nvt.py) is a free byproduct: it both confirms the NPT
volume corresponds to the intended target pressure, and accumulates
(T, V, P) points towards an equation of state as more state points are run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import units
from ase.atoms import Atoms

from mace_ch4.records import append_record, load_records

GPA_PER_EV_A3 = 1.0 / (1e9 * units.Pascal)


def pressure_GPa(atoms: Atoms) -> float:
    """Instantaneous hydrostatic pressure from the virial stress, in GPa.

    Positive means compressive, matching the convention used by
    ase.md.nptberendsen.NPTBerendsen's `pressure_au` argument.
    """
    stress = atoms.get_stress(voigt=False, include_ideal_gas=True)
    pressure_eV_A3 = -np.trace(stress) / 3.0
    return float(pressure_eV_A3 * GPA_PER_EV_A3)


def append_eos_point(path: str | Path, record: dict) -> None:
    append_record(path, record)


def load_eos_points(path: str | Path) -> list[dict]:
    return load_records(path)
