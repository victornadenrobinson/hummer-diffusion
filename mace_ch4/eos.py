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


# CH4 density (g/cm^3) along the 450 K isotherm from the Setzmann-Wagner
# reference EOS (as on the NIST WebBook; generated with CoolProp), keyed by
# pressure in GPa. Fallback for reference_density_g_cm3 without CoolProp.
NIST_DENSITY_450K_G_CM3 = {
    0.05: 0.17992, 0.1: 0.26473, 0.15: 0.31269, 0.2: 0.34586, 0.25: 0.37132,
    0.3: 0.39208, 0.35: 0.40967, 0.4: 0.42499, 0.45: 0.43858, 0.5: 0.45084,
    0.55: 0.46201, 0.6: 0.47229, 0.65: 0.48183, 0.7: 0.49074, 0.75: 0.49910,
    0.8: 0.50698, 0.85: 0.51445, 0.9: 0.52154, 0.95: 0.52830, 1.0: 0.53477,
}


def reference_density_g_cm3(temperature_K: float, pressure_GPa: float) -> float | None:
    """Experimental-reference CH4 density at (T, P), or None if unavailable.

    Uses CoolProp (Setzmann-Wagner) when installed, for any (T, P) in its
    range. Otherwise interpolates the 450 K table: exact at tabulated
    pressures, within ~3% in between (worst at the low-pressure end).
    """
    try:
        import CoolProp.CoolProp as CP

        return CP.PropsSI("D", "T", temperature_K, "P", pressure_GPa * 1e9, "Methane") / 1e3
    except ImportError:
        pass
    pressures = np.array(sorted(NIST_DENSITY_450K_G_CM3))
    if temperature_K != 450.0 or not pressures[0] <= pressure_GPa <= pressures[-1]:
        return None
    return float(np.interp(pressure_GPa, pressures, [NIST_DENSITY_450K_G_CM3[p] for p in pressures]))


def fit_pressure_vs_density(
    density: np.ndarray, pressure: np.ndarray, stderr: np.ndarray | None = None, degree: int = 3
) -> np.ndarray:
    """Weighted polynomial fit P(rho) along an isotherm; returns np.polyfit coefficients."""
    density = np.asarray(density, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    if len(density) <= degree:
        raise ValueError(f"Need more than {degree} points for a degree-{degree} fit")
    weights = None if stderr is None else 1.0 / np.maximum(np.asarray(stderr, dtype=float), 1e-12)
    return np.polyfit(density, pressure, degree, w=weights)


def density_at_pressure(coeffs: np.ndarray, pressure: float, density_range: tuple[float, float]) -> float:
    """Invert a fitted P(rho) for rho, searching only inside the fitted density range."""
    from scipy.optimize import brentq

    lo, hi = density_range
    f = lambda rho: np.polyval(coeffs, rho) - pressure  # noqa: E731
    if f(lo) * f(hi) > 0:
        raise ValueError(
            f"P={pressure} GPa is outside the fitted range "
            f"[{np.polyval(coeffs, lo):.3f}, {np.polyval(coeffs, hi):.3f}] GPa; extend the density scan"
        )
    return float(brentq(f, lo, hi))
