"""Yeh-Hummer finite-size correction for self-diffusion coefficients.

Yeh & Hummer, J. Phys. Chem. B 108, 15873 (2004):
    D_PBC(L) = D_0 - kB*T*XI / (6*pi*eta*L)
for a cubic box of side length L. With D_PBC measured at >=2 box sizes,
D_PBC vs 1/L is linear; extrapolating to 1/L -> 0 gives D_0 without needing
an independent viscosity calculation.
"""
from __future__ import annotations

import numpy as np

XI_CUBIC = 2.837297  # Ewald self-term constant for a cubic lattice


def yeh_hummer_extrapolate(box_lengths: np.ndarray, d_pbc: np.ndarray) -> dict:
    """Linearly extrapolate D_PBC(L) vs 1/L to infinite box size.

    Returns dict with D0 (intercept), slope, and the eta implied by the slope
    at the given temperature (if temperature_K is provided).
    """
    box_lengths = np.asarray(box_lengths, dtype=float)
    d_pbc = np.asarray(d_pbc, dtype=float)
    if len(box_lengths) != len(d_pbc):
        raise ValueError("box_lengths and d_pbc must have the same length")
    if len(box_lengths) < 2:
        raise ValueError("Need at least 2 box sizes to extrapolate")
    if np.any(box_lengths <= 0):
        raise ValueError("box_lengths must be positive")
    if np.any(d_pbc <= 0):
        raise ValueError("D_PBC values must be positive (unphysical fit input)")

    inv_l = 1.0 / box_lengths
    slope, intercept = np.polyfit(inv_l, d_pbc, 1)
    return {"D0": intercept, "slope": slope, "inv_L": inv_l}


def implied_viscosity(slope: float, temperature_K: float) -> float:
    """Shear viscosity (eV*fs/A^3) implied by the Yeh-Hummer fit slope.

    slope must be negative (D decreases as 1/L grows); returns eta such that
    slope = -kB*T*XI / (6*pi*eta).
    """
    from ase import units

    if slope >= 0:
        raise ValueError("slope must be negative for a physical Yeh-Hummer fit")
    kb_t = units.kB * temperature_K
    return -kb_t * XI_CUBIC / (6.0 * np.pi * slope)
