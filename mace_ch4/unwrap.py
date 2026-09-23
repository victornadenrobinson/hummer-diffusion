"""Molecule center-of-mass extraction and PBC-aware unwrapping."""
from __future__ import annotations

import numpy as np


def molecule_centers_of_mass(
    positions: np.ndarray, masses: np.ndarray, atoms_per_molecule: int
) -> np.ndarray:
    """Mass-weighted center of mass for each molecule.

    Assumes atoms are grouped into contiguous blocks of ``atoms_per_molecule``.
    """
    n_atoms = positions.shape[0]
    if n_atoms % atoms_per_molecule != 0:
        raise ValueError(
            f"{n_atoms} atoms is not divisible by atoms_per_molecule={atoms_per_molecule}"
        )
    n_mol = n_atoms // atoms_per_molecule
    pos = positions.reshape(n_mol, atoms_per_molecule, 3)
    m = masses.reshape(n_mol, atoms_per_molecule, 1)
    return (pos * m).sum(axis=1) / m.sum(axis=1)


def minimum_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Map displacement vectors into the minimum-image convention for ``cell``."""
    frac = delta @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return frac @ cell


class UnwrappedCOMTracker:
    """Accumulates PBC-unwrapped molecule COM trajectories one step at a time.

    Unwraps incrementally (via minimum-image displacement each update) rather
    than post-hoc, so it is robust even if a molecule crosses several box
    images between samples is the only case it cannot handle correctly.
    """

    def __init__(self, initial_com: np.ndarray, cell: np.ndarray):
        self.unwrapped = np.array(initial_com, dtype=float, copy=True)
        self._last_wrapped = np.array(initial_com, dtype=float, copy=True)
        self.cell = np.array(cell, dtype=float, copy=True)

    def update(self, new_wrapped_com: np.ndarray, cell: np.ndarray | None = None) -> np.ndarray:
        if cell is not None:
            self.cell = np.array(cell, dtype=float, copy=True)
        delta = minimum_image(new_wrapped_com - self._last_wrapped, self.cell)
        self.unwrapped = self.unwrapped + delta
        self._last_wrapped = np.array(new_wrapped_com, dtype=float, copy=True)
        return self.unwrapped

    def state_dict(self) -> dict:
        return {
            "unwrapped": self.unwrapped,
            "last_wrapped": self._last_wrapped,
            "cell": self.cell,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "UnwrappedCOMTracker":
        tracker = cls(state["last_wrapped"], state["cell"])
        tracker.unwrapped = np.array(state["unwrapped"], dtype=float, copy=True)
        return tracker
