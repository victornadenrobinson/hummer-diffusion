"""Cheap flexible CH4 model for pipeline tests (not physically tuned).

Harmonic C-H and H-H springs within each molecule (keeps molecules intact
and tetrahedral-ish) plus shifted LJ between carbons of different molecules.
Provides energy, forces and virial stress, so it can stand in for MACE in
MD drivers, barostat included.
"""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

from mace_ch4.unwrap import minimum_image

ATOMS_PER_MOLECULE = 5


class ToyCH4Calculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, k_ch=30.0, r_ch=1.09, k_hh=5.0, r_hh=1.78,
                 epsilon=0.0128, sigma=3.73, cutoff=6.5, **kwargs):
        super().__init__(**kwargs)
        self.k_ch, self.r_ch, self.k_hh, self.r_hh = k_ch, r_ch, k_hh, r_hh
        self.epsilon, self.sigma, self.cutoff = epsilon, sigma, cutoff

    def _springs(self, n_atoms):
        ch, hh = [], []
        for start in range(0, n_atoms, ATOMS_PER_MOLECULE):
            hydrogens = range(start + 1, start + ATOMS_PER_MOLECULE)
            ch += [(start, h) for h in hydrogens]
            hh += [(a, b) for a in hydrogens for b in hydrogens if a < b]
        return np.array(ch), np.array(hh)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = self.atoms.get_positions()
        cell = np.array(self.atoms.get_cell())
        n = len(pos)
        forces = np.zeros_like(pos)
        virial = np.zeros((3, 3))
        energy = 0.0

        def add_pairs(i, j, d, dudr, u):
            # d = r_j - r_i (minimum image); dudr = dU/dr for each pair.
            nonlocal energy
            r = np.linalg.norm(d, axis=1)
            f_j = -(dudr / r)[:, None] * d
            np.add.at(forces, j, f_j)
            np.add.at(forces, i, -f_j)
            virial[:] += np.einsum("pa,pb->ab", d, f_j)
            energy += u.sum()

        ch, hh = self._springs(n)
        for pairs, k, r0 in ((ch, self.k_ch, self.r_ch), (hh, self.k_hh, self.r_hh)):
            i, j = pairs[:, 0], pairs[:, 1]
            d = minimum_image(pos[j] - pos[i], cell)
            r = np.linalg.norm(d, axis=1)
            add_pairs(i, j, d, k * (r - r0), 0.5 * k * (r - r0) ** 2)

        carbons = np.arange(0, n, ATOMS_PER_MOLECULE)
        sub = self.atoms[carbons]
        ci, cj, d = neighbor_list("ijD", sub, self.cutoff)
        keep = ci < cj
        ci, cj, d = ci[keep], cj[keep], d[keep]
        r = np.linalg.norm(d, axis=1)
        sr6 = (self.sigma / r) ** 6
        shift = 4 * self.epsilon * ((self.sigma / self.cutoff) ** 12 - (self.sigma / self.cutoff) ** 6)
        u = 4 * self.epsilon * (sr6 ** 2 - sr6) - shift
        dudr = 4 * self.epsilon * (-12 * sr6 ** 2 + 6 * sr6) / r
        add_pairs(carbons[ci], carbons[cj], d, dudr, u)

        volume = self.atoms.get_volume()
        stress = -virial / volume  # ASE sign: negative = compressive
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
            "stress": stress.flat[[0, 4, 8, 5, 2, 1]],
        }
