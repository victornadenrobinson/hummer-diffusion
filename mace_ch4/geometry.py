"""Build an initial periodic box of methane molecules (pure geometry, no calculator)."""
from __future__ import annotations

import math

import numpy as np
from ase import Atoms
from ase.build import molecule as ase_molecule
from scipy.spatial.transform import Rotation

CH4_MOLAR_MASS_KG_MOL = 0.016043
AVOGADRO = 6.02214076e23
ATOMS_PER_MOLECULE = 5  # C + 4H, as returned by ase.build.molecule("CH4")


def guess_initial_density_kg_m3(pressure_GPa: float) -> float:
    """Rough density guess to seed packing; NPT equilibration refines this.

    Anchored to liquid CH4 at its normal boiling point (422.6 kg/m^3, 111.7K,
    1 atm) with a crude linear pressure bump. Not meant to be quantitatively
    accurate -- only good enough that the pre-relaxation step doesn't have to
    close an enormous density gap.
    """
    baseline = 422.6
    return baseline * (1.0 + 0.15 * pressure_GPa)


def box_length_for_density(n_molecules: int, density_kg_m3: float) -> float:
    """Cubic box side length (Angstrom) giving the requested mass density."""
    mass_per_molecule_kg = CH4_MOLAR_MASS_KG_MOL / AVOGADRO
    total_mass_kg = n_molecules * mass_per_molecule_kg
    volume_m3 = total_mass_kg / density_kg_m3
    volume_a3 = volume_m3 * 1e30
    return volume_a3 ** (1.0 / 3.0)


def build_methane_box(n_molecules: int, density_kg_m3: float, seed: int = 0) -> Atoms:
    """Pack n_molecules CH4 molecules on a jittered grid in a cubic PBC box.

    Deterministic given the same (n_molecules, density_kg_m3, seed). Produces
    atoms grouped in contiguous 5-atom (C,H,H,H,H) blocks per molecule, which
    downstream code (mace_ch4.unwrap) relies on.
    """
    if n_molecules < 1:
        raise ValueError("n_molecules must be >= 1")

    rng = np.random.default_rng(seed)
    box_length = box_length_for_density(n_molecules, density_kg_m3)

    template = ase_molecule("CH4")
    template.positions -= template.get_center_of_mass()

    n_side = math.ceil(n_molecules ** (1.0 / 3.0))
    spacing = box_length / n_side
    grid = np.array(
        [(i, j, k) for i in range(n_side) for j in range(n_side) for k in range(n_side)],
        dtype=float,
    )[:n_molecules]
    sites = (grid + 0.5) * spacing

    jitter_scale = 0.15 * spacing
    all_positions = []
    all_numbers = []
    for site in sites:
        rot = Rotation.from_euler("xyz", rng.uniform(0, 360, size=3), degrees=True)
        jitter = rng.uniform(-jitter_scale, jitter_scale, size=3)
        all_positions.append(rot.apply(template.get_positions()) + site + jitter)
        all_numbers.append(template.get_atomic_numbers())

    atoms = Atoms(
        numbers=np.concatenate(all_numbers),
        positions=np.concatenate(all_positions),
        cell=[box_length, box_length, box_length],
        pbc=True,
    )
    atoms.wrap()
    return atoms


def min_pairwise_distance(atoms: Atoms) -> float:
    """Smallest interatomic distance under minimum-image convention."""
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())
