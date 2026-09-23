import numpy as np
import pytest

from mace_ch4.geometry import (
    ATOMS_PER_MOLECULE,
    box_length_for_density,
    build_methane_box,
    guess_initial_density_kg_m3,
    min_pairwise_distance,
)


def test_build_methane_box_atom_count_and_grouping():
    n_molecules = 8
    density = guess_initial_density_kg_m3(pressure_GPa=0.2)
    atoms = build_methane_box(n_molecules, density, seed=1)

    assert len(atoms) == n_molecules * ATOMS_PER_MOLECULE
    numbers = atoms.get_atomic_numbers().reshape(n_molecules, ATOMS_PER_MOLECULE)
    # Every molecule block must be [C, H, H, H, H].
    for block in numbers:
        assert block[0] == 6
        assert (block[1:] == 1).all()


def test_build_methane_box_is_deterministic_given_seed():
    density = guess_initial_density_kg_m3(pressure_GPa=0.2)
    atoms_a = build_methane_box(27, density, seed=7)
    atoms_b = build_methane_box(27, density, seed=7)
    np.testing.assert_allclose(atoms_a.get_positions(), atoms_b.get_positions())


def test_build_methane_box_different_seeds_differ():
    density = guess_initial_density_kg_m3(pressure_GPa=0.2)
    atoms_a = build_methane_box(27, density, seed=1)
    atoms_b = build_methane_box(27, density, seed=2)
    assert not np.allclose(atoms_a.get_positions(), atoms_b.get_positions())


def test_build_methane_box_no_grossly_overlapping_atoms():
    density = guess_initial_density_kg_m3(pressure_GPa=0.2)
    atoms = build_methane_box(27, density, seed=3)
    # Not a rigorous overlap check (jitter can bring atoms fairly close) --
    # just a guard against a packing bug placing two atoms on top of each other.
    assert min_pairwise_distance(atoms) > 0.3


def test_build_methane_box_rejects_zero_molecules():
    with pytest.raises(ValueError):
        build_methane_box(0, 400.0, seed=0)


def test_box_length_for_density_scales_as_expected():
    length_1 = box_length_for_density(n_molecules=64, density_kg_m3=400.0)
    length_2 = box_length_for_density(n_molecules=64, density_kg_m3=800.0)
    # Doubling the density at fixed molecule count halves the volume,
    # so the box length should shrink by a factor of 2^(1/3).
    assert length_2 == pytest.approx(length_1 / (2 ** (1 / 3)), rel=1e-6)

    length_n64 = box_length_for_density(n_molecules=64, density_kg_m3=400.0)
    length_n128 = box_length_for_density(n_molecules=128, density_kg_m3=400.0)
    assert length_n128 == pytest.approx(length_n64 * (2 ** (1 / 3)), rel=1e-6)
