"""Sanity checks against the real MACE-OFF calculator (downloads a checkpoint)."""
import numpy as np
import pytest
from ase.build import molecule

from mace_ch4.calculators import get_calculator

pytestmark = pytest.mark.network


def test_mace_off23_small_gives_finite_energy_and_forces():
    calc = get_calculator(model="off23-small", device="cpu", default_dtype="float64")
    atoms = molecule("CH4")
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert np.isfinite(energy)
    assert np.isfinite(forces).all()
    # A relaxed-ish CH4 molecule should feel forces well below "something is
    # badly broken" magnitude (tens of eV/A would indicate a bad calculator setup).
    assert np.abs(forces).max() < 20.0


def test_mace_off23_forces_match_finite_difference_energy_gradient():
    calc = get_calculator(model="off23-small", device="cpu", default_dtype="float64")
    atoms = molecule("CH4")
    atoms.calc = calc

    analytic_forces = atoms.get_forces()

    delta = 1e-3
    atom_index, cart_index = 0, 0
    displaced = atoms.copy()
    displaced.calc = calc

    positions = displaced.get_positions()
    positions[atom_index, cart_index] += delta
    displaced.set_positions(positions)
    e_plus = displaced.get_potential_energy()

    positions[atom_index, cart_index] -= 2 * delta
    displaced.set_positions(positions)
    e_minus = displaced.get_potential_energy()

    numerical_force = -(e_plus - e_minus) / (2 * delta)
    assert numerical_force == pytest.approx(
        analytic_forces[atom_index, cart_index], abs=0.05
    )
