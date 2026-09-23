import numpy as np
import pytest
from ase import units
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mace_ch4.eos import append_eos_point, load_eos_points, pressure_GPa


def _atoms_with_known_stress(pressure_eV_A3: float) -> Atoms:
    # Stress = -P * I (voigt: xx, yy, zz, yz, xz, xy) is a pure hydrostatic
    # pressure P with zero shear -- the standard sign convention (compressive
    # pressure is positive stress is negative).
    atoms = Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 10.0, pbc=True)
    stress_voigt = np.array([-pressure_eV_A3] * 3 + [0.0, 0.0, 0.0])
    atoms.calc = SinglePointCalculator(atoms, stress=stress_voigt, energy=0.0, forces=np.zeros((1, 3)))
    return atoms


def test_pressure_GPa_recovers_known_hydrostatic_pressure():
    pressure_eV_A3 = 0.01
    atoms = _atoms_with_known_stress(pressure_eV_A3)

    expected_GPa = pressure_eV_A3 / (1e9 * units.Pascal)
    assert pressure_GPa(atoms) == pytest.approx(expected_GPa, rel=1e-10)


def test_pressure_GPa_zero_for_zero_stress():
    atoms = _atoms_with_known_stress(0.0)
    assert pressure_GPa(atoms) == pytest.approx(0.0, abs=1e-12)


def test_eos_points_roundtrip(tmp_path):
    path = tmp_path / "eos.json"
    append_eos_point(path, {"temperature_K_target": 400.0, "pressure_GPa_measured": 0.19})
    append_eos_point(path, {"temperature_K_target": 400.0, "pressure_GPa_measured": 0.21})

    points = load_eos_points(path)
    assert len(points) == 2
    assert points[0]["pressure_GPa_measured"] == pytest.approx(0.19)
