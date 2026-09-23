import numpy as np
import pytest

from mace_ch4.finite_size import XI_CUBIC, implied_viscosity, yeh_hummer_extrapolate


def test_yeh_hummer_recovers_known_synthetic_line():
    d0_true = 2.3e-5  # A^2/fs, arbitrary units consistent with the fit
    slope_true = -1.2e-4
    box_lengths = np.array([20.0, 30.0, 45.0])
    d_pbc = d0_true + slope_true / box_lengths

    result = yeh_hummer_extrapolate(box_lengths, d_pbc)
    assert result["D0"] == pytest.approx(d0_true, rel=1e-8)
    assert result["slope"] == pytest.approx(slope_true, rel=1e-8)


def test_yeh_hummer_two_points_exact():
    box_lengths = np.array([20.0, 40.0])
    d_pbc = np.array([1.0e-5, 1.5e-5])
    result = yeh_hummer_extrapolate(box_lengths, d_pbc)

    # A line through exactly 2 points must reproduce them exactly.
    predicted = result["D0"] + result["slope"] / box_lengths
    np.testing.assert_allclose(predicted, d_pbc, rtol=1e-10)


def test_yeh_hummer_rejects_bad_input():
    with pytest.raises(ValueError):
        yeh_hummer_extrapolate([20.0], [1e-5])  # only 1 box size
    with pytest.raises(ValueError):
        yeh_hummer_extrapolate([-20.0, 30.0], [1e-5, 2e-5])  # negative length
    with pytest.raises(ValueError):
        yeh_hummer_extrapolate([20.0, 30.0], [-1e-5, 2e-5])  # negative D


def test_implied_viscosity_matches_formula_used_to_generate_slope():
    temperature_k = 300.0
    eta_true = 1.0e-4  # eV*fs/A^3, arbitrary but self-consistent
    from ase import units

    slope = -(units.kB * temperature_k * XI_CUBIC) / (6 * np.pi * eta_true)

    eta_recovered = implied_viscosity(slope, temperature_k)
    assert eta_recovered == pytest.approx(eta_true, rel=1e-8)


def test_implied_viscosity_rejects_positive_slope():
    with pytest.raises(ValueError):
        implied_viscosity(1.0, 300.0)
