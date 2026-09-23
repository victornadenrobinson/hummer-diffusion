import numpy as np

from mace_ch4.msd import (
    block_average_diffusion,
    ensemble_msd,
    fit_diffusion_coefficient,
    msd_fft,
)


def _brute_force_msd(positions: np.ndarray) -> np.ndarray:
    """O(n^2) reference MSD for a single particle's trajectory."""
    n = positions.shape[0]
    msd = np.zeros(n)
    for lag in range(n):
        diffs = positions[: n - lag] - positions[lag:]
        msd[lag] = np.square(diffs).sum(axis=1).mean()
    return msd


def test_msd_fft_matches_brute_force():
    rng = np.random.default_rng(0)
    positions = np.cumsum(rng.normal(size=(200, 3)), axis=0)

    fast = msd_fft(positions)
    slow = _brute_force_msd(positions)
    np.testing.assert_allclose(fast, slow, atol=1e-8)


def test_ensemble_msd_of_identical_particles_matches_single_particle():
    rng = np.random.default_rng(1)
    single = np.cumsum(rng.normal(size=(100, 3)), axis=0)
    stacked = np.stack([single, single], axis=1)  # (n_frames, 2 particles, 3)

    result = ensemble_msd(stacked)
    expected = msd_fft(single)
    np.testing.assert_allclose(result, expected, atol=1e-8)


def test_fit_diffusion_coefficient_recovers_known_D():
    d_true = 0.5
    dt = 1.0
    n_frames, n_particles = 4000, 200
    sigma = np.sqrt(2 * d_true * dt)

    rng = np.random.default_rng(12345)
    steps = rng.normal(scale=sigma, size=(n_frames, n_particles, 3))
    positions = np.cumsum(steps, axis=0)

    times = np.arange(n_frames) * dt
    msd = ensemble_msd(positions)
    result = fit_diffusion_coefficient(times, msd, fit_fraction=(0.05, 0.5))

    assert result["is_diffusive"]
    assert abs(result["D"] - d_true) / d_true < 0.15


def test_fit_diffusion_coefficient_flags_non_diffusive_ballistic_regime():
    # Pure ballistic motion (MSD ~ t^2) should NOT look diffusive.
    times = np.linspace(1.0, 100.0, 500)
    msd = times**2
    result = fit_diffusion_coefficient(times, msd, fit_fraction=(0.1, 0.9))
    assert not result["is_diffusive"]


def test_block_average_diffusion_recovers_known_D_with_finite_spread():
    d_true = 0.3
    dt = 1.0
    n_frames, n_particles = 6000, 100
    sigma = np.sqrt(2 * d_true * dt)

    rng = np.random.default_rng(999)
    steps = rng.normal(scale=sigma, size=(n_frames, n_particles, 3))
    positions = np.cumsum(steps, axis=0)
    times = np.arange(n_frames) * dt

    result = block_average_diffusion(times, positions, n_blocks=5, fit_fraction=(0.05, 0.5))

    assert len(result["D_values"]) == 5
    assert abs(result["D_mean"] - d_true) / d_true < 0.25
    assert result["D_std"] >= 0.0
    assert result["D_stderr"] <= result["D_std"]


def test_block_average_diffusion_rejects_too_few_frames_per_block():
    times = np.arange(10.0)
    positions = np.zeros((10, 2, 3))
    try:
        block_average_diffusion(times, positions, n_blocks=5)
        assert False, "expected ValueError for too-short blocks"
    except ValueError:
        pass

