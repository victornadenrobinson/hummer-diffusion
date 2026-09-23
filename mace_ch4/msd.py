"""FFT-based mean-squared displacement and diffusion-coefficient fitting."""
from __future__ import annotations

import numpy as np


def _autocorr_fft(x: np.ndarray) -> np.ndarray:
    """1D autocorrelation of x via FFT, normalized by the overlap count."""
    n = len(x)
    f = np.fft.fft(x, n=2 * n)
    power = f * f.conjugate()
    corr = np.fft.ifft(power)[:n].real
    return corr / (n - np.arange(n))


def msd_fft(positions: np.ndarray) -> np.ndarray:
    """MSD(t) for a single particle's trajectory, positions shape (n_frames, 3).

    O(n log n) algorithm (see e.g. Calandrini et al., "nMoldyn", 2011),
    equivalent to but much faster than the brute-force O(n^2) definition.
    """
    n = positions.shape[0]
    sq = np.square(positions).sum(axis=1)
    sq = np.append(sq, 0.0)
    s2 = sum(_autocorr_fft(positions[:, dim]) for dim in range(positions.shape[1]))

    s1 = np.zeros(n)
    q = 2.0 * sq.sum()
    for m in range(n):
        q -= sq[m - 1] + sq[n - m]
        s1[m] = q / (n - m)
    return s1 - 2.0 * s2


def ensemble_msd(positions: np.ndarray) -> np.ndarray:
    """Average MSD(t) over particles, positions shape (n_frames, n_particles, 3)."""
    n_frames, n_particles, _ = positions.shape
    total = np.zeros(n_frames)
    for p in range(n_particles):
        total += msd_fft(positions[:, p, :])
    return total / n_particles


def fit_diffusion_coefficient(
    times: np.ndarray,
    msd: np.ndarray,
    fit_fraction: tuple[float, float] = (0.1, 0.9),
    n_dims: int = 3,
) -> dict:
    """Fit D from the Einstein relation MSD(t) = 2*n_dims*D*t + const.

    Returns a dict with D, intercept, the fit window indices, and a
    diffusive-regime sanity check (log-log slope, should be close to 1).
    """
    n = len(times)
    i0 = int(fit_fraction[0] * n)
    i1 = int(fit_fraction[1] * n)
    if i1 - i0 < 2:
        raise ValueError("Fit window too narrow; need at least 2 points")

    slope, intercept = np.polyfit(times[i0:i1], msd[i0:i1], 1)
    d_coeff = slope / (2.0 * n_dims)

    # Diffusive-regime sanity check: slope of log(MSD) vs log(t) should be ~1.
    valid = (times[i0:i1] > 0) & (msd[i0:i1] > 0)
    if valid.sum() >= 2:
        loglog_slope = np.polyfit(
            np.log(times[i0:i1][valid]), np.log(msd[i0:i1][valid]), 1
        )[0]
    else:
        loglog_slope = float("nan")

    return {
        "D": d_coeff,
        "intercept": intercept,
        "fit_start_index": i0,
        "fit_end_index": i1,
        "loglog_slope": loglog_slope,
        "is_diffusive": bool(0.8 <= loglog_slope <= 1.2),
    }


def block_average_diffusion(
    times: np.ndarray,
    positions: np.ndarray,
    n_blocks: int = 5,
    fit_fraction: tuple[float, float] = (0.1, 0.9),
    n_dims: int = 3,
) -> dict:
    """Split a trajectory into n_blocks contiguous segments and fit D on each.

    A simple (not rigorous) uncertainty estimate: treats each block as an
    independent replicate. Only meaningful if the block length is well above
    the diffusive correlation time; the caller is responsible for checking that.
    """
    n_frames = positions.shape[0]
    block_len = n_frames // n_blocks
    if block_len < 3:
        raise ValueError(
            f"Only {n_frames} frames for {n_blocks} blocks; too short to fit per block"
        )

    d_values = []
    for b in range(n_blocks):
        sl = slice(b * block_len, (b + 1) * block_len)
        block_positions = positions[sl] - positions[sl][0]
        block_times = times[sl] - times[sl][0]
        block_msd = ensemble_msd(block_positions)
        fit = fit_diffusion_coefficient(block_times, block_msd, fit_fraction, n_dims)
        d_values.append(fit["D"])

    d_values = np.array(d_values)
    return {
        "D_values": d_values,
        "D_mean": float(d_values.mean()),
        "D_std": float(d_values.std(ddof=1)) if len(d_values) > 1 else 0.0,
        "D_stderr": float(d_values.std(ddof=1) / np.sqrt(len(d_values))) if len(d_values) > 1 else 0.0,
    }
