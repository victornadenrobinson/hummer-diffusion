#!/usr/bin/env python
"""Compute a self-diffusion coefficient from a saved unwrapped-COM trajectory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mace_ch4.checkpoint import load_com_trajectory  # noqa: E402
from mace_ch4.msd import block_average_diffusion, ensemble_msd, fit_diffusion_coefficient  # noqa: E402

A2_FS_TO_M2_S = 1e-5  # 1 Angstrom^2/fs = 1e-20 m^2 / 1e-15 s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--com-traj", required=True, help="Output of run_production.py --com-output")
    p.add_argument("--fit-start", type=float, default=0.1)
    p.add_argument("--fit-end", type=float, default=0.9)
    p.add_argument("--n-blocks", type=int, default=5, help="Blocks for uncertainty estimate")
    p.add_argument("--out", required=True, help="JSON file to write results to")
    p.add_argument("--plot", default=None, help="Optional PNG path for a log-log MSD plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    times_fs, unwrapped_com = load_com_trajectory(args.com_traj)

    msd = ensemble_msd(unwrapped_com)
    fit = fit_diffusion_coefficient(times_fs, msd, fit_fraction=(args.fit_start, args.fit_end))

    if not fit["is_diffusive"]:
        print(
            f"WARNING: log-log MSD slope {fit['loglog_slope']:.2f} is not close to 1 "
            "in the fit window -- the trajectory may not be long enough, or the "
            "fit window includes a non-diffusive regime.",
            file=sys.stderr,
        )

    block_stats = block_average_diffusion(
        times_fs, unwrapped_com, n_blocks=args.n_blocks, fit_fraction=(args.fit_start, args.fit_end)
    )

    result = {
        "D_A2_per_fs": fit["D"],
        "D_m2_per_s": fit["D"] * A2_FS_TO_M2_S,
        "loglog_slope": fit["loglog_slope"],
        "is_diffusive": fit["is_diffusive"],
        "n_frames": int(len(times_fs)),
        "n_molecules": int(unwrapped_com.shape[1]),
        "block_D_mean_m2_per_s": block_stats["D_mean"] * A2_FS_TO_M2_S,
        "block_D_stderr_m2_per_s": block_stats["D_stderr"] * A2_FS_TO_M2_S,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.loglog(times_fs[1:], msd[1:], label="MSD(t)")
        i0, i1 = fit["fit_start_index"], fit["fit_end_index"]
        ax.loglog(times_fs[i0:i1], msd[i0:i1], lw=3, label="fit window")
        ax.set_xlabel("time (fs)")
        ax.set_ylabel(r"MSD ($\mathrm{\AA}^2$)")
        ax.legend()
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"Wrote plot to {args.plot}")


if __name__ == "__main__":
    main()
