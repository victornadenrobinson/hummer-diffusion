#!/usr/bin/env python
"""Yeh-Hummer finite-size extrapolation from multiple per-box-size D_PBC results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mace_ch4.finite_size import implied_viscosity, yeh_hummer_extrapolate  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--result",
        action="append",
        nargs=2,
        metavar=("BOX_LENGTH_A", "MSD_JSON"),
        required=True,
        help="Repeatable: a box length (Angstrom) and the compute_msd_diffusion.py JSON output for that box size",
    )
    p.add_argument("--temperature-K", type=float, required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.result) < 2:
        raise SystemExit("Need at least 2 --result entries (box sizes) to extrapolate")

    box_lengths = []
    d_values_m2_s = []
    for box_length_str, msd_json_path in args.result:
        with open(msd_json_path) as f:
            data = json.load(f)
        box_lengths.append(float(box_length_str))
        d_values_m2_s.append(data["D_m2_per_s"])

    fit = yeh_hummer_extrapolate(box_lengths, d_values_m2_s)
    try:
        eta = implied_viscosity(fit["slope"], args.temperature_K)
    except ValueError:
        eta = None

    result = {
        "box_lengths_A": box_lengths,
        "D_PBC_m2_per_s": d_values_m2_s,
        "D0_m2_per_s": fit["D0"],
        "slope": fit["slope"],
        "implied_viscosity_eV_fs_per_A3": eta,
        "temperature_K": args.temperature_K,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
