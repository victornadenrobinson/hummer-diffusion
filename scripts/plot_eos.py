#!/usr/bin/env python
"""Plot accumulated EOS points (pressure vs density, isotherms) from equilibrate_nvt.py."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mace_ch4.eos import load_eos_points  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eos-log", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    points = load_eos_points(args.eos_log)
    if not points:
        raise SystemExit(f"No EOS points found in {args.eos_log}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_temperature = defaultdict(list)
    for pt in points:
        by_temperature[pt["temperature_K_target"]].append(pt)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for temperature, rows in sorted(by_temperature.items()):
        rows = sorted(rows, key=lambda r: r["density_kg_m3"])
        density = [r["density_kg_m3"] for r in rows]
        pressure = [r["pressure_GPa_measured"] for r in rows]
        error = [r["pressure_GPa_stderr"] for r in rows]
        ax.errorbar(density, pressure, yerr=error, fmt="o-", capsize=3, label=f"{temperature:.0f} K")

    ax.set_xlabel(r"density (kg/m$^3$)")
    ax.set_ylabel("pressure (GPa)")
    ax.set_title("Methane equation of state (measured)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
