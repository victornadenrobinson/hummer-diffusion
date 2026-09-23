#!/usr/bin/env python
"""Plot accumulated MACE-OFF throughput benchmarks (see benchmark_throughput.py)."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mace_ch4.benchmark import load_benchmark_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = load_benchmark_results(args.results)
    if not records:
        raise SystemExit(f"No benchmark records found in {args.results}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for r in records:
        key = (r.get("model", "?"), r.get("device", "?"), r.get("runner_label", "?"))
        groups[key].append(r)

    fig, (ax_throughput, ax_speed) = plt.subplots(1, 2, figsize=(11, 4.5))
    for key, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["n_molecules"])
        n_mol = [r["n_molecules"] for r in rows]
        label = " / ".join(key)
        ax_throughput.plot(n_mol, [r["ns_per_day"] for r in rows], "o-", label=label)
        ax_speed.plot(n_mol, [r["steps_per_second"] for r in rows], "o-", label=label)

    ax_throughput.set_xlabel("n methane molecules")
    ax_throughput.set_ylabel("ns/day")
    ax_throughput.set_title("Production throughput")
    ax_throughput.legend(fontsize="small")

    ax_speed.set_xlabel("n methane molecules")
    ax_speed.set_ylabel("MD steps/second")
    ax_speed.set_title("Raw step rate")
    ax_speed.legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
