#!/usr/bin/env python
"""MACE-OFF24 CH4 equation of state from the state-point dataset, vs the NIST reference.

    python analysis/plot_eos.py            # -> analysis/eos_PV.png/.pdf

Left: molar volume vs P (log-log). Right: density vs P (log P).
One point per run: P is the measured NVE mean (horizontal bars = its standard
error), rho / V the run's fixed volume. Filled = NPT-equilibrated runs,
hollow = the 10 ps fixed-density 200 K series. Dashed = reference EOS at the
same temperatures (analysis/data/nist_ch4_isotherms.csv).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
M_CH4 = 16.043  # g/mol
T_COLOURS = {200: "#2a78d6", 300: "#eb6834", 400: "#1baf7a", 450: "#eda100"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, which="both", labelcolor=INK)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mace", default=HERE / "data" / "mace_off24_D.csv")
    p.add_argument("--ref", default=HERE / "data" / "nist_ch4_isotherms.csv")
    p.add_argument("--out", default=HERE / "eos_PV")
    args = p.parse_args()

    mace = pd.read_csv(args.mace)
    mace = mace.assign(T=mace.T_target_K.round().astype(int), Vm=M_CH4 / mace.rho_g_cm3,
                       fixed_rho=mace.statepoint.str.contains("_rho"))
    ref = pd.read_csv(args.ref, comment="#").assign(Vm=lambda d: M_CH4 / d.rho_g_cm3)

    fig, (axv, axr) = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, col in ((axv, "Vm"), (axr, "rho_g_cm3")):
        style(ax)
        for T, c in T_COLOURS.items():
            r = ref[ref.T_K == T]
            ax.plot(r.P_GPa, r[col], "--", color=c, lw=1.5, alpha=0.9, zorder=1)
            m = mace[mace["T"] == T].sort_values("P_GPa")
            for fixed, grp in m.groupby("fixed_rho"):
                ax.errorbar(grp.P_GPa, grp[col], xerr=grp.P_err_GPa, fmt="o", ms=5.5,
                            mfc="white" if fixed else c, mec=c, mew=1.4, ecolor=c,
                            elinewidth=1, capsize=0, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("P (GPa)", color=INK)

    axv.set_yscale("log")
    axv.set_ylabel(r"molar volume $V_m$ (cm$^3$/mol)", color=INK)
    axv.set_title(r"CH$_4$ P–V: MACE-OFF24 vs reference EOS", loc="left", fontsize=11, color=INK)
    axr.set_ylabel(r"density $\rho$ (g/cm$^3$)", color=INK)
    axr.set_title(r"CH$_4$ density vs pressure", loc="left", fontsize=11, color=INK)
    axr.axvline(0.004599, color=MUTED, lw=0.8, ls=":")
    axr.text(0.0047, 0.585, r"expt $P_c$", color=MUTED, fontsize=8, va="top")

    handles = [Line2D([], [], color=c, marker="o", ms=5.5, lw=0, label=f"{T} K") for T, c in T_COLOURS.items()]
    handles += [Line2D([], [], color=MUTED, marker="o", ms=5.5, lw=0, label="MACE-OFF24, NPT runs"),
                Line2D([], [], color=MUTED, marker="o", ms=5.5, mfc="white", lw=0, label="MACE-OFF24, fixed ρ (10 ps)"),
                Line2D([], [], color=MUTED, ls="--", lw=1.5, label="reference EOS (NIST)")]
    axr.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False)
    fig.text(0.01, 0.005, "P = measured NVE mean ± stderr; reference: Setzmann–Wagner EOS via CoolProp.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print(f"wrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
