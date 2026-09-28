#!/usr/bin/env python
"""Yeh-Hummer-corrected MACE-OFF24 self-diffusion of CH4 vs experiment.

    python analysis/plot_D_vs_experiment.py            # -> analysis/D_vs_experiment.png/.pdf

Left: D vs P (log-log), colour = temperature, marker = experimental source;
MACE-OFF24 as lines + circles. Right: D_MACE / D_expt at each experimental
point, with MACE interpolated in log D vs log P along its own isotherm.

MACE value plotted (--mace-column, default D_MSD_eta_GK): D_inf from the
MSD with the YH correction using eta from Green-Kubo. Points are drawn hollow
when the run is not converged for D: MSD log-log slope outside 0.9-1.1 or
Green-Kubo and MSD D differing by more than 5%. The 10 ps 200 K density
series (statepoint_*_rho*) is left out by default (--include-rho-series).
P is the measured NVE mean, not the NPT target.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent

# Temperature -> colour (fixed categorical order, validated for CVD; the
# yellow/aqua contrast warning is covered by direct labels and the CSV table).
T_COLOURS = {200: "#2a78d6", 300: "#eb6834", 400: "#1baf7a", 450: "#eda100"}
EXPT_T_GROUP = {200: 200, 300: 300, 405: 400, 455: 450}  # Greiner 405/455 K sit with MACE 400/450 K
SOURCES = {  # source -> (marker, size, label)
    "Ranieri": ("D", 42, "Ranieri"),
    "Ranieri NatCom": ("o", 42, "Ranieri (NatCom)"),
    "Harris": ("*", 90, "Harris"),
    "Greiner": ("s", 38, "Greiner"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mace", default=HERE / "data" / "mace_off24_D.csv")
    p.add_argument("--expt", default=HERE / "data" / "expt_D_digitized.csv")
    p.add_argument("--mace-column", default="D_MSD_eta_GK",
                   choices=["D_MSD_eta_GK", "D_GK_eta_GK", "D_MSD_eta_EH", "D_GK_eta_EH", "D_MSD", "D_GK"])
    p.add_argument("--include-rho-series", action="store_true")
    p.add_argument("--out", default=HERE / "D_vs_experiment")
    return p.parse_args()


def load(args):
    mace = pd.read_csv(args.mace)
    if not args.include_rho_series:
        mace = mace[~mace.statepoint.str.contains("_rho")]
    mace = mace.assign(
        T=mace.T_target_K.round().astype(int),
        D=mace[args.mace_column],
        D_err=mace[args.mace_column + "_err"] if args.mace_column + "_err" in mace else 0.0,
        converged=mace.msd_loglog_slope.between(0.9, 1.1) & ((mace.GK_over_MSD - 1).abs() <= 0.05),
    ).sort_values(["T", "P_GPa"])
    expt = pd.read_csv(args.expt, comment="#")
    expt = expt.assign(group=expt.T_K.map(EXPT_T_GROUP), D=expt["D_1e-5_cm2_s"])
    return mace, expt


def mace_at(mace, T, P):
    """log-log interpolation of converged MACE D along the T isotherm; NaN outside its range."""
    iso = mace[(mace["T"] == T) & mace.converged].sort_values("P_GPa")
    if len(iso) < 2 or not iso.P_GPa.min() <= P <= iso.P_GPa.max():
        return np.nan
    return 10 ** np.interp(np.log10(P), np.log10(iso.P_GPa), np.log10(iso.D))


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, which="both", labelcolor=INK)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    args = parse_args()
    mace, expt = load(args)

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})
    for a in (ax, axr):
        style(a)

    # --- MACE isotherms
    label_offset = {200: (6, 0), 300: (6, 0), 400: (6, -7), 450: (6, 7)}
    for T, iso in mace.groupby("T"):
        c = T_COLOURS[T]
        # Break the line across a density gap of > 0.15 g/cm3 between neighbouring
        # runs (200 K: 0.037 -> 0.358 g/cm3 across the near-critical region, no runs).
        segment = (iso.rho_g_cm3.diff() > 0.15).cumsum()
        for _, seg in iso.groupby(segment):
            ax.plot(seg.P_GPa, seg.D, "-", color=c, lw=2, zorder=2)
        ok, bad = iso[iso.converged], iso[~iso.converged]
        ax.errorbar(ok.P_GPa, ok.D, yerr=ok.D_err, fmt="o", ms=5.5, color=c, mec=c,
                    ecolor=c, elinewidth=1, capsize=0, zorder=3)
        ax.scatter(bad.P_GPa, bad.D, s=34, facecolors="white", edgecolors=c, linewidths=1.5, zorder=3)
        last = iso.iloc[-1]
        ax.annotate(f"{T} K", (last.P_GPa, last.D), xytext=label_offset[T], textcoords="offset points",
                    va="center", fontsize=10, color=INK, fontweight="bold")

    # --- experiment
    for (src, T), grp in expt.groupby(["source", "T_K"]):
        marker, size, _ = SOURCES[src]
        ax.scatter(grp.P_GPa, grp.D, marker=marker, s=size, color=T_COLOURS[EXPT_T_GROUP[T]],
                   edgecolors=INK, linewidths=0.6, zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("P (GPa)", color=INK)
    ax.set_ylabel(r"D  ($10^{-5}$ cm$^2$/s)", color=INK)
    ax.set_title(r"CH$_4$ self-diffusion: MACE-OFF24 (Yeh–Hummer corrected) vs experiment",
                 loc="left", fontsize=11, color=INK)

    # legend: colour = T (MACE line), marker = source (grey), open = unconverged
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=T_COLOURS[T], lw=2, marker="o", ms=5.5,
                      label=f"MACE-OFF24 {T} K" + (" (expt 405 K)" if T == 400 else " (expt 455 K)" if T == 450 else ""))
               for T in T_COLOURS]
    handles += [Line2D([], [], ls="none", marker="o", ms=6, mfc="white", mec=MUTED, mew=1.5,
                       label="MACE, D not converged")]
    handles += [Line2D([], [], ls="none", marker=m, ms=np.sqrt(s) * 0.95, color="#b8b7b0", mec=INK, mew=0.6, label=lab)
                for m, s, lab in SOURCES.values()]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, frameon=False, ncol=2)

    # --- ratio panel
    expt = expt.assign(D_mace=[mace_at(mace, g, p) for g, p in zip(expt.group, expt.P_GPa)])
    expt["ratio"] = expt.D_mace / expt.D
    axr.axhline(1, color=MUTED, lw=1, zorder=1)
    for (src, T), grp in expt.dropna(subset=["ratio"]).groupby(["source", "T_K"]):
        marker, size, _ = SOURCES[src]
        axr.scatter(grp.P_GPa, grp.ratio, marker=marker, s=size, color=T_COLOURS[EXPT_T_GROUP[T]],
                    edgecolors=INK, linewidths=0.6, zorder=3)
    axr.set_xscale("log")
    axr.set_xlabel("P (GPa)", color=INK)
    axr.set_ylabel(r"$D_\mathrm{MACE} / D_\mathrm{expt}$", color=INK)
    axr.set_title("Ratio at each experimental point", loc="left", fontsize=11, color=INK)
    n_out = expt.ratio.isna().sum()
    axr.text(0.02, 0.02, f"MACE interpolated (log–log) along its isotherm, converged points only;\n"
             f"{n_out} expt points outside the MACE pressure range are omitted",
             transform=axr.transAxes, fontsize=8, color=MUTED, va="bottom")

    fig.text(0.01, 0.005, f"MACE: {args.mace_column}; P = measured NVE mean.   "
             "Experiment digitized from the Veusz figure (approximate, ~1%).",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    expt.to_csv(f"{args.out}_ratios.csv", index=False, float_format="%.4g")
    print(f"wrote {args.out}.png/.pdf and {args.out}_ratios.csv")
    summary = expt.dropna(subset=["ratio"]).groupby(["source", "T_K"]).ratio.agg(["count", "mean", "min", "max"])
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
