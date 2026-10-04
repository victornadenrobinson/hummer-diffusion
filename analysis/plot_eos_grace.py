#!/usr/bin/env python
"""GRACE-OFF (and MACE-OFF24) equation of state at ~400 K vs reference EOSs.

    python analysis/plot_eos_grace.py        # -> analysis/eos_grace_400K.png/.pdf, analysis/eos_grace_400K_summary.csv

(a) H2O: P vs rho, GRACE-OFF seed means (T-corrected to 400 K with the reference
    (dP/dT)_rho; bars = seed standard deviation) vs IAPWS-95 at 400 K.
(b) Density error at equal pressure, rho_model / rho_ref(400 K, P_model) - 1, vs P:
    GRACE-OFF H2O and CH4, and the MACE-OFF24 400 K CH4 runs for comparison.
(c) CH4/H2O mixture: GRACE-OFF vs the P_EOS supplied with the data (source and
    composition to be confirmed -- shown as given, not judged).
Reference EOSs (IAPWS-95, Setzmann-Wagner via CoolProp) are validated to 1 GPa;
beyond that the reference curve is dashed (extrapolation).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import CoolProp.CoolProp as CP  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
FLUID = {"H2O": "Water", "CH4": "Methane"}
COL = {"H2O": "#2a78d6", "CH4": "#eb6834", "mix": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
T0 = 400.0
P_VALID = 1.0  # GPa, upper validation limit of both reference EOSs


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, which="both", labelcolor=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def p_ref(fluid, T, rho):
    return CP.PropsSI("P", "T", T, "Dmass", rho * 1e3, FLUID[fluid]) / 1e9


def rho_ref(fluid, T, P):
    try:
        return CP.PropsSI("Dmass", "T", T, "P", P * 1e9, FLUID[fluid]) / 1e3
    except ValueError:
        return np.nan


def density_err_bar_pct(fluid, P, P_err):
    """Density error bar (%) from a pressure error: 100 * beta_T * dP, beta_T from the reference EOS."""
    if not (np.isfinite(P_err) and P_err > 0):
        return np.nan
    h = max(1e-4, 0.01 * P)
    r1, r2 = rho_ref(fluid, T0, P - h), rho_ref(fluid, T0, P + h)
    beta = (r2 - r1) / (2 * h) / rho_ref(fluid, T0, P)
    return 100 * beta * P_err


def t_correct(df):
    """P at 400 K: P - (dP/dT)_rho (T - 400), derivative from the reference EOS."""
    out = []
    for _, r in df.iterrows():
        if r.kind in FLUID:
            dpdt = (p_ref(r.kind, r.T_K + 1, r.rho_g_cm3) - p_ref(r.kind, r.T_K - 1, r.rho_g_cm3)) / 2
            out.append(r.P_GPa - dpdt * (r.T_K - T0))
        else:
            out.append(r.P_GPa)  # no mixture reference here: left as measured
    return df.assign(P400=out)


def seed_means(df):
    g = df.groupby(["kind", "rho_g_cm3"])
    return g.agg(P=("P400", "mean"), P_sd=("P400", lambda x: x.std(ddof=1) if len(x) > 1 else np.nan),
                 n=("P400", "size"), T=("T_K", "mean"), P_EOS=("P_EOS_GPa", "first")).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grace", default=HERE / "data" / "grace_eos_400K.txt")
    ap.add_argument("--mace", default=HERE / "data" / "mace_off24_D.csv")
    ap.add_argument("--out", default=HERE / "eos_grace_400K")
    args = ap.parse_args()

    cols = ["kind", "run", "N", "rho_g_cm3", "nve_ps", "T_K", "P_GPa", "P_err_GPa", "P_EOS_GPa"]
    df = t_correct(pd.read_csv(args.grace, sep=r"\s+", comment="#", names=cols))
    m = seed_means(df[df.N == 128])
    m["rho_ref_at_P"] = [rho_ref(k, T0, P) if k in FLUID else np.nan for k, P in zip(m.kind, m.P)]
    m["density_err_pct"] = 100 * (m.rho_g_cm3 / m.rho_ref_at_P - 1)
    m["density_err_bar_pct"] = [density_err_bar_pct(k, P, e) if k in FLUID and np.isfinite(r) else np.nan
                                for k, P, e, r in zip(m.kind, m.P, m.P_sd, m.rho_ref_at_P)]

    mace = pd.read_csv(args.mace)
    mace = mace[(mace.T_target_K == 400) & ~mace.statepoint.str.contains("_rho")].copy()
    mace["P400"] = [P - (p_ref("CH4", T + 1, r) - p_ref("CH4", T - 1, r)) / 2 * (T - T0)
                    for P, T, r in zip(mace.P_GPa, mace.T_K, mace.rho_g_cm3)]
    mace["density_err_pct"] = [100 * (r / rho_ref("CH4", T0, P) - 1) for r, P in zip(mace.rho_g_cm3, mace.P400)]
    mace["density_err_bar_pct"] = [density_err_bar_pct("CH4", P, e) for P, e in zip(mace.P400, mace.P_err_GPa)]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5.2))
    for a in (a1, a2, a3):
        style(a)

    # (a) water P(rho)
    w = m[m.kind == "H2O"]
    rr = np.linspace(0.96, 1.34, 200)
    pr = np.array([p_ref("H2O", T0, x) for x in rr])
    ok = pr <= P_VALID
    a1.plot(rr[ok], pr[ok], "-", color=INK, lw=1.4, label="IAPWS-95, 400 K")
    a1.plot(rr[~ok | np.roll(ok, -1)], pr[~ok | np.roll(ok, -1)], "--", color=INK, lw=1.2,
            label="IAPWS-95 extrapolated (> 1 GPa)")
    a1.errorbar(w.rho_g_cm3, w.P, yerr=w.P_sd, fmt="o", ms=6, color=COL["H2O"], mec="white", mew=0.8,
                capsize=0, label="GRACE-OFF 2L-M (mean of 3 seeds)")
    a1.set_yscale("log")
    a1.set_xlabel(r"density (g/cm$^3$)", color=INK)
    a1.set_ylabel("P (GPa)", color=INK)
    a1.set_title(r"(a) H$_2$O at 400 K", loc="left", fontsize=11, color=INK)
    a1.legend(fontsize=8.5, frameon=False, loc="upper left")
    hi = w[w.rho_g_cm3 > 1.32]
    if len(hi):
        r0 = hi.iloc[0]
        a1.annotate("7.2 GPa: check for freezing\n(ice VII) before using",
                    (r0.rho_g_cm3, r0.P), xytext=(1.335, 0.12), textcoords="data", ha="right",
                    fontsize=8, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    # (b) density error at equal P
    a2.axhline(0, color=MUTED, lw=1)
    a2.axvspan(P_VALID, 3, color=GRID, alpha=0.5, lw=0)
    a2.text(1.04, 0.98, "reference EOS\nextrapolated", transform=a2.get_xaxis_transform(), fontsize=8, color=MUTED,
            va="top")
    for kind in ("H2O", "CH4"):
        s = m[(m.kind == kind) & m.density_err_pct.notna()]
        a2.errorbar(s.P, s.density_err_pct, yerr=s.density_err_bar_pct, fmt="o-" if len(s) > 1 else "o",
                    color=COL[kind], ms=6, lw=2, mec="white", mew=0.8, capsize=0, elinewidth=1)
    ms = mace.sort_values("P400")
    a2.errorbar(ms.P400, ms.density_err_pct, yerr=ms.density_err_bar_pct, fmt="s--", color=COL["CH4"],
                mfc="white", mew=1.4, ms=5.5, lw=1.2, capsize=0, elinewidth=1)
    a2.set_xscale("log")
    a2.set_xlim(0.007, 3)
    a2.set_xlabel("P (GPa)", color=INK)
    a2.set_ylabel(r"$\rho_\mathrm{model}/\rho_\mathrm{ref}(400\,\mathrm{K}, P) - 1$  (%)", color=INK)
    a2.set_title("(b) density error at equal pressure, 400 K", loc="left", fontsize=11, color=INK)
    a2.legend(handles=[
        Line2D([], [], color=COL["H2O"], marker="o", lw=2, label=r"H$_2$O, GRACE-OFF"),
        Line2D([], [], color=COL["CH4"], marker="o", lw=0, label=r"CH$_4$, GRACE-OFF (1 density, 3 seeds)"),
        Line2D([], [], color=COL["CH4"], marker="s", mfc="white", ls="--", label=r"CH$_4$, MACE-OFF24"),
    ], fontsize=8.5, frameon=False, loc="lower left")

    # (c) mixture vs supplied reference
    x = m[m.kind == "mix"]
    a3.errorbar(x.rho_g_cm3, x.P, yerr=x.P_sd, fmt="o", ms=6, color=COL["mix"], mec="white", mew=0.8,
                capsize=0, label="GRACE-OFF, N = 128 (seed mean)")
    a3.plot(x.rho_g_cm3, x.P_EOS, "x", color=INK, ms=7, mew=1.5, label=r"$P_\mathrm{EOS}$ as supplied")
    big = df[(df.kind == "mix") & (df.N > 128)]
    a3.scatter(big.rho_g_cm3 + 0.004, big.P400, s=22, color=COL["mix"], marker="D", edgecolors=INK, lw=0.5,
               label="N = 256-1024 (finite-size check)", zorder=4)
    a3.set_xlabel(r"density (g/cm$^3$)", color=INK)
    a3.set_ylabel("P (GPa)", color=INK)
    a3.set_title(r"(c) CH$_4$/H$_2$O mixture, ~400 K", loc="left", fontsize=11, color=INK)
    a3.legend(fontsize=8.5, frameon=False, loc="upper left")
    a3.text(0.98, 0.03, "reference source & composition: TBC;\nCH$_4$/H$_2$O may not be fully miscible here",
            transform=a3.transAxes, ha="right", fontsize=8, color=MUTED)

    fig.text(0.01, 0.005, "Pressures from NVE, T-corrected to 400 K with the reference (dP/dT)$_\\rho$ (pure fluids "
             "only); bars = seed std (GRACE) or block stderr (MACE), in (b) times the reference compressibility. "
             "Reference EOSs via CoolProp, validated to 1 GPa.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    m.to_csv(f"{args.out}_summary.csv", index=False, float_format="%.5g")
    print(m.round(4).to_string(index=False))
    print(f"wrote {args.out}.png/.pdf and {args.out}_summary.csv")


if __name__ == "__main__":
    main()
