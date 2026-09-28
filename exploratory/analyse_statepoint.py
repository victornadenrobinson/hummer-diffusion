#!/usr/bin/env python3
"""
analyse_statepoint.py -- transport analysis of run_statepoint.py output.

    python -u analyse_statepoint.py statepoint_mace_off24_450K_0.3GPa          # one state point
    python -u analyse_statepoint.py DIR1 DIR2 ...                              # several
    python -u analyse_statepoint.py N032 N064 N128 N256 N512 --series nseries  # + Yeh-Hummer N-series fit
    python -u analyse_statepoint.py DIR --show     # also open the figure (needs a display)
    ~/Scratch/envs/conda-viewer/bin/python analyse_statepoint.py DIR --replot --show
                                                   # redraw/view only (no scipy needed)

Per state point (reads thermo_nve.log + nve_com.npz / nve_stress.npz, or, while a run
is still going, the latest NVE checkpoint restart.npz: --source auto|final|checkpoint):
  1. T, P       NVE means with block-averaged standard errors; energy drift
  1b. equipartition  T_trans (from COM velocities, always available) vs T_total per block;
                T_rot / T_vib too if the run recorded nve_modes (run_statepoint.py v3).
                T_trans uses 3N-3 dof (total momentum is zero), as run_statepoint.py does.
  2. D_MSD      multi-origin COM MSD (FFT); linear fit in --msd-fit window (fractions of
                the run); log-log slope; D(t) = local slope / 6; 5-block error
  3. D_GK       COM velocity ACF (FFT); running integral D(t); plateau mean over
                --gk-window ps; 5-block error; D_GK / D_MSD
  4. eta        shear viscosity from the full pressure tensor (virial + kinetic): from the
                MOLECULAR tensor if the run recorded it (run_state_molpressure.py; faster
                convergence), with the atomic tensor as a cross-check; otherwise atomic. Averaged
                over the 5 independent shear components of an isotropic fluid:
                P_yz, P_xz, P_xy, (P_xx-P_yy)/2, (P_yy-P_zz)/2
                a) Green-Kubo  eta = V/kT int <P_ab(0)P_ab(t)> dt: running integral per block;
                   time-decomposition method (Zhang, Otani & Maginn, JCTC 2015): cutoff where
                   the block std exceeds 40 % of the mean, double-exponential fit weighted by
                   the fitted std ~ t^b. Both the cutoff search and the fit start at --tdm-tmin
                   (default 0.2 ps): before that the ATOMIC running integral still oscillates with
                   the C-H vibrations and passes close to zero, which made the 40 % test cut at
                   ~20 fs. Also the plain plateau mean over --eta-window.
                b) Einstein-Helfand  eta = V/2kT d/dt <[int_0^t P_ab dt']^2>, slope fitted over
                   --eh-window ps; block error.
  5. Yeh-Hummer D_inf = D_PBC + xi kT / (6 pi eta L), xi = 2.837297, for D_MSD and D_GK with
                eta_GK (primary) and eta_EH; --eta-mPas overrides eta (e.g. runs without a
                pressure-tensor record). If CoolProp is importable, its (experiment-based)
                viscosity at the run's (T, rho) is printed for comparison.
Writes into each DIR: analysis.json, analysis_curves.npz, analysis.png.

--series OUT: after the individual analyses, fits D_PBC vs 1/L across the DIRs
(same T and density, different N): intercept = D_inf, slope = -xi kT/(6 pi eta), so
eta_fit is an independent viscosity. Writes OUT.json and OUT.png.

Units: D in m^2/s, cm^2/s and 1e-5 cm^2/s; eta in mPa s; T in K; P in GPa.
"""

import argparse
import json
import os
import platform
import sys
import time
import warnings

import numpy as np

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:9.1f}s] {msg}", flush=True)


KB = 1.380649e-23          # J/K
XI_CUBIC = 2.837297        # Yeh-Hummer constant, cubic box
A2_FS_TO_M2_S = 1e-5
M2_S_TO_CM2_S = 1e4
GPA2_FS_TO_PA2_S = 1e3     # GPa^2 * fs -> Pa^2 * s  (1e18 * 1e-15)
AMU_A3_TO_G_CM3 = 1.66053906660
APM = 5
M_CH4_AMU = 16.043


# ---------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="run_statepoint.py output directories")
    p.add_argument("--source", choices=["auto", "final", "checkpoint"], default="auto",
                   help="final npz files, the NVE checkpoint, or whichever has more data (default)")
    p.add_argument("--msd-fit", type=float, nargs=2, default=(0.1, 0.5), metavar=("LO", "HI"),
                   help="MSD fit lag window, fractions of the BLOCK length = NVE length / --blocks "
                        "(default 0.1 0.5); the full run and the blocks use the same window")
    p.add_argument("--gk-window", type=float, nargs=2, default=(2.0, 10.0), metavar=("LO", "HI"),
                   help="lag window (ps) for the D_GK plateau (default 2 10)")
    p.add_argument("--eta-window", type=float, nargs=2, default=(0.5, 2.0), metavar=("LO", "HI"),
                   help="lag window (ps) for the plain GK viscosity plateau (default 0.5 2)")
    p.add_argument("--eta-maxlag", type=float, default=5.0, help="max lag (ps) for the stress ACF (default 5)")
    p.add_argument("--eta-blocks", type=int, default=10, help="blocks for the time-decomposition method")
    p.add_argument("--tdm-tmin", type=float, default=0.2,
                   help="TDM: earliest lag (ps) for the 40%% cutoff search and the fit, past the fast "
                        "intramolecular oscillations of the atomic stress (default 0.2)")
    p.add_argument("--eh-window", type=float, nargs=2, default=(1.0, 4.0), metavar=("LO", "HI"),
                   help="Einstein-Helfand slope window, ps (default 1 4)")
    p.add_argument("--blocks", type=int, default=5, help="blocks for D errors (default 5)")
    p.add_argument("--eta-mPas", type=float, default=None,
                   help="use this viscosity (mPa s) for Yeh-Hummer instead of the computed one")
    p.add_argument("--series", default=None, metavar="OUT", help="N-series Yeh-Hummer fit across DIRs")
    p.add_argument("--show", action="store_true", help="show figures interactively")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--replot", action="store_true",
                   help="no analysis: redraw analysis.png from analysis.json + analysis_curves.npz "
                        "(use with --show from a python that has a display, e.g. conda-viewer)")
    return p.parse_args()


# ---------------------------------------------------------------- loading
def read_thermo(path):
    header, names = {}, None
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if body.startswith("step "):
                names = body.split()
            elif ":" in body and not body.startswith("stage"):
                k, v = body.split(":", 1)
                header[k.strip()] = v.strip()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = np.loadtxt(path, comments="#", ndmin=2)
    if names is None or data.shape[1] != len(names):
        sys.exit(f"{path}: unexpected column layout")
    return header, {n: data[:, i] for i, n in enumerate(names)}


def load_run(d, source):
    """Collect NVE data from final files and/or the checkpoint."""
    thermo_path = os.path.join(d, "thermo_nve.log")
    if not os.path.exists(thermo_path):
        raise FileNotFoundError(f"{thermo_path} missing: NVE has not started")
    header, th = read_thermo(thermo_path)
    dt_fs = float(header.get("dt_fs", 0.5))
    summary = {}
    if os.path.exists(os.path.join(d, "summary.json")):
        with open(os.path.join(d, "summary.json")) as fh:
            summary = json.load(fh)

    final = None
    if os.path.exists(os.path.join(d, "nve_com.npz")):
        with np.load(os.path.join(d, "nve_com.npz")) as z:
            final = {"com_t": z["times_fs"], "com": z["unwrapped_com"], "com_v": z["com_vel_A_fs"]}
        mp = os.path.join(d, "nve_modes.npz")
        if os.path.exists(mp):
            with np.load(mp) as z:
                final["modes"] = np.stack([z["times_fs"], z["T_trans_K"], z["T_rot_K"], z["T_vib_K"]], axis=1)
        sp = os.path.join(d, "nve_stress.npz")
        if os.path.exists(sp):
            with np.load(sp) as z:
                final.update(st_t=z["times_fs"], st_vir=z["P_virial_GPa"], st_kin=z["P_kinetic_GPa"])
                if "P_mol_virial_GPa" in z.files:
                    final.update(st_mvir=z["P_mol_virial_GPa"], st_mkin=z["P_mol_kinetic_GPa"])
    ckpt = None
    cp = os.path.join(d, "restart.npz")
    if os.path.exists(cp):
        with np.load(cp) as z:
            if str(z["stage"]) == "nve":
                ckpt = {"com_t": z["com_t"], "com": z["com"], "com_v": z["com_v"],
                        "nve_step": int(z["nve_step"]), "cell": z["cell"]}
                if "modes" in z.files and len(z["modes"]):
                    ckpt["modes"] = z["modes"]
                if "stress_t" in z.files and len(z["stress_t"]):
                    ckpt.update(st_t=z["stress_t"], st_vir=z["stress_vir"], st_kin=z["stress_kin"])
                    if "stress_molvir" in z.files:
                        ckpt.update(st_mvir=z["stress_molvir"], st_mkin=z["stress_molkin"])
    n_final = len(final["com_t"]) if final else 0
    n_ckpt = len(ckpt["com_t"]) if ckpt else 0
    if source == "final" or (source == "auto" and n_final >= n_ckpt and n_final > 0):
        if not final:
            raise FileNotFoundError(f"{d}: no nve_com.npz (NVE not finished); use --source checkpoint")
        data, used = final, "final files (nve_com.npz, nve_stress.npz)"
    else:
        if not ckpt:
            raise FileNotFoundError(f"{d}: no NVE checkpoint yet (first NVE checkpoint not reached)")
        data, used = ckpt, f"checkpoint restart.npz (NVE step {ckpt['nve_step']})"
    t_end = data["com_t"][-1]
    keep = th["time_ps"] * 1000 <= t_end + 1e-6
    th = {k: v[keep] for k, v in th.items()}
    return {"dir": d, "header": header, "thermo": th, "dt_fs": dt_fs, "summary": summary,
            "data": data, "source": used}


# ---------------------------------------------------------------- numerics
def msd_fft(x):
    """Multi-origin MSD, averaged over the 2nd axis. x: (n, m, d) -> (n,)."""
    n = x.shape[0]
    sq = np.sum(x ** 2, axis=2)                                     # (n, m)
    f = np.fft.rfft(x, n=2 * n, axis=0)
    s2 = np.fft.irfft(np.sum(f * f.conj(), axis=2), n=2 * n, axis=0)[:n]
    cnt = (n - np.arange(n))[:, None]
    S = np.vstack([np.zeros((1, sq.shape[1])), np.cumsum(sq, axis=0)])   # S[k] = sum sq[:k]
    lag = np.arange(n)
    s1 = S[n - lag] + (S[n] - S[lag])                               # sum_{i<n-m} sq[i] + sum_{i>=m} sq[i]
    return ((s1 - 2 * s2) / cnt).mean(axis=1)


def acf_fft(x, nlag=None):
    """Multi-origin autocorrelation <x(0).x(t)> averaged over the 2nd axis. x: (n, m, d)."""
    n = x.shape[0]
    nlag = n if nlag is None else min(nlag, n)
    f = np.fft.rfft(x, n=2 * n, axis=0)
    ac = np.fft.irfft(np.sum(f * f.conj(), axis=2), n=2 * n, axis=0)[:nlag]
    return (ac / (n - np.arange(nlag))[:, None]).mean(axis=1)


def cumtrapz(y, dx):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)])


def block_stderr(x, n_blocks=10):
    x = np.asarray(x)
    n_blocks = min(n_blocks, len(x) // 2)
    if n_blocks < 2:
        return float("nan")
    b = len(x) // n_blocks
    means = x[: b * n_blocks].reshape(n_blocks, b).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(n_blocks))


def sem(vals):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    return float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")


def fit_msd(t, msd, window_fs):
    """D from the MSD slope over lag times window_fs = (lo, hi), fs; and the log-log slope there."""
    m = (t >= window_fs[0]) & (t <= window_fs[1]) & (t > 0)
    if m.sum() < 3:
        return float("nan"), float("nan")
    slope = np.polyfit(t[m], msd[m], 1)[0]
    loglog = np.polyfit(np.log(t[m]), np.log(msd[m]), 1)[0]
    return slope / 6 * A2_FS_TO_M2_S, float(loglog)


def local_slope_D(t, msd, n_pts=60, r=1.25):
    """D(t) = (1/6) dMSD/dt from a centred difference over [t/r, t*r], log-spaced t."""
    if len(t) < 10:
        return np.array([]), np.array([])
    tg = np.logspace(np.log10(t[1] * r), np.log10(t[-1] / r), n_pts)
    lo, hi = np.interp(tg / r, t, msd), np.interp(tg * r, t, msd)
    return tg, (hi - lo) / (tg * r - tg / r) / 6 * A2_FS_TO_M2_S


def clip_window(win, t_max_ps, frac_hi, name, notes, why="run only {:.3g} ps long"):
    """Clip a (lo, hi) ps window so hi <= frac_hi * t_max_ps; note if clipped."""
    lo, hi = win
    cap = frac_hi * t_max_ps
    if hi > cap:
        hi = cap
        lo = min(lo, 0.5 * hi)
        notes.append(f"{name} window clipped to {lo:.3g}-{hi:.3g} ps ({why.format(t_max_ps)})")
    return lo, hi


def shear_components(P):
    """(n, 6) Voigt xx yy zz yz xz xy -> (n, 5, 1) shear components, mean removed."""
    s = np.stack([P[:, 3], P[:, 4], P[:, 5], 0.5 * (P[:, 0] - P[:, 1]), 0.5 * (P[:, 1] - P[:, 2])], axis=1)
    return (s - s.mean(axis=0))[:, :, None]


def tdm_fit(t_ps, eta_mean, eta_std, t_min_ps=0.2):
    """Time-decomposition method: cutoff where std > 0.4 mean, double-exp fit weighted by std fit.

    The cutoff search and the fit both start at t_min_ps: before that the atomic running integral
    oscillates with the intramolecular vibrations and swings close to zero, where std/|mean| is large,
    so the 40 % test would fire at ~20 fs. Always returns a dict; on failure it has an 'error' key."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return {"error": "scipy not installed in this python: use the mace-ng env for the analysis"}
    ok = (t_ps > 0) & np.isfinite(eta_std) & (eta_std > 0)
    t_min_ps = min(t_min_ps, 0.5 * float(t_ps[-1]))
    bad = ok & (t_ps >= t_min_ps) & (eta_std > 0.4 * np.abs(eta_mean))
    t_cut = float(t_ps[np.argmax(bad)]) if bad.any() else float(t_ps[-1])
    m = ok & (t_ps >= t_min_ps) & (t_ps <= t_cut)
    if m.sum() < 8:
        return {"t_cut_ps": t_cut, "t_min_ps": t_min_ps,
                "error": f"only {int(m.sum())} lags between t_min {t_min_ps:.3g} ps and the cutoff {t_cut:.3g} ps"}
    b = np.polyfit(np.log(t_ps[m]), np.log(eta_std[m]), 1)[0]
    plate = float(eta_mean[m][-max(1, m.sum() // 5):].mean())

    def f(t, e1, tau1, e2, tau2):
        return e1 * (1 - np.exp(-t / tau1)) + e2 * (1 - np.exp(-t / tau2))

    try:
        # both amplitudes >= 0 and both decay times <= t_cut (as in the TDM paper: A > 0, 0 <= alpha <= 1)
        popt, _ = curve_fit(f, t_ps[m], eta_mean[m],
                            p0=[0.5 * abs(plate), min(0.05, 0.3 * t_cut), 0.5 * abs(plate), min(0.5, 0.8 * t_cut)],
                            sigma=t_ps[m] ** b,
                            bounds=([0, 1e-4, 0, 1e-4], [np.inf, t_cut, np.inf, t_cut]),
                            maxfev=20000)
    except Exception as e:
        return {"t_cut_ps": t_cut, "t_min_ps": t_min_ps, "b": float(b), "plateau_in_cut": plate,
                "error": f"{type(e).__name__}: {e}"}
    eta_inf = float(popt[0] + popt[2])
    out = {"eta_inf": eta_inf, "params": [float(v) for v in popt], "t_cut_ps": t_cut, "t_min_ps": t_min_ps,
           "b": float(b), "plateau_in_cut": plate, "curve": f(t_ps, *popt)}
    if not (0.5 * plate <= eta_inf <= 2 * plate) or plate <= 0:
        out["error"] = f"fit eta_inf {eta_inf:.4g} inconsistent with the running integral ({plate:.4g})"
        del out["eta_inf"]
    return out


def yeh_hummer_dD(T, eta_Pa_s, L_m):
    return XI_CUBIC * KB * T / (6 * np.pi * eta_Pa_s * L_m)


def coolprop_eta(T, rho_g_cm3):
    try:
        import CoolProp.CoolProp as CP
        return float(CP.PropsSI("V", "T", T, "Dmass", rho_g_cm3 * 1000, "Methane"))
    except Exception:
        return None


def fmtD(D_m2s, err=None):
    """'x.xxxe-08 m^2/s = xx.xx e-5 cm^2/s' with optional error."""
    c = D_m2s * M2_S_TO_CM2_S / 1e-5
    if err is None or not np.isfinite(err):
        return f"{D_m2s:.4e} m^2/s = {c:6.2f} x1e-5 cm^2/s"
    ce = err * M2_S_TO_CM2_S / 1e-5
    return f"{D_m2s:.4e} +/- {err:.1e} m^2/s = {c:6.2f} +/- {ce:.2f} x1e-5 cm^2/s"


def dict_D(D, err):
    return {"m2_s": D, "err_m2_s": err, "cm2_s": D * M2_S_TO_CM2_S, "err_cm2_s": err * M2_S_TO_CM2_S,
            "e-5_cm2_s": D * M2_S_TO_CM2_S / 1e-5, "err_e-5_cm2_s": err * M2_S_TO_CM2_S / 1e-5}


def viscosity(tag, st_t, P_tot, V_m3, Tm, args, notes):
    """Green-Kubo (TDM + plateau) and Einstein-Helfand viscosity from one pressure tensor (n, 6), GPa."""
    eta, curves = {}, {}
    dts = st_t[1] - st_t[0]
    span_ps = st_t[-1] / 1000
    S = shear_components(P_tot)                                      # (n, 5, 1)
    pref = V_m3 / (KB * Tm) * GPA2_FS_TO_PA2_S * 1e3                 # -> mPa s per (GPa^2 fs)
    # (a) Green-Kubo, time-decomposition over blocks
    nb = args.eta_blocks
    blk_len_ps = span_ps / nb
    maxlag_ps = min(args.eta_maxlag, 0.5 * blk_len_ps)
    if maxlag_ps < args.eta_maxlag:
        notes.append(f"eta max lag clipped to {maxlag_ps:.3g} ps ({nb} blocks of {blk_len_ps:.3g} ps)")
    nlag = int(maxlag_ps * 1000 / dts) + 1
    lag = np.arange(nlag) * dts / 1000
    acf_full = acf_fft(S, nlag)                                      # GPa^2
    eta_full = cumtrapz(acf_full, dts) * pref
    runs = []
    for blk in np.array_split(np.arange(len(st_t)), nb):
        Sb = S[blk]  # global mean removed: a per-block mean would bias the integral low by ~2t/T_block
        runs.append(cumtrapz(acf_fft(Sb, nlag), dts) * pref)
    runs = np.array(runs)
    e_mean, e_std = runs.mean(axis=0), runs.std(axis=0, ddof=1)
    elo, ehi = clip_window(args.eta_window, maxlag_ps, 1.0, "eta GK plateau", notes, why="max lag only {:.3g} ps")
    w = (lag >= elo) & (lag <= ehi)
    plateau = float(eta_full[w].mean()) if w.sum() >= 2 else float("nan")
    blk_plateaus = runs[:, w].mean(axis=1) if w.sum() >= 2 else np.full(nb, np.nan)
    plateau_err = sem(blk_plateaus)
    tdm = tdm_fit(lag, e_mean, e_std, args.tdm_tmin)
    if "eta_inf" in tdm:
        eta_gk = tdm["eta_inf"]
        eta_gk_how = (f"TDM double-exp fit ({tdm['t_min_ps']:.3g}-{tdm['t_cut_ps']:.3g} ps, b {tdm['b']:.2f}; "
                      f"plateau mean {plateau:.4f})")
    else:
        eta_gk, eta_gk_how = plateau, f"plateau mean (TDM fit failed: {tdm['error']})"
        notes.append(f"TDM fit failed: eta_GK is the plain plateau mean ({tdm['error']})")
    eta["GK"] = {"eta_mPas": eta_gk, "err_mPas": plateau_err, "method": eta_gk_how,
                 "plateau_mPas": plateau, "plateau_window_ps": [elo, ehi], "blocks": nb,
                 "block_plateaus_mPas": blk_plateaus.tolist(), "max_lag_ps": maxlag_ps,
                 "acf0_GPa2": float(acf_full[0]),
                 "tdm": {k: v for k, v in tdm.items() if k != "curve"}}
    log(f"[eta_GK:{tag}] <P_shear^2> = {acf_full[0]:.4e} GPa^2; plateau {elo:.3g}-{ehi:.3g} ps: "
        f"{plateau:.4f} +/- {plateau_err:.4f} mPa s ({nb} blocks)")
    log(f"[eta_GK:{tag}] eta = {eta_gk:.4f} +/- {plateau_err:.4f} mPa s  [{eta_gk_how}]")
    curves.update(eta_lag_ps=lag, eta_acf_GPa2=acf_full, eta_gk_full_mPas=eta_full,
                  eta_gk_mean_mPas=e_mean, eta_gk_std_mPas=e_std)
    if "curve" in tdm:
        curves["eta_tdm_fit_mPas"] = tdm["curve"]
    # (b) Einstein-Helfand
    Iint = np.stack([cumtrapz(S[:, k, 0], dts) for k in range(S.shape[1])], axis=1)[:, :, None]  # GPa fs
    nmax = min(len(st_t), int(0.5 * span_ps * 1000 / dts) + 1)
    G = msd_fft(Iint)[:nmax]
    tG = np.arange(nmax) * dts / 1000
    hlo, hhi = clip_window(args.eh_window, span_ps, 0.2, "Einstein-Helfand", notes)
    wm = (tG >= hlo) & (tG <= hhi)
    pref_eh = V_m3 / (2 * KB * Tm) * GPA2_FS_TO_PA2_S * 1e3          # mPa s per (GPa^2 fs^2 / fs)
    eta_eh = float(np.polyfit(tG[wm] * 1000, G[wm], 1)[0] * pref_eh) if wm.sum() >= 3 else float("nan")
    eh_blocks = []
    for blk in np.array_split(np.arange(len(st_t)), args.blocks):
        Sb = S[blk]
        Ib = np.stack([cumtrapz(Sb[:, k, 0], dts) for k in range(5)], axis=1)[:, :, None]
        Gb = msd_fft(Ib)
        tb = np.arange(len(Gb)) * dts / 1000
        wb = (tb >= hlo) & (tb <= hhi)
        eh_blocks.append(float(np.polyfit(tb[wb] * 1000, Gb[wb], 1)[0] * pref_eh) if wb.sum() >= 3 else np.nan)
    eta_eh_err = sem(eh_blocks)
    eta["EH"] = {"eta_mPas": eta_eh, "err_mPas": eta_eh_err, "fit_window_ps": [hlo, hhi],
                 "block_values_mPas": [float(v) for v in eh_blocks]}
    log(f"[eta_EH:{tag}] slope {hlo:.3g}-{hhi:.3g} ps: eta = {eta_eh:.4f} +/- {eta_eh_err:.4f} mPa s "
        f"({args.blocks} blocks)")
    r = eta_eh / eta_gk if eta_gk else float("nan")
    log(f"[eta:{tag}] EH / GK = {r:.3f}")
    # local slope of G (what the fit measures), log-spaced, up to 3x the fit window
    tl, sl = local_slope_D(tG[tG <= 3 * hhi] * 1000, G[tG <= 3 * hhi])      # returns slope/6 * 1e-5
    curves.update(eh_t_ps=tG, eh_G=G, eh_loc_t_ps=tl / 1000, eh_eta_local_mPas=sl * 6 / A2_FS_TO_M2_S * pref_eh)
    return eta, curves


# ---------------------------------------------------------------- one state point
def analyse(run, args):
    d, th, data = run["dir"], run["thermo"], run["data"]
    notes, res, curves = [], {"dir": os.path.abspath(d), "source": run["source"]}, {}
    cfg = run["summary"].get("config", {})
    log(f"========== {d}")
    log(f"[load] {run['source']}")
    res["model"] = run["header"].get("model", cfg.get("model"))
    res["T_target_K"] = float(run["header"].get("T_K", cfg.get("temperature", "nan")))
    res["P_target_GPa"] = float(run["header"].get("P_target_GPa", cfg.get("pressure", "nan")))
    n_mol = int(run["header"].get("n_molecules", cfg.get("n_molecules", data["com"].shape[1])))
    res["n_molecules"] = n_mol

    # ---- 1. thermo
    V_A3 = float(th["V_A3"].mean())
    L_A = V_A3 ** (1 / 3)
    rho = float(th["rho_g_cm3"].mean())
    T, P = th["T_K"], th["P_GPa"]
    t_ps = th["time_ps"] - th["time_ps"][0]
    dE = th["dEtot_per_mol_meV"]
    res["thermo"] = {"n_rows": len(T), "nve_ps": float(t_ps[-1]), "mean_T_K": float(T.mean()),
                     "stderr_T_K": block_stderr(T), "mean_P_GPa": float(P.mean()), "stderr_P_GPa": block_stderr(P),
                     "std_P_GPa": float(P.std()), "volume_A3": V_A3, "box_length_A": L_A, "density_g_cm3": rho,
                     "max_abs_dE_per_mol_meV": float(np.abs(dE).max()),
                     "drift_meV_per_mol_per_ps": float(np.polyfit(t_ps, dE, 1)[0]) if len(dE) > 2 else float("nan")}
    Tm = res["thermo"]["mean_T_K"]
    log(f"[thermo] {res['model']}, {n_mol} CH4, target {res['T_target_K']:g} K / {res['P_target_GPa']:g} GPa; "
        f"NVE {t_ps[-1]:.2f} ps ({len(T)} rows)")
    log(f"[thermo] <T> = {Tm:.2f} +/- {res['thermo']['stderr_T_K']:.2f} K, <P> = {P.mean():.4f} +/- "
        f"{res['thermo']['stderr_P_GPa']:.4f} GPa (std {P.std():.3f}), rho = {rho:.5f} g/cm3, L = {L_A:.4f} A")
    log(f"[thermo] energy: max|dE|/mol = {res['thermo']['max_abs_dE_per_mol_meV']:.3f} meV, "
        f"drift {res['thermo']['drift_meV_per_mol_per_ps']:+.5f} meV/mol/ps")
    curves.update(thermo_t_ps=t_ps, thermo_T=T, thermo_P=P)

    # ---- 1b. equipartition: translational vs total temperature per block (+ rot/vib if recorded)
    # The total momentum is zero, so COM translation has 3N-3 dof, not 3N: the plain per-molecule
    # average M<v^2>/3k is low by (N-1)/N (3 % for N = 32). Corrected here, as in run_statepoint.py.
    M_kg = M_CH4_AMU * 1.66053906660e-27
    dof_corr = n_mol / (n_mol - 1) if n_mol > 1 else 1.0
    Ttr_series = M_kg * (np.asarray(data["com_v"], float) ** 2).sum(-1).mean(-1) * 1e10 / (3 * KB) * dof_corr
    nb = 5
    Ttr_b = [float(b.mean()) for b in np.array_split(Ttr_series, nb)]
    Ttot_b = [float(b.mean()) for b in np.array_split(T, nb)]
    eq = {"T_trans_blocks_K": Ttr_b, "T_total_blocks_K": Ttot_b, "T_trans_mean_K": float(Ttr_series.mean()),
          "T_trans_dof": "3N-3"}
    log(f"[equip] T_trans (COM, 3N-3 dof) per {nb} blocks: " + " ".join(f"{v:.0f}" for v in Ttr_b)
        + f"  | T_total: " + " ".join(f"{v:.0f}" for v in Ttot_b) + " K")
    if "modes" in data and len(data["modes"]):
        md = np.asarray(data["modes"], float)
        mb = np.array([b[:, 1:].mean(0) for b in np.array_split(md, nb)])
        eq.update(T_rot_blocks_K=mb[:, 1].tolist(), T_vib_blocks_K=mb[:, 2].tolist(),
                  T_rot_mean_K=float(md[:, 2].mean()), T_vib_mean_K=float(md[:, 3].mean()))
        log("[equip] T_rot per block: " + " ".join(f"{v:.0f}" for v in mb[:, 1])
            + "  | T_vib: " + " ".join(f"{v:.0f}" for v in mb[:, 2]) + " K")
    dev = Ttr_series.mean() / T.mean() - 1
    eq["T_trans_over_T_total_minus_1"] = float(dev)
    if abs(dev) > 0.02:
        msg = (f"T_trans = {Ttr_series.mean():.1f} K vs T_total = {T.mean():.1f} K ({100 * dev:+.1f} %): not in "
               f"equipartition; D corresponds to the translational temperature")
        notes.append(msg)
        log(f"[equip] WARNING: {msg}")
    res["equipartition"] = eq

    # ---- 2. D from MSD
    ct = data["com_t"] - data["com_t"][0]
    com, com_v = np.asarray(data["com"], float), np.asarray(data["com_v"], float)
    T_run_ps = ct[-1] / 1000
    msd = msd_fft(com - com[0])
    T_blk = ct[-1] / args.blocks
    win_fs = (args.msd_fit[0] * T_blk, args.msd_fit[1] * T_blk)   # same lag window for the full run and blocks
    D_msd, loglog = fit_msd(ct, msd, win_fs)
    blocks = []
    for blk in np.array_split(np.arange(len(ct)), args.blocks):
        if len(blk) > 20:
            blocks.append(fit_msd(ct[blk] - ct[blk[0]], msd_fft(com[blk] - com[blk[0]]), win_fs)[0])
    D_msd_err = sem(blocks)
    tg, D_loc = local_slope_D(ct, msd)
    res["D_MSD"] = dict_D(D_msd, D_msd_err) | {"loglog_slope": loglog, "fit_window_ps": [w / 1000 for w in win_fs],
                                                 "block_values_m2_s": blocks}
    log(f"[D_MSD] {len(ct)} COM frames every {ct[1] - ct[0]:.1f} fs over {T_run_ps:.2f} ps; fit lag window "
        f"{win_fs[0] / 1000:.3g}-{win_fs[1] / 1000:.3g} ps (same for the {args.blocks} blocks)")
    log(f"[D_MSD] {fmtD(D_msd, D_msd_err)}  (log-log slope {loglog:.3f})")
    if not 0.9 < loglog < 1.1:
        notes.append(f"MSD log-log slope {loglog:.3f} in the fit window: not diffusive yet")
        log("[D_MSD] WARNING: log-log slope not ~1: not diffusive in the fit window (run too short?)")
    curves.update(msd_t_fs=ct, msd_A2=msd, Dloc_t_fs=tg, Dloc_m2_s=D_loc)

    # ---- 3. D from VACF (Green-Kubo)
    dtc = ct[1] - ct[0]
    vacf = acf_fft(com_v)
    D_run = cumtrapz(vacf, dtc) / 3 * A2_FS_TO_M2_S
    lo, hi = clip_window(args.gk_window, T_run_ps, 0.25, "D_GK", notes)
    lag_ps = ct / 1000
    win = (lag_ps >= lo) & (lag_ps <= hi)
    D_gk = float(D_run[win].mean()) if win.sum() >= 2 else float("nan")
    gk_blocks = []
    for blk in np.array_split(np.arange(len(ct)), args.blocks):
        vb = acf_fft(com_v[blk])
        Db = cumtrapz(vb, dtc) / 3 * A2_FS_TO_M2_S
        wb = (lag_ps[:len(Db)] >= lo) & (lag_ps[:len(Db)] <= hi)
        gk_blocks.append(float(Db[wb].mean()) if wb.sum() >= 2 else float("nan"))
    D_gk_err = sem(gk_blocks)
    m_mol_kg = M_CH4_AMU * 1.66053906660e-27
    # expected <v^2> per molecule at T with zero total momentum: 3kT/m * (N-1)/N
    v2_exp = 3 * KB * Tm / m_mol_kg * 1e-10 / dof_corr  # m^2/s^2 -> A^2/fs^2
    ratio = D_gk / D_msd if np.isfinite(D_gk) and D_msd else float("nan")
    ratio_err = ratio * np.hypot(D_gk_err / D_gk, D_msd_err / D_msd) if np.isfinite(ratio) else float("nan")
    res["D_GK"] = dict_D(D_gk, D_gk_err) | {"plateau_window_ps": [lo, hi], "vacf0_A2_fs2": float(vacf[0]),
                                             "vacf0_expected_A2_fs2": v2_exp, "block_values_m2_s": gk_blocks,
                                             "D_GK_over_D_MSD": ratio, "ratio_err": ratio_err}
    log(f"[D_GK] <v(0)^2> = {vacf[0]:.4e} A^2/fs^2 (3kT/m x (N-1)/N: {v2_exp:.4e}, ratio {vacf[0] / v2_exp:.3f})")
    log(f"[D_GK] {fmtD(D_gk, D_gk_err)}  (plateau {lo:.3g}-{hi:.3g} ps)")
    log(f"[D]    D_GK / D_MSD = {ratio:.3f} +/- {ratio_err:.3f}"
        + ("" if not np.isfinite(ratio) or abs(ratio - 1) < 3 * max(ratio_err, 0.02) else "  <-- disagree"))
    nl = min(len(vacf), int(hi * 1000 / dtc * 1.5) + 2)
    curves.update(vacf_t_fs=ct[:nl], vacf=vacf[:nl], Dgk_run_m2_s=D_run[:nl])

    # ---- 4. viscosity (molecular tensor first if recorded: it converges faster; atomic as cross-check)
    eta = {}
    V_m3 = V_A3 * 1e-30
    if "st_t" in data and len(data["st_t"]) > 20:
        st_t = data["st_t"] - data["st_t"][0]
        dts = st_t[1] - st_t[0]
        log(f"[eta] pressure tensor: {len(st_t)} samples every {dts:.2f} fs, {st_t[-1] / 1000:.2f} ps "
            f"(from NVE t = {data['st_t'][0] / 1000:.2f} ps)")
        if data["st_t"][0] > data["com_t"][0] + 1e-6:
            notes.append(f"pressure tensor only from NVE t = {data['st_t'][0] / 1000:.2f} ps (older run, extended)")
        tensors = []
        if "st_mvir" in data:
            tensors.append(("molecular", np.asarray(data["st_mvir"], float) + np.asarray(data["st_mkin"], float)))
        tensors.append(("atomic", np.asarray(data["st_vir"], float) + np.asarray(data["st_kin"], float)))
        for name, P_tot in tensors:
            log(f"[eta] {name} tensor: <P> = {P_tot[:, :3].mean():.4f} GPa, <P_shear^2>^1/2 = "
                f"{np.sqrt((shear_components(P_tot) ** 2).mean()):.4f} GPa")
        eta["tensor"] = tensors[0][0]
        eta["by_tensor"] = {}
        for i, (name, P_tot) in enumerate(tensors):
            sub, cv = viscosity(name, st_t, P_tot, V_m3, Tm, args, notes if i == 0 else [])
            eta["by_tensor"][name] = sub
            if i == 0:
                eta.update(sub)
                curves.update(cv)
            else:
                curves.update({"alt_" + k: v for k, v in cv.items()})
                curves["alt_name"] = np.array(name)
        if len(tensors) == 2:
            a, m = eta["by_tensor"]["atomic"], eta["by_tensor"]["molecular"]
            log(f"[eta] molecular vs atomic: GK {m['GK']['eta_mPas']:.4f} +/- {m['GK']['err_mPas']:.4f} vs "
                f"{a['GK']['eta_mPas']:.4f} +/- {a['GK']['err_mPas']:.4f};  EH {m['EH']['eta_mPas']:.4f} vs "
                f"{a['EH']['eta_mPas']:.4f} mPa s (same limit expected)")
    else:
        log("[eta] no pressure-tensor record in this run (older run_statepoint version, or NVE too short): "
            "no viscosity" + (f"; using --eta-mPas {args.eta_mPas}" if args.eta_mPas else ""))
        notes.append("no pressure-tensor record: no computed viscosity")
    eta_ref = coolprop_eta(Tm, rho)
    if eta_ref is not None:
        eta["CoolProp_reference_mPas"] = eta_ref * 1e3
        log(f"[eta] CoolProp (experiment-based correlation) at T = {Tm:.1f} K, rho = {rho:.4f}: "
            f"{eta_ref * 1e3:.4f} mPa s (for comparison only)")
    res["eta"] = eta

    # ---- 5. Yeh-Hummer
    choices = []
    if args.eta_mPas:
        choices.append(("user", args.eta_mPas, 0.0))
    if "GK" in eta and np.isfinite(eta["GK"]["eta_mPas"]) and eta["GK"]["eta_mPas"] > 0:
        choices.append(("GK", eta["GK"]["eta_mPas"], eta["GK"]["err_mPas"]))
    if "EH" in eta and np.isfinite(eta["EH"]["eta_mPas"]) and eta["EH"]["eta_mPas"] > 0:
        choices.append(("EH", eta["EH"]["eta_mPas"], eta["EH"]["err_mPas"]))
    for name, sub in eta.get("by_tensor", {}).items():   # the other tensor's GK, as a cross-check
        if name != eta.get("tensor") and np.isfinite(sub["GK"]["eta_mPas"]) and sub["GK"]["eta_mPas"] > 0:
            choices.append((f"GK_{name}", sub["GK"]["eta_mPas"], sub["GK"]["err_mPas"]))
    if choices and "tensor" in eta:
        log(f"[YH] eta_GK / eta_EH below are from the {eta['tensor']} pressure tensor")
    yh = {}
    L_m = L_A * 1e-10
    for name, e_mPas, e_err in choices:
        dD = yeh_hummer_dD(Tm, e_mPas * 1e-3, L_m)
        dD_err = dD * (e_err / e_mPas if np.isfinite(e_err) else 0.0)
        for Dname, Dv, De in (("MSD", D_msd, D_msd_err), ("GK", D_gk, D_gk_err)):
            if not np.isfinite(Dv):
                continue
            Dinf = Dv + dD
            err = float(np.hypot(De if np.isfinite(De) else 0, dD_err))
            yh[f"D_{Dname}_eta_{name}"] = dict_D(Dinf, err) | {"eta_mPas": e_mPas, "dD_m2_s": dD,
                                                               "dD_percent_of_D_PBC": 100 * dD / Dv}
        log(f"[YH] eta_{name} = {e_mPas:.4f} mPa s, L = {L_A:.3f} A -> dD = {dD:.3e} m^2/s "
            f"({100 * dD / D_msd:.1f} % of D_MSD)")
    for k, v in yh.items():
        log(f"[YH] {k:16s}: D_inf = {fmtD(v['m2_s'], v['err_m2_s'])}")
    if not yh:
        log("[YH] no viscosity available: no Yeh-Hummer correction (pass --eta-mPas to force one)")
    res["yeh_hummer"] = yh
    res["notes"] = notes
    for n in notes:
        log(f"[note] {n}")
    return res, curves


# ---------------------------------------------------------------- plotting
def plot_one(res, curves, path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    title = (f"{res['model']}  {res['n_molecules']} CH4  {res['T_target_K']:g} K / {res['P_target_GPa']:g} GPa  "
             f"<T>={res['thermo']['mean_T_K']:.1f} K  <P>={res['thermo']['mean_P_GPa']:.3f} GPa  "
             f"rho={res['thermo']['density_g_cm3']:.4f}")
    fig.suptitle(title, fontsize=11)
    a = ax[0, 0]
    a.plot(curves["thermo_t_ps"], curves["thermo_P"], lw=0.5, color="C0")
    a.set_xlabel("NVE time (ps)"); a.set_ylabel("P (GPa)", color="C0")
    a2 = a.twinx(); a2.plot(curves["thermo_t_ps"], curves["thermo_T"], lw=0.5, color="C3"); a2.set_ylabel("T (K)", color="C3")
    a.set_title("thermo (NVE)")
    a = ax[0, 1]
    t, m = curves["msd_t_fs"][1:] / 1000, curves["msd_A2"][1:]
    a.loglog(t, m, label="COM MSD")
    D = res["D_MSD"]["m2_s"]
    if np.isfinite(D):
        a.loglog(t, 6 * D / A2_FS_TO_M2_S * t * 1000, "--", lw=0.8, label="6 D_MSD t")
    a.set_xlabel("t (ps)"); a.set_ylabel("MSD (A^2)"); a.legend(); a.set_title(f"MSD (log-log slope {res['D_MSD']['loglog_slope']:.2f})")
    a = ax[0, 2]
    sc = M2_S_TO_CM2_S / 1e-5
    if len(curves["Dloc_t_fs"]):
        a.semilogx(curves["Dloc_t_fs"] / 1000, curves["Dloc_m2_s"] * sc, label="MSD local slope / 6")
    tv = curves["vacf_t_fs"][1:] / 1000
    a.semilogx(tv, curves["Dgk_run_m2_s"][1:] * sc, label="GK running integral")
    for key, c in (("D_MSD", "C0"), ("D_GK", "C1")):
        v = res[key]["m2_s"] * sc
        if np.isfinite(v):
            a.axhline(v, color=c, ls=":", lw=1)
    lo, hi = res["D_GK"]["plateau_window_ps"]
    a.axvspan(lo, hi, color="C1", alpha=0.1)
    a.set_xlabel("t (ps)"); a.set_ylabel("D(t) (1e-5 cm^2/s)"); a.legend(); a.set_title("D(t)")
    if "eta_lag_ps" in curves:
        lag = curves["eta_lag_ps"]
        a = ax[1, 0]
        acf = curves["eta_acf_GPa2"]
        tname = res["eta"].get("tensor", "atomic")
        a.plot(lag, acf / acf[0], label=tname)
        if "alt_eta_acf_GPa2" in curves:
            aa = curves["alt_eta_acf_GPa2"]
            a.plot(curves["alt_eta_lag_ps"], aa / aa[0], color="grey", lw=0.6, label=str(curves["alt_name"]))
            a.legend(fontsize=8)
        a.axhline(0, color="k", lw=0.5)
        a.set_xlabel("lag (ps)"); a.set_ylabel("<P_ab(0)P_ab(t)> / <P_ab^2>"); a.set_title("shear-stress ACF (5 comps)")
        a.set_xlim(0, min(2, lag[-1]))
        a = ax[1, 1]
        mu, sd = curves["eta_gk_mean_mPas"], curves["eta_gk_std_mPas"]
        a.plot(lag, curves["eta_gk_full_mPas"], lw=1, label=f"full-run ACF ({tname})")
        if "alt_eta_gk_full_mPas" in curves:
            a.plot(curves["alt_eta_lag_ps"], curves["alt_eta_gk_full_mPas"], color="grey", lw=0.6,
                   label=f"full-run ACF ({curves['alt_name']})")
        a.plot(lag, mu, lw=1, label="block mean")
        a.fill_between(lag, mu - sd, mu + sd, alpha=0.2, label="block std")
        if "eta_tdm_fit_mPas" in curves:
            a.plot(lag, curves["eta_tdm_fit_mPas"], "k--", lw=1, label="TDM fit")
        g = res["eta"]["GK"]
        a.axvspan(*g["plateau_window_ps"], color="C2", alpha=0.1)
        tdm = g.get("tdm", {})
        if "t_cut_ps" in tdm:
            a.axvline(tdm["t_cut_ps"], color="k", ls=":", lw=0.8, label="TDM cutoff")
        if "CoolProp_reference_mPas" in res["eta"]:
            a.axhline(res["eta"]["CoolProp_reference_mPas"], color="grey", ls=":", label="CoolProp ref")
        a.set_xlabel("lag (ps)"); a.set_ylabel("eta (mPa s)"); a.legend(fontsize=8)
        a.set_title(f"Green-Kubo eta ({tname}) = {g['eta_mPas']:.4f} mPa s")
        a = ax[1, 2]
        a.semilogx(curves["eh_loc_t_ps"], curves["eh_eta_local_mPas"], label="V/2kT dG/dt (local)")
        e = res["eta"]["EH"]
        a.axhline(e["eta_mPas"], color="C1", ls="--", label=f"slope fit {e['eta_mPas']:.4f}")
        a.axvspan(*e["fit_window_ps"], color="C1", alpha=0.1)
        a.set_xlabel("t (ps)"); a.set_ylabel("eta (mPa s)"); a.legend(fontsize=8); a.set_title(f"Einstein-Helfand ({tname})")
    else:
        for a in ax[1]:
            a.text(0.5, 0.5, "no pressure-tensor record", ha="center", va="center", transform=a.transAxes)
            a.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- N-series
def series_fit(results, out, show, do_plot):
    log(f"========== N-series Yeh-Hummer fit over {len(results)} runs -> {out}.json")
    Ns = [r["n_molecules"] for r in results]
    if len(set(Ns)) < 2:
        log("[series] need at least two different N: skipped")
        return
    Ts = np.array([r["thermo"]["mean_T_K"] for r in results])
    rhos = np.array([r["thermo"]["density_g_cm3"] for r in results])
    if np.ptp(rhos) / rhos.mean() > 0.005:
        log(f"[series] WARNING: densities differ by {100 * np.ptp(rhos) / rhos.mean():.2f} % across N "
            "(Yeh-Hummer assumes one state point)")
    L = np.array([r["thermo"]["box_length_A"] for r in results]) * 1e-10
    D = np.array([r["D_MSD"]["m2_s"] for r in results])
    De = np.array([r["D_MSD"]["err_m2_s"] for r in results])
    De = np.where(np.isfinite(De) & (De > 0), De, np.nanmean(De[De > 0]) if (De > 0).any() else 1.0)
    x = 1 / L
    w = 1 / De
    (slope, icpt), cov = np.polyfit(x, D, 1, w=w, cov="unscaled")
    s_err, i_err = np.sqrt(np.diag(cov))
    Tm = Ts.mean()
    eta_fit = -XI_CUBIC * KB * Tm / (6 * np.pi * slope) * 1e3 if slope < 0 else float("nan")
    eta_fit_err = abs(eta_fit * s_err / slope) if np.isfinite(eta_fit) else float("nan")
    out_d = {"N": Ns, "L_A": (L * 1e10).tolist(), "T_K": Ts.tolist(), "density_g_cm3": rhos.tolist(),
             "D_PBC_MSD_m2_s": D.tolist(), "D_PBC_err_m2_s": De.tolist(),
             "D_inf_fit": dict_D(float(icpt), float(i_err)), "slope_m3_s": float(slope),
             "eta_from_slope_mPas": eta_fit, "eta_from_slope_err_mPas": eta_fit_err, "per_run_YH": []}
    log(f"[series] fit D_PBC = D_inf + slope/L: D_inf = {fmtD(icpt, i_err)}")
    log(f"[series] eta from slope = {eta_fit:.4f} +/- {eta_fit_err:.4f} mPa s")
    for r in results:
        g = r["eta"].get("GK", {})
        yh = r["yeh_hummer"].get("D_MSD_eta_GK")
        out_d["per_run_YH"].append({"N": r["n_molecules"], "eta_GK_mPas": g.get("eta_mPas"),
                                    "D_inf_MSD_eta_GK": yh})
        log(f"[series] N={r['n_molecules']:4d} L={r['thermo']['box_length_A']:7.3f} A  D_PBC = "
            f"{fmtD(r['D_MSD']['m2_s'], r['D_MSD']['err_m2_s'])}"
            + (f"  eta_GK = {g['eta_mPas']:.4f}" if g else "")
            + (f"  D_inf(YH) = {yh['e-5_cm2_s']:.2f} +/- {yh['err_e-5_cm2_s']:.2f} x1e-5 cm^2/s" if yh else ""))
    with open(out + ".json", "w") as fh:
        json.dump(out_d, fh, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    log(f"wrote {out}.json")
    if not do_plot:
        return
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sc = M2_S_TO_CM2_S / 1e-5
    fig, a = plt.subplots(figsize=(7, 5))
    a.errorbar(x * 1e-10, D * sc, yerr=De * sc, fmt="o", label="D_PBC (MSD)")
    xx = np.linspace(0, x.max() * 1.05, 50)
    a.plot(xx * 1e-10, (icpt + slope * xx) * sc, "k--", lw=1,
           label=f"fit: D_inf = {icpt * sc:.2f} x1e-5 cm2/s, eta = {eta_fit:.3f} mPa s")
    yhs = [(1 / (r["thermo"]["box_length_A"]), r["yeh_hummer"]["D_MSD_eta_GK"]) for r in results
           if "D_MSD_eta_GK" in r["yeh_hummer"]]
    if yhs:
        a.errorbar([p[0] for p in yhs], [p[1]["e-5_cm2_s"] for p in yhs], yerr=[p[1]["err_e-5_cm2_s"] for p in yhs],
                   fmt="s", mfc="none", label="D_PBC + YH(eta_GK)")
    for n, xi_, di in zip(Ns, x * 1e-10, D * sc):
        a.annotate(f"N={n}", (xi_, di), textcoords="offset points", xytext=(4, -12), fontsize=8)
    a.set_xlabel("1/L (1/A)"); a.set_ylabel("D (1e-5 cm^2/s)"); a.set_xlim(left=0); a.legend(fontsize=8)
    a.set_title(f"Yeh-Hummer N-series, <T> = {Tm:.1f} K, rho = {rhos.mean():.4f} g/cm3")
    fig.tight_layout(); fig.savefig(out + ".png", dpi=130)
    log(f"wrote {out}.png")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    log(f"analyse_statepoint.py | python {platform.python_version()} on {platform.node()} | numpy {np.__version__}")
    if args.replot:
        n = 0
        for d in args.dirs:
            jp, cp = os.path.join(d, "analysis.json"), os.path.join(d, "analysis_curves.npz")
            if not (os.path.exists(jp) and os.path.exists(cp)):
                log(f"[skip] {d}: no analysis.json / analysis_curves.npz (run the analysis first)")
                continue
            with open(jp) as fh:
                res = json.load(fh)
            with np.load(cp) as z:
                curves = {k: z[k] for k in z.files}
            plot_one(res, curves, os.path.join(d, "analysis.png"), args.show)
            log(f"wrote {os.path.join(d, 'analysis.png')}")
            n += 1
        if args.series and os.path.exists(args.series + ".png") and args.show:
            log(f"[replot] N-series figure is {args.series}.png (re-run without --replot to redraw it)")
        return 0 if n else 1
    try:
        import scipy  # noqa: F401
    except ImportError:
        sys.exit("scipy is not installed in this python: run the analysis with the mace-ng env "
                 "(source ~/Scratch/envs/mace-env-ng.sh), then view with --replot --show")
    results = []
    for d in args.dirs:
        if not os.path.isdir(d):
            log(f"[skip] {d}: not a directory")
            continue
        try:
            run = load_run(d, args.source)
        except FileNotFoundError as e:
            log(f"[skip] {e}")
            continue
        res, curves = analyse(run, args)
        with open(os.path.join(d, "analysis.json"), "w") as fh:
            json.dump(res, fh, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else
                      (o.tolist() if isinstance(o, np.ndarray) else str(o)))
        log(f"wrote {os.path.join(d, 'analysis.json')}")
        np.savez(os.path.join(d, "analysis_curves.npz"), **curves)
        log(f"wrote {os.path.join(d, 'analysis_curves.npz')}")
        if not args.no_plot:
            try:
                plot_one(res, curves, os.path.join(d, "analysis.png"), args.show)
                log(f"wrote {os.path.join(d, 'analysis.png')}")
            except Exception as e:
                log(f"[plot] failed ({type(e).__name__}: {e}); results are in analysis.json")
        results.append(res)
    if args.series and results:
        series_fit(results, args.series, args.show, not args.no_plot)
    log(f"=== done: {len(results)} state point(s) analysed ===")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
