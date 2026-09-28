#!/usr/bin/env python
"""Parse analyse_statepoint.py console output into one CSV row per state point.

    python analysis/parse_statepoint_log.py analysis/data/mace_off24_analysis_log.txt \
        --out analysis/data/mace_off24_D.csv

D values are in 1e-5 cm^2/s (as printed), viscosities in mPa s, P in GPa.
"""
import argparse
import csv
import re
import sys

NUM = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"

PATTERNS = {
    "thermo_target": re.compile(rf"\[thermo\] (.+?), (\d+) CH4, target {NUM} K / {NUM} GPa; NVE {NUM} ps"),
    "thermo_state": re.compile(
        rf"\[thermo\] <T> = {NUM} \+/- {NUM} K, <P> = {NUM} \+/- {NUM} GPa \(std {NUM}\), "
        rf"rho = {NUM} g/cm3, L = {NUM} A"),
    "thermo_energy": re.compile(rf"\[thermo\] energy: max\|dE\|/mol = {NUM} meV, drift {NUM} meV/mol/ps"),
    "D_MSD": re.compile(rf"\[D_MSD\] {NUM} \+/- {NUM} m\^2/s = +{NUM} \+/- {NUM} x1e-5 cm\^2/s  \(log-log slope {NUM}\)"),
    "D_GK": re.compile(rf"\[D_GK\] {NUM} \+/- {NUM} m\^2/s = +{NUM} \+/- {NUM} x1e-5 cm\^2/s  \(plateau"),
    "ratio": re.compile(rf"\[D\] +D_GK / D_MSD = {NUM} \+/- {NUM}"),
    "eta_GK": re.compile(rf"\[eta_GK:atomic\] eta = {NUM} \+/- {NUM} mPa s"),
    "eta_EH": re.compile(rf"\[eta_EH:atomic\] slope [^:]*: eta = {NUM} \+/- {NUM} mPa s"),
    "eta_ref": re.compile(rf"\[eta\] CoolProp .*: {NUM} mPa s"),
    "YH": re.compile(rf"\[YH\] (D_(?:MSD|GK)_eta_(?:GK|EH)) +: D_inf = {NUM} \+/- {NUM} m\^2/s = +{NUM} \+/- {NUM} x1e-5"),
    "equip_warn": re.compile(r"\[equip\] WARNING"),
}

COLUMNS = [
    "statepoint", "model", "n_molecules", "T_target_K", "P_target_GPa", "nve_ps",
    "T_K", "T_err_K", "P_GPa", "P_err_GPa", "P_std_GPa", "rho_g_cm3", "L_A",
    "max_dE_meV_per_mol", "drift_meV_per_mol_ps",
    "D_MSD", "D_MSD_err", "msd_loglog_slope", "D_GK", "D_GK_err", "GK_over_MSD", "GK_over_MSD_err",
    "eta_GK_mPas", "eta_GK_err", "eta_EH_mPas", "eta_EH_err", "eta_ref_mPas",
    "D_MSD_eta_GK", "D_MSD_eta_GK_err", "D_GK_eta_GK", "D_GK_eta_GK_err",
    "D_MSD_eta_EH", "D_MSD_eta_EH_err", "D_GK_eta_EH", "D_GK_eta_EH_err",
    "msd_diffusive", "equipartition_ok",
]


def parse(lines):
    rows, cur = [], None
    for line in lines:
        m = re.search(r"========== (\S+)", line)
        if m:
            if cur:
                rows.append(cur)
            cur = {"statepoint": m.group(1), "equipartition_ok": True}
            continue
        if cur is None:
            continue
        for key, pat in PATTERNS.items():
            m = pat.search(line)
            if not m:
                continue
            g = m.groups()
            if key == "thermo_target":
                cur.update(model=g[0], n_molecules=int(g[1]), T_target_K=float(g[2]),
                           P_target_GPa=float(g[3]), nve_ps=float(g[4]))
            elif key == "thermo_state":
                cur.update(T_K=g[0], T_err_K=g[1], P_GPa=g[2], P_err_GPa=g[3], P_std_GPa=g[4],
                           rho_g_cm3=g[5], L_A=g[6])
            elif key == "thermo_energy":
                cur.update(max_dE_meV_per_mol=g[0], drift_meV_per_mol_ps=g[1])
            elif key == "D_MSD":
                slope = float(g[4])
                cur.update(D_MSD=g[2], D_MSD_err=g[3], msd_loglog_slope=g[4],
                           msd_diffusive=0.9 <= slope <= 1.1)
            elif key == "D_GK":
                cur.update(D_GK=g[2], D_GK_err=g[3])
            elif key == "ratio":
                cur.update(GK_over_MSD=g[0], GK_over_MSD_err=g[1])
            elif key == "eta_GK":
                cur.update(eta_GK_mPas=g[0], eta_GK_err=g[1])
            elif key == "eta_EH":
                cur.update(eta_EH_mPas=g[0], eta_EH_err=g[1])
            elif key == "eta_ref":
                cur.update(eta_ref_mPas=g[0])
            elif key == "YH":
                cur.update({g[0]: g[3], g[0] + "_err": g[4]})
            elif key == "equip_warn":
                cur["equipartition_ok"] = False
            break
    if cur:
        rows.append(cur)
    return [r for r in rows if "D_MSD" in r]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    with open(args.log) as fh:
        rows = parse(fh)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    print(f"{len(rows)} state points -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
