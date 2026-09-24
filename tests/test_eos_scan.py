"""End-to-end EOS scan with a toy calculator, and feeding its fit to run_state_point.py."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from toy_ch4_calculator import ToyCH4Calculator

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.get_calculator = lambda **kwargs: ToyCH4Calculator()
    return module


SCAN_ARGS = [
    "--n-molecules", "32", "--device", "cpu", "--densities", "0.25", "0.30", "0.35", "0.40",
    "--first-equil-steps", "200", "--equil-steps", "50", "--sample-steps", "200",
    "--thermo-interval", "5", "--print-interval", "100", "--n-blocks", "4", "--fit-degree", "2",
    "--target-pressures", "0.2",
]


def test_eos_scan_then_production_from_fit(tmp_path):
    scan_dir = tmp_path / "eos"
    load("eos_scan").main(SCAN_ARGS + ["--outdir", str(scan_dir)])

    points = json.loads((scan_dir / "eos_points.json").read_text())
    assert [pt["density_g_cm3"] for pt in points] == [0.40, 0.35, 0.30, 0.25]  # high -> low
    assert all(pt["pressure_GPa_stderr"] > 0 for pt in points)
    for pt in points:
        frame = read(scan_dir / f"rho{pt['density_g_cm3']:.4f}.extxyz")
        density = frame.get_masses().sum() / frame.get_volume() * 1.66053906660
        assert density == pytest.approx(pt["density_g_cm3"], rel=1e-9)
    assert (scan_dir / "eos_fit.png").is_file()

    fit = json.loads((scan_dir / "eos_fit.json").read_text())
    assert fit["density_range_g_cm3"] == [0.25, 0.40]
    rho_02 = fit["targets"]["0.2"]["density_g_cm3"]
    assert np.polyval(fit["coeffs"], rho_02) == pytest.approx(0.2, abs=1e-9)

    run_dir = tmp_path / "run"
    load("run_state_point").main([
        "--pressure-GPa", "0.2", "--n-molecules", "32", "--device", "cpu",
        "--eos-fit", str(scan_dir / "eos_fit.json"), "--outdir", str(run_dir),
        "--nvt-steps", "200", "--nve-steps", "200", "--thermo-interval", "10", "--print-interval", "100",
        "--traj-interval", "50", "--com-interval", "5", "--com-flush-interval", "100",
    ])
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "npt" not in summary
    assert summary["volume"]["density_g_cm3"] == pytest.approx(rho_02, rel=1e-9)
    assert summary["volume"]["source"].startswith("eos_fit:")


def test_eos_scan_rerun_skips_finished_densities(tmp_path, capsys):
    scan = load("eos_scan")
    scan.main(SCAN_ARGS + ["--outdir", str(tmp_path)])
    capsys.readouterr()
    scan.main(SCAN_ARGS + ["--outdir", str(tmp_path)])
    out = capsys.readouterr().out
    assert out.count("already done, skipping") == 4
    assert len(json.loads((tmp_path / "eos_points.json").read_text())) == 4
