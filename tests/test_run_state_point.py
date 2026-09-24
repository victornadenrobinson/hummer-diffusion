"""End-to-end run of scripts/run_state_point.py with a toy calculator standing in for MACE."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from toy_ch4_calculator import ToyCH4Calculator

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_state_point.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_state_point", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.get_calculator = lambda **kwargs: ToyCH4Calculator()
    return module


ARGS = [
    "--pressure-GPa", "0.2", "--n-molecules", "32", "--device", "cpu",
    "--npt-steps", "200", "--nvt-steps", "400", "--nve-steps", "400",
    "--thermo-interval", "10", "--print-interval", "100", "--traj-interval", "50",
    "--com-interval", "5", "--com-flush-interval", "100",
]


def test_full_pipeline_writes_expected_outputs(tmp_path):
    module = load_script()
    module.main(ARGS + ["--outdir", str(tmp_path)])

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert {"config", "npt", "volume", "nvt", "nve", "diffusion"} <= summary.keys()
    assert summary["volume"]["reference_density_g_cm3"] == pytest.approx(0.34586, abs=1e-4)

    # NPT/NVT: first and last frame only; NVE: every traj-interval steps incl. step 0.
    assert len(read(tmp_path / "npt.extxyz", index=":")) == 2
    assert len(read(tmp_path / "nvt.extxyz", index=":")) == 2
    nve_frames = read(tmp_path / "nve.extxyz", index=":")
    assert [f.info["step"] for f in nve_frames] == list(range(0, 401, 50))
    assert nve_frames[0].get_momenta().any()

    with np.load(tmp_path / "nve_com.npz") as data:
        assert data["unwrapped_com"].shape == (81, 32, 3)
        np.testing.assert_allclose(data["times_fs"], np.arange(0, 401, 5) * 0.5)

    # NVE starts from the volume NPT averaged to, and conserves energy.
    last_nvt = read(tmp_path / "nvt.extxyz", index=-1)
    assert abs(last_nvt.get_volume() - summary["volume"]["volume_A3"]) < 1e-6
    assert summary["nve"]["max_abs_dE_per_mol_meV"] < 5.0

    rows = [ln for ln in (tmp_path / "thermo_nve.log").read_text().splitlines() if not ln.startswith("#")]
    assert len(rows) == 41
    assert len(rows[0].split()) == 12


def test_rerun_skips_finished_stages(tmp_path, capsys):
    module = load_script()
    module.main(ARGS + ["--outdir", str(tmp_path)])
    before = (tmp_path / "nve.extxyz").stat().st_mtime_ns
    capsys.readouterr()

    module.main(ARGS + ["--outdir", str(tmp_path)])
    out = capsys.readouterr().out
    for stage in ("volume", "nvt", "nve"):
        assert f"[{stage}] already done" in out
    assert (tmp_path / "nve.extxyz").stat().st_mtime_ns == before
