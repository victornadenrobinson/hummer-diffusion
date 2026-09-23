import numpy as np
import pytest
from ase.calculators.lj import LennardJones
from ase.cluster.cubic import FaceCenteredCubic

from mace_ch4.benchmark import append_benchmark_result, load_benchmark_results, run_benchmark


def _tiny_atoms():
    # A small LJ argon-like cluster: fast, no network/MACE dependency needed
    # just to exercise the timing/throughput-math logic in run_benchmark.
    atoms = FaceCenteredCubic("Ar", [(1, 0, 0), (1, 1, 0), (1, 1, 1)], [2, 2, 2], 4.0)
    atoms.calc = LennardJones()
    return atoms


def test_run_benchmark_reports_positive_throughput():
    atoms = _tiny_atoms()
    result = run_benchmark(atoms, timestep_fs=1.0, n_warmup_steps=2, n_timed_steps=5)

    assert result["n_atoms"] == len(atoms)
    assert result["n_timed_steps"] == 5
    assert result["elapsed_seconds"] > 0
    assert result["steps_per_second"] > 0
    assert result["ns_per_day"] == pytest.approx(
        result["steps_per_second"] * 1.0 * 1e-6 * 86400.0
    )


def test_append_and_load_benchmark_results_roundtrip(tmp_path):
    path = tmp_path / "results.json"
    assert not path.is_file()

    append_benchmark_result(path, {"n_molecules": 64, "ns_per_day": 1.5})
    append_benchmark_result(path, {"n_molecules": 128, "ns_per_day": 0.8})

    records = load_benchmark_results(path)
    assert records == [
        {"n_molecules": 64, "ns_per_day": 1.5},
        {"n_molecules": 128, "ns_per_day": 0.8},
    ]


def test_append_benchmark_result_no_leftover_tmp_file(tmp_path):
    path = tmp_path / "results.json"
    append_benchmark_result(path, {"a": 1})
    append_benchmark_result(path, {"a": 2})
    leftovers = list(tmp_path.glob("*.tmp*"))
    assert leftovers == []
