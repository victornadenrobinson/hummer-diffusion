"""Benchmark MACE MD throughput and persist results for later plotting."""
from __future__ import annotations

import time
from pathlib import Path

from ase import units
from ase.atoms import Atoms
from ase.md.verlet import VelocityVerlet

from mace_ch4.records import append_record, load_records


def run_benchmark(
    atoms: Atoms,
    *,
    timestep_fs: float,
    n_warmup_steps: int,
    n_timed_steps: int,
) -> dict:
    """Time n_timed_steps of NVE dynamics after n_warmup_steps of untimed warmup.

    atoms must already have a calculator attached. Warmup excludes one-off
    costs (model JIT/compile, first neighbor-list build) from the timing.
    """
    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
    dyn.run(n_warmup_steps)

    start = time.perf_counter()
    dyn.run(n_timed_steps)
    elapsed_seconds = time.perf_counter() - start

    steps_per_second = n_timed_steps / elapsed_seconds
    ns_per_day = steps_per_second * timestep_fs * 1e-6 * 86400.0
    return {
        "n_atoms": len(atoms),
        "timestep_fs": timestep_fs,
        "n_warmup_steps": n_warmup_steps,
        "n_timed_steps": n_timed_steps,
        "elapsed_seconds": elapsed_seconds,
        "steps_per_second": steps_per_second,
        "ns_per_day": ns_per_day,
    }


def append_benchmark_result(path: str | Path, record: dict) -> None:
    """Append one benchmark record to a growing JSON list, written atomically."""
    append_record(path, record)


def load_benchmark_results(path: str | Path) -> list[dict]:
    return load_records(path)
