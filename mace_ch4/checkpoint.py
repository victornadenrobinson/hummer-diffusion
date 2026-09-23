"""Checkpoint save/load and COM-trajectory persistence for resumable production MD runs."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_checkpoint(
    path: str | Path,
    *,
    positions: np.ndarray,
    velocities: np.ndarray,
    cell: np.ndarray,
    step: int,
    tracker_unwrapped: np.ndarray,
    tracker_last_wrapped: np.ndarray,
    tracker_cell: np.ndarray,
    initial_energy: float,
    done: bool = False,
) -> None:
    """Write a checkpoint atomically (write to temp file, then rename)."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.savez(
        tmp_path,
        positions=positions,
        velocities=velocities,
        cell=cell,
        step=np.array(step),
        tracker_unwrapped=tracker_unwrapped,
        tracker_last_wrapped=tracker_last_wrapped,
        tracker_cell=tracker_cell,
        initial_energy=np.array(initial_energy),
        done=np.array(done),
    )
    # np.savez appends .npz; account for that when renaming into place.
    Path(str(tmp_path) + ".npz").replace(path)


def load_checkpoint(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "positions": data["positions"],
            "velocities": data["velocities"],
            "cell": data["cell"],
            "step": int(data["step"]),
            "tracker_unwrapped": data["tracker_unwrapped"],
            "tracker_last_wrapped": data["tracker_last_wrapped"],
            "tracker_cell": data["tracker_cell"],
            "initial_energy": float(data["initial_energy"]),
            "done": bool(data["done"]),
        }


def checkpoint_exists(path: str | Path) -> bool:
    return Path(path).is_file()


def append_com_trajectory(
    path: str | Path, new_times_fs: np.ndarray, new_unwrapped_com: np.ndarray
) -> None:
    """Append a chunk of sampled (time, unwrapped COM) frames to a growing file.

    new_unwrapped_com: shape (n_new_frames, n_molecules, 3).
    """
    path = Path(path)
    if path.is_file():
        prev_times, prev_com = load_com_trajectory(path)
        times_fs = np.concatenate([prev_times, new_times_fs])
        unwrapped_com = np.concatenate([prev_com, new_unwrapped_com], axis=0)
    else:
        times_fs = np.array(new_times_fs)
        unwrapped_com = np.array(new_unwrapped_com)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.savez(tmp_path, times_fs=times_fs, unwrapped_com=unwrapped_com)
    Path(str(tmp_path) + ".npz").replace(path)


def load_com_trajectory(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return data["times_fs"], data["unwrapped_com"]
