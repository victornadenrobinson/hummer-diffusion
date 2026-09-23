import numpy as np
import pytest

from mace_ch4.checkpoint import (
    append_com_trajectory,
    checkpoint_exists,
    load_checkpoint,
    load_com_trajectory,
    save_checkpoint,
)


def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "state.npz"
    positions = np.arange(30.0).reshape(10, 3)
    velocities = -np.arange(30.0).reshape(10, 3)
    cell = np.eye(3) * 20.0
    tracker_unwrapped = np.arange(6.0).reshape(2, 3)
    tracker_last_wrapped = tracker_unwrapped.copy()

    assert not checkpoint_exists(path)
    save_checkpoint(
        path,
        positions=positions,
        velocities=velocities,
        cell=cell,
        step=42,
        tracker_unwrapped=tracker_unwrapped,
        tracker_last_wrapped=tracker_last_wrapped,
        tracker_cell=cell,
        initial_energy=-123.456,
        done=False,
    )
    assert checkpoint_exists(path)

    restored = load_checkpoint(path)
    np.testing.assert_allclose(restored["positions"], positions)
    np.testing.assert_allclose(restored["velocities"], velocities)
    np.testing.assert_allclose(restored["cell"], cell)
    assert restored["step"] == 42
    np.testing.assert_allclose(restored["tracker_unwrapped"], tracker_unwrapped)
    np.testing.assert_allclose(restored["tracker_last_wrapped"], tracker_last_wrapped)
    np.testing.assert_allclose(restored["tracker_cell"], cell)
    assert restored["initial_energy"] == pytest.approx(-123.456)
    assert restored["done"] is False


def test_checkpoint_no_leftover_tmp_file(tmp_path):
    path = tmp_path / "state.npz"
    save_checkpoint(
        path,
        positions=np.zeros((1, 3)),
        velocities=np.zeros((1, 3)),
        cell=np.eye(3),
        step=0,
        tracker_unwrapped=np.zeros((1, 3)),
        tracker_last_wrapped=np.zeros((1, 3)),
        tracker_cell=np.eye(3),
        initial_energy=0.0,
    )
    leftovers = list(tmp_path.glob("*.tmp*"))
    assert leftovers == []


def test_load_checkpoint_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist.npz")


def test_append_com_trajectory_creates_then_appends(tmp_path):
    path = tmp_path / "com_traj.npz"
    n_molecules = 3

    chunk1_times = np.array([0.0, 1.0, 2.0])
    chunk1_com = np.arange(3 * n_molecules * 3, dtype=float).reshape(3, n_molecules, 3)
    append_com_trajectory(path, chunk1_times, chunk1_com)

    times, com = load_com_trajectory(path)
    np.testing.assert_allclose(times, chunk1_times)
    np.testing.assert_allclose(com, chunk1_com)

    chunk2_times = np.array([3.0, 4.0])
    chunk2_com = np.ones((2, n_molecules, 3))
    append_com_trajectory(path, chunk2_times, chunk2_com)

    times, com = load_com_trajectory(path)
    np.testing.assert_allclose(times, np.concatenate([chunk1_times, chunk2_times]))
    np.testing.assert_allclose(com, np.concatenate([chunk1_com, chunk2_com], axis=0))


def test_append_com_trajectory_no_leftover_tmp_file(tmp_path):
    path = tmp_path / "com_traj.npz"
    append_com_trajectory(path, np.array([0.0]), np.zeros((1, 2, 3)))
    append_com_trajectory(path, np.array([1.0]), np.zeros((1, 2, 3)))
    leftovers = list(tmp_path.glob("*.tmp*"))
    assert leftovers == []
