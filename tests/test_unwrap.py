import numpy as np
import pytest

from mace_ch4.unwrap import UnwrappedCOMTracker, minimum_image, molecule_centers_of_mass


def test_molecule_centers_of_mass_two_molecules():
    # Two 2-atom "molecules": atom masses 1 and 3, at known positions.
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],  # molecule 0: COM at x=3 (weighted 1:3)
            [10.0, 0.0, 0.0],
            [10.0, 4.0, 0.0],  # molecule 1: COM at y=3
        ]
    )
    masses = np.array([1.0, 3.0, 1.0, 3.0])
    com = molecule_centers_of_mass(positions, masses, atoms_per_molecule=2)
    np.testing.assert_allclose(com[0], [3.0, 0.0, 0.0])
    np.testing.assert_allclose(com[1], [10.0, 3.0, 0.0])


def test_molecule_centers_of_mass_rejects_bad_grouping():
    positions = np.zeros((5, 3))
    masses = np.ones(5)
    with pytest.raises(ValueError):
        molecule_centers_of_mass(positions, masses, atoms_per_molecule=2)


def test_minimum_image_wraps_to_shortest_vector():
    cell = np.eye(3) * 10.0
    # A displacement of 9 in a box of side 10 should wrap to -1.
    delta = np.array([[9.0, 0.0, 0.0]])
    wrapped = minimum_image(delta, cell)
    np.testing.assert_allclose(wrapped, [[-1.0, 0.0, 0.0]])


def test_unwrapped_tracker_accumulates_across_boundary_crossing():
    cell = np.eye(3) * 10.0
    tracker = UnwrappedCOMTracker(initial_com=np.array([[9.5, 0.0, 0.0]]), cell=cell)

    # Particle drifts +1 in x and wraps from 9.5 -> 0.5 (crossed the boundary).
    tracker.update(np.array([[0.5, 0.0, 0.0]]))
    np.testing.assert_allclose(tracker.unwrapped, [[10.5, 0.0, 0.0]])

    # Drifts another +1, wrapping again from 0.5 -> 1.5 is NOT a wrap (no jump).
    tracker.update(np.array([[1.5, 0.0, 0.0]]))
    np.testing.assert_allclose(tracker.unwrapped, [[11.5, 0.0, 0.0]])


def test_tracker_state_dict_roundtrip():
    cell = np.eye(3) * 10.0
    tracker = UnwrappedCOMTracker(initial_com=np.array([[1.0, 2.0, 3.0]]), cell=cell)
    tracker.update(np.array([[1.5, 2.5, 3.5]]))

    restored = UnwrappedCOMTracker.from_state_dict(tracker.state_dict())
    np.testing.assert_allclose(restored.unwrapped, tracker.unwrapped)

    # Continuing the trajectory from the restored tracker must match continuing the original.
    tracker.update(np.array([[1.6, 2.6, 3.6]]))
    restored.update(np.array([[1.6, 2.6, 3.6]]))
    np.testing.assert_allclose(restored.unwrapped, tracker.unwrapped)
