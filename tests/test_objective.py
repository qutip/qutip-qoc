"""
Tests for Objective class.
"""
import numpy as np
import qutip as qt
from qutip_qoc.objective import Objective


def test_objective_init():
    initial = qt.basis(2, 0)
    target = qt.basis(2, 1)
    H = [qt.sigmax(), [qt.sigmay(), np.sin]]
    obj = Objective(initial, H, target)

    assert np.all(obj.initial == initial)
    assert obj.H == H
    assert np.all(obj.target == target)


def test_objective_getstate():
    initial = qt.basis(2, 0)
    target = qt.basis(2, 1)
    H = [qt.sigmax(), [qt.sigmay(), np.sin]]
    obj = Objective(initial, H, target)

    state = obj.__getstate__()

    assert np.all(state[0] == initial)
    assert state[1] == [H[0], H[1][0]]
    assert np.all(state[2] == target)
