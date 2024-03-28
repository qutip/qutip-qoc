"""
Tests for GOAT and JOPT algorithms.
"""

import pytest
import qutip as qt
import qutip_jax  # noqa: F401
import numpy as np
import jax.numpy as jnp
import collections

from qutip_qoc.pulse_optim import optimize_pulses
from qutip_qoc.objective import Objective
from qutip_qoc.time import TimeInterval

Case = collections.namedtuple(
    "Case",
    [
        "objectives",
        "control_parameters",
        "time_interval",
        "algorithm_kwargs",
        "optimizer_kwargs",
    ],
)

# --------------------------- System and Control ---------------------------


def sin(t, p):
    return p[0] * np.sin(p[1] * t + p[2])


def grad_sin(t, p, idx):
    if idx == 0:
        return np.sin(p[1] * t + p[2])
    if idx == 1:
        return p[0] * np.cos(p[1] * t + p[2]) * t
    if idx == 2:
        return p[0] * np.cos(p[1] * t + p[2])
    if idx == 3:
        return p[0] * np.cos(p[1] * t + p[2]) * p[1]  # w.r.t. time


p_guess = q_guess = [1, 1, 0]
p_bounds = q_bounds = [(-10, 10), (-10, 10), (-np.pi, np.pi)]

H_d = [qt.sigmaz()]
H_c = [
    [qt.sigmax(), lambda t, p: sin(t, p), {"grad": grad_sin}],
    [qt.sigmay(), lambda t, q: sin(t, q), {"grad": grad_sin}],
]

H = H_d + H_c

# ------------------------------- Objective -------------------------------

# state to state transfer
initial = qt.basis(2, 0)
target = qt.basis(2, 1)

state2state = Case(
    objectives=[Objective(initial, H, target)],
    control_parameters={
        "p": {"guess": p_guess, "bounds": p_bounds},
        "q": {"guess": q_guess, "bounds": q_bounds},
    },
    time_interval=TimeInterval(evo_time=1.0),
    algorithm_kwargs={
        "alg": "GOAT",
        "fid_err_targ": 0.01,
    },
    optimizer_kwargs={"seed": 0},
)


# unitary gate synthesis
initial_U = qt.qeye(2)
target_U = qt.sigmaz()

unitary = Case(
    objectives=[Objective(initial_U, H, target_U)],
    control_parameters={
        "p": {"guess": p_guess, "bounds": p_bounds},
        "q": {"guess": q_guess, "bounds": q_bounds},
    },
    time_interval=TimeInterval(evo_time=1.0),
    algorithm_kwargs={
        "alg": "GOAT",
        "fid_err_targ": 0.01,
    },
    optimizer_kwargs={"seed": 0},
)


# unitary gate synthesis - time optimization
time = Case(
    objectives=[Objective(initial_U, H, target_U)],
    control_parameters={
        "p": {"guess": p_guess, "bounds": p_bounds},
        "q": {"guess": q_guess, "bounds": q_bounds},
        "__time__": {
            "guess": 5,
            "bounds": (0, 10),
        },
    },
    time_interval=TimeInterval(evo_time=1.0),
    algorithm_kwargs={
        "alg": "GOAT",
        "fid_err_targ": 0.01,
    },
    optimizer_kwargs={"seed": 0},
)


# map synthesis
initial_map = qt.sprepost(qt.qeye(2), qt.qeye(2).dag())
target_map = qt.sprepost(qt.sigmaz(), qt.sigmaz().dag())

L_d = [qt.liouvillian(qt.sigmaz(), c_ops=[qt.destroy(2)])]
L_c = [
    [qt.liouvillian(qt.sigmax()), lambda t, p: sin(t, p), {"grad": grad_sin}],
    [qt.liouvillian(qt.sigmay()), lambda t, q: sin(t, q), {"grad": grad_sin}],
]

L = L_d + L_c

mapping = Case(
    objectives=[Objective(initial_map, L, target_map)],
    control_parameters={
        "p": {"guess": p_guess, "bounds": p_bounds},
        "q": {"guess": q_guess, "bounds": q_bounds},
    },
    time_interval=TimeInterval(evo_time=1.0),
    algorithm_kwargs={
        "alg": "GOAT",
        "fid_err_targ": 0.1,  # relaxed objective
    },
    optimizer_kwargs={"seed": 0},
)


# ----------------------- System and JAX Control ---------------------


def sin_jax(t, p):
    return p[0] * jnp.sin(p[1] * t + p[2])


Hc_jax = [
    [qt.sigmax(), lambda t, p: sin_jax(t, p)],
    [qt.sigmay(), lambda t, q: sin_jax(t, q)],
]

H_jax = H_d + Hc_jax

# ------------------------------- Objective -------------------------------

# state to state transfer
state2state_jax = state2state._replace(
    objectives=[Objective(initial, H_jax, target)],
    algorithm_kwargs={"alg": "JOPT", "fid_err_targ": 0.01},
)

# unitary gate synthesis
unitary_jax = unitary._replace(
    objectives=[Objective(initial_U, H_jax, target_U)],
    algorithm_kwargs={"alg": "JOPT", "fid_err_targ": 0.01},
)

# unitary gate synthesis - time optimization
time_jax = time._replace(
    objectives=[Objective(initial_U, H_jax, target_U)],
    algorithm_kwargs={"alg": "JOPT", "fid_err_targ": 0.01},
)

# map synthesis
Lc_jax = [
    [qt.liouvillian(qt.sigmax()), lambda t, p: sin_jax(t, p)],
    [qt.liouvillian(qt.sigmay()), lambda t, q: sin_jax(t, q)],
]
L_jax = L_d + Lc_jax

mapping_jax = mapping._replace(
    objectives=[Objective(initial_map, L_jax, target_map)],
    algorithm_kwargs={"alg": "JOPT", "fid_err_targ": 0.1},  # relaxed objective
)


@pytest.fixture(
    params=[
        # GOAT
        pytest.param(state2state, id="State to state (GOAT)"),
        pytest.param(unitary, id="Unitary gate (GOAT)"),
        pytest.param(time, id="Time optimization (GOAT)"),
        pytest.param(mapping, id="Map synthesis (GOAT)"),
        # JOPT
        pytest.param(state2state_jax, id="State to state (JAX)"),
        pytest.param(unitary_jax, id="Unitary gate (JAX)"),
        pytest.param(time_jax, id="Time optimization (JAX)"),
        pytest.param(mapping_jax, id="Map synthesis (JAX)"),
    ]
)
def tst(request):
    return request.param


def test_optimize_pulses(tst):
    result = optimize_pulses(
        tst.objectives,
        tst.control_parameters,
        tst.time_interval,
        tst.algorithm_kwargs,
        tst.optimizer_kwargs,
    )
    assert result.infidelity <= tst.algorithm_kwargs.get("fid_err_targ", 0.01)
