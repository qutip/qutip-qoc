import pytest
import qutip as qt

from qutip_qoc.optimize import optimize_pulses
from qutip_qoc.objective import Objective
from qutip_qoc.time_interval import TimeInterval
from qutip_qoc.result import Result

# ------------------------------- Control H ------------------------------- #


def f(t, p): return p * t


def grad_f(t, p, idx):
    if idx == 0:
        return t
    if idx == 1:
        return p


p_guess = 0.
p_bounds = [-1., 1.]

H_x = [qt.sigmax(), f, {"grad": grad_f}]

# ------------------------------- Objective ------------------------------- #
initial = qt.base(2, 0)
target = qt.base(2, 1)

t_interval = TimeInterval(evo_time=1.)
fid_err_targ = 0.01


@pytest.mark.parametrize("objectives, pulse_options, time_interval, time_options, algorithm_kwargs, optimizer_kwargs, minimizer_kwargs, integrator_kwargs, expected", [
    # Add your test cases here. For example:
    ([Objective(initial, [H_x], target)],
     {"p": {"guess": p_guess, "bounds": p_bounds}},
     t_interval, {}, {"method": "GOAT"}, {}, {}, {}, {}, {}, Result),
    # ([objectives1], {pulse_options1}, time_interval1, {time_options1}, {algorithm_kwargs1}, {optimizer_kwargs1}, {minimizer_kwargs1}, {integrator_kwargs1}, ExpectedResultType1),
    # ([objectives2], {pulse_options2}, time_interval2, {time_options2}, {algorithm_kwargs2}, {optimizer_kwargs2}, {minimizer_kwargs2}, {integrator_kwargs2}, ExpectedResultType2),
])
def test_optimize_pulses(
        objectives,
        pulse_options,
        time_interval,
        time_options,
        algorithm_kwargs,
        optimizer_kwargs,
        minimizer_kwargs,
        integrator_kwargs,
        expected):
    result = optimize_pulses(
        objectives,
        pulse_options,
        time_interval,
        time_options,
        algorithm_kwargs,
        optimizer_kwargs,
        minimizer_kwargs,
        integrator_kwargs)
    assert isinstance(result, expected)
