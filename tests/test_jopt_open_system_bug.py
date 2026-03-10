import numpy as np
import qutip as qt
from qutip_qoc import Objective, optimize_pulses

from jax import jit, numpy

def test_open_system_jopt_runs_without_error():
    Hd = qt.Qobj(np.diag([1, 2]))
    c_ops = [np.sqrt(0.1) * qt.sigmam()]
    Hc = qt.sigmax()

    Ld = qt.liouvillian(H=Hd, c_ops=c_ops)
    Lc = qt.liouvillian(Hc)

    initial_state = qt.fock_dm(2, 0)
    target_state = qt.fock_dm(2, 1)

    times = np.linspace(0, 2 * np.pi, 250)

    @jit
    def sin_x(t, c, **kwargs):
        return c[0] * numpy.sin(c[1] * t)
    L = [Ld, [Lc, sin_x]]

    guess_params = [1, 0.5]

    res_jopt = optimize_pulses(
        objectives = Objective(initial_state, L, target_state),
        control_parameters = {
            "ctrl_x": {"guess": guess_params, "bounds": [(-1, 1), (0, 2 * np.pi)]}
        },
        tlist = times,
        algorithm_kwargs = {
            "alg": "JOPT",
            "fid_err_targ": 0.001,
        },
    )

    assert res_jopt.infidelity < 0.25, f"Fidelity error too high: {res_jopt.infidelity}"