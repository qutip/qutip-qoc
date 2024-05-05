"""
This module contains the Objective class for storing information about an
optimization objective, and the _MultiObjective class for optimizing multiple
objectives simultaneously.
"""

import numpy as np
import qutip as qt

from qutip_qoc._jopt import _JOPT
from qutip_qoc._goat import _GOAT
from qutip_qoc._crab import _CRAB
from qutip_qoc._grape import _GRAPE

__all__ = ["Objective"]


class Objective:
    """
    A class for storing information about an optimization objective.

    Examples
    --------
    To specify an objective for GOAT, provide a gradient function::

        initial = qt.basis(2, 0)
        target  = qt.basis(2, 1)

        sin = lambda t, p: np.sin(p * t)

        def d_sin(t, p, idx):
            if idx==0: return t * np.cos(t) # w.r.t. p
            if idx==1: return p * np.cos(t) # w.r.t. t

        H = [qt.sigmax(), [qt.sigmay(), sin, {'grad': d_sin}]]

        obj = Objective(initial, H, target)


    To specify an objective for JOPT, no need for gradient function::

        initial = qt.basis(2, 0)
        target  = qt.basis(2, 1)

        @jax.jit
        def sin(t, p)
            return jax.numpy.sin(p * t)

        def d_sin(t, p, idx):
            if idx==0: return t * np.cos(t) # w.r.t. p
            if idx==1: return p * np.cos(t) # w.r.t. t

        H = [qt.sigmax(), [qt.sigmay(), sin]]

        obj = Objective(initial, H, target)

    Attributes
    ----------
    initial : :class:`qutip.Qobj`
        The initial state or operator to be transformed.

    H : callable, list
        A specification of the time-depedent quantum object.
        Usually a list with drift Hamiltonian at zero-index
        followed by the control Hamiltonians.
        See :class:`qutip.QobjEvo` for details and examples.

    target : :class:`qutip.Qobj`
        The target state or operator.

    weight : float
        Only used in multi-objective scenarios.
        The weight of this objective in the optimization.
        All weights are normalized to sum to 1.
    """

    def __init__(self, initial, H, target, weight=1):
        self.initial = initial
        self.H = H
        self.target = target
        self.weight = weight

    def __getstate__(self):
        """
        Extract picklable information from the objective.
        Callable functions will be lost.
        """
        only_H = [self.H[0]] + [H[0] for H in self.H[1:]]
        return (self.initial, only_H, self.target)


class _MultiObjective:
    """
    Composite class for multiple GOAT, CRAB, GRAP, JOPT instances
    to optimize multiple objectives simultaneously. Each instance
    is associated with one objective.
    """

    def __init__(
        self,
        objectives,
        time_interval,
        alg_kwargs,
        guess_params,
        qtrl_optimizers=None,
        time_options=None,
        control_parameters=None,
        **integrator_kwargs,
    ):
        alg = alg_kwargs.get("alg")

        # normalized weights
        weights = [obj.weight for obj in objectives]
        self._weights = np.array(weights) / np.sum(weights)

        if alg == "GOAT" or alg == "JOPT":
            kwargs = {
                "time_interval": time_interval,
                "time_options": time_options,
                "control_parameters": control_parameters,
                "alg_kwargs": alg_kwargs,
                "guess_params": guess_params,
                **integrator_kwargs,
            }
            if alg == "GOAT":
                self._alg_list = [_GOAT(objective=obj, **kwargs) for obj in objectives]
            elif alg == "JOPT":
                with qt.CoreOptions(default_dtype="jax"):
                    self._alg_list = [
                        _JOPT(objective=obj, **kwargs) for obj in objectives
                    ]
        elif alg == "CRAB":
            self._alg_list = [_CRAB(optimizer) for optimizer in qtrl_optimizers]
        elif alg == "GRAPE":
            self._alg_list = [_GRAPE(optimizer) for optimizer in qtrl_optimizers]

    def goal_fun(self, params):
        """
        Calculates the weighted infidelity over all objectives
        """
        infid = 0
        for i, alg in enumerate(self._alg_list):
            infid += self._weights[i] * alg.infidelity(params)
        return infid

    def grad_fun(self, params):
        """
        Calculates the weighted sum of gradients over all objectives
        """
        grads = 0
        for i, alg in enumerate(self._alg_list):
            grads += self._weights[i] * alg.gradient(params)
        return grads
