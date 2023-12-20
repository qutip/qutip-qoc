"""
This module contains functions that implement the JOAT algorithm to
calculate optimal parameters for analytical control pulse sequences.
"""
import qutip as qt
from qutip import Qobj, QobjEvo
import qutip_jax

from diffrax import Dopri5, PIDController

import jax
from jax import custom_jvp
import jax.numpy as jnp

__all__ = ["JOAT", "Multi_JOAT"]


@custom_jvp
def abs(x):
    return jnp.abs(x)


def abs_jvp(primals, tangents):
    """
    Custom jvp for absolute value of complex functions
    """
    x, = primals
    t, = tangents

    abs_x = abs(x)
    res = jnp.where(abs_x == 0, 0., # prevent division by zero
                    jnp.real(jnp.multiply(jnp.conj(x), t)) / abs_x)

    return abs_x, res


# register custom jvp for absolut value of complex functions
abs.defjvp(abs_jvp)


class JOAT:
    """
    Class for storing a control problem and calculating
    the fidelity error function and its gradient wrt the control parameters.
    """

    def __init__(
            self,
            objective,
            time_interval,
            time_options,
            pulse_options,
            alg_kwargs,
            guess_params,
            **integrator_kwargs):

        self.Hd = objective.H[0]
        self.Hc_lst = objective.H[1:]

        self.pulse_options = pulse_options
        self.guess_params = guess_params
        self.H = self.prepare_H()

        self.initial = objective.initial.to("jax")
        self.target = objective.target.to("jax")

        self.evo_time = time_interval.evo_time
        self.var_t = "guess" in time_options

        # inferred attributes
        self.norm_fac = 1 / self.target.norm()

        # integrator options
        self.integrator_kwargs = integrator_kwargs
        self.integrator_kwargs["method"] = "diffrax"

        self.rtol = self.integrator_kwargs.get("rtol", 1e-5)
        self.atol = self.integrator_kwargs.get("atol", 1e-5)

        self.integrator_kwargs.setdefault(
            "stepsize_controller", PIDController(
                rtol=self.rtol, atol=self.atol
            )
        )
        self.integrator_kwargs.setdefault(
            "solver", Dopri5()
        )

        # choose solver and fidelity type according to problem
        if self.Hd.issuper:
            self.fid_type = alg_kwargs.get("fid_type", "TRACEDIFF")
            self.solver = qt.MESolver(H=self.H, options=self.integrator_kwargs)

        else:
            self.fid_type = alg_kwargs.get("fid_type", "PSU")
            self.solver = qt.SESolver(H=self.H, options=self.integrator_kwargs)

        self.infidelity = jax.jit(self.infid)
        self.gradient = jax.jit(jax.grad(self.infidelity))

    def prepare_H(self):
        """
        prepare Hamiltonian call signature
        to only take one parameter vector 'p' for mesolve like:
        qt.mesolve(H, psi0, tlist, args={'p': p})
        """

        def helper(control, lower, upper):
            # to fix parameter index in loop
            return jax.jit(lambda t, p: control(t, p[lower:upper]))

        H = QobjEvo(self.Hd)
        idx = 0

        for Hc, p_opt in zip(self.Hc_lst, self.pulse_options.values()):
            hc, ctrl = Hc[0], Hc[1]

            guess = p_opt.get("guess")
            M = len(guess)

            evo = QobjEvo(
                [hc, helper(ctrl, idx, idx + M)],
                args={"p": self.guess_params}
            )
            H += evo
            idx += M

        return H.to("jax")

    def infid(self, params):
        """
        calculate infidelity to be minimized
        """
        # adjust integration time-interval, if time is parameter
        evo_time = self.evo_time if self.var_t == False else params[-1]

        X = self.solver.run(
            self.initial, [0., evo_time],
            args={'p': params}
        ).final_state

        if self.fid_type == "TRACEDIFF":
            diff = X - self.target
            # to prevent if/else in qobj.dag() and qobj.tr()
            diff_dag = Qobj(diff.data.adjoint(), dims=diff.dims)
            g = 1 / 2 * (diff_dag * diff).data.trace()
            infid = jnp.real(self.norm_fac * g)
        else:
            g = self.norm_fac * self.target.overlap(X)
            if self.fid_type == "PSU":  # f_PSU (drop global phase)
                infid = 1 - abs(g)  # custom_jvp for abs
            elif self.fid_type == "SU":  # f_SU (incl global phase)
                infid = 1 - jnp.real(g)

        return infid


class Multi_JOAT:
    """
    Composite class for multiple JOAT instances
    to optimize multiple objectives simultaneously
    """

    def __init__(self, objectives, time_interval, time_options, pulse_options,
                 alg_kwargs, guess_params, **integrator_kwargs):

        self.joats = [
            JOAT(obj, time_interval, time_options, pulse_options,
                 alg_kwargs, guess_params, **integrator_kwargs)
            for obj in objectives
        ]

        self.mean_infid = None

    def goal_fun(self, params):
        """
        Calculates the mean infidelity over all objectives
        """
        infid_sum = 0

        for j in self.joats:  # TODO: parallelize
            infid = j.infidelity(params)
            infid_sum += infid

        self.mean_infid = jnp.mean(infid_sum)
        return self.mean_infid

    def grad_fun(self, params):
        """
        Calculates the sum of gradients over all objectives
        """
        grads = 0

        for j in self.joats:
            grad = j.gradient(params)
            grads += grad

        return grads
