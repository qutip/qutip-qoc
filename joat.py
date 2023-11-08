"""
This module contains functions that implement the JOAT algorithm to
calculate optimal parameters for analytical control pulse sequences.
"""
import jax
import jax.numpy as jnp

import qutip as qt
from qutip import Qobj, QobjEvo
import qutip_jax

from diffrax import Dopri5, Tsit5, PIDController


class JOAT:
    """
    Class for storing a control problem and calculating
    the fidelity error function and its gradient wrt the control parameters.
    """
    # calculated during optimization
    X = None  # current evolution operator
    infid = None  # projective SU distance (infidelity)

    def __init__(self, objective, time_interval, time_options, pulse_options, alg_kwargs, guess_params, **integrator_kwargs):

        self.Hd = objective.H_evo[0]
        self.Hc_lst = objective.H_evo[1:]

        self.pulse_options = pulse_options
        self.guess_params = guess_params
        self.H = self.prepare_H()

        self.initial = objective.initial.to("jaxdia")
        self.target = objective.target.to("jaxdia")

        self.evo_time = time_interval.evo_time
        self.var_t = True if time_options.get("guess", False) else False

        # inferred attributes
        self.norm_fac = 1 / self.target.norm()

        # integrator options
        self.integrator_kwargs = integrator_kwargs
        self.integrator_kwargs["method"] = "diffrax"

        self.rtol = self.integrator_kwargs.get("rtol", 1e-5)
        self.atol = self.integrator_kwargs.get("atol", 1e-5)

        self.integrator_kwargs.setdefault(
            "stepsize_controller", self.integrator_kwargs.get(
                "stepsize_controller", PIDController(
                    rtol=self.rtol, atol=self.atol
                )
            )
        )
        self.integrator_kwargs.setdefault(
            "solver", self.integrator_kwargs.get(
                "solver", Tsit5()
            )
        )

        if self.Hd.issuper:
            self.fid_type = alg_kwargs.get("fid_type", "TRACEDIFF")

            self.solver = qt.MESolver(
                H=self.H,
                options=self.integrator_kwargs
            )

        else:
            self.fid_type = alg_kwargs.get("fid_type", "PSU")

            self.solver = qt.SESolver(
                H=self.H,
                options=self.integrator_kwargs
            )

        self.gradient = jax.grad(self.infidelity)

    def prepare_H(self):

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

        return H.to("jaxdia")

    def infidelity(self, params):
        """
        projective SU distance (infidelity) to be minimized
        store intermediate results for gradient calculation
        returns the infidelity, the normalized overlap,
        the current unitary and its gradient for later use
        """
        # adjust integration time-interval, if time is parameter
        evo_time = self.evo_time if self.var_t == False else params[-1]

        X = self.solver.run(
            self.initial, [0., evo_time],
            args={'p': params}
        ).final_state

        self.X = Qobj(X, dims=self.target.dims)

        if self.fid_type == "TRACEDIFF":
            diff = self.X - self.target
            g = 1/2 * (diff.dag() * diff).tr()
            self.infid = jnp.real(self.norm_fac * g)
        else:
            g = self.norm_fac * X.overlap(self.target)
            if self.fid_type == "PSU":  # f_PSU (drop global phase)
                self.infid = 1 - jnp.abs(g)
            elif self.fid_type == "SU":  # f_SU (incl global phase)
                self.infid = 1 - jnp.real(g)

        return self.infid


class Multi_JOAT:
    """
    Composite class for multiple JOAT instances
    to optimize multiple objectives simultaneously
    """

    def __init__(self, objectives, time_interval, time_options, pulse_options, alg_kwargs, guess_params, **integrator_kwargs):
        self.joats = [JOAT(obj, time_interval, time_options, pulse_options, alg_kwargs, guess_params, ** integrator_kwargs)
                      for obj in objectives]

        self.mean_infid = None

    def goal_fun(self, params):
        infid_sum = 0
        for joat in self.joats:  # TODO: parallelize
            infid = joat.infidelity(params)
            if infid < 0:
                print(
                    "WARNING: infidelity < 0 -> inaccurate integration, "
                    "try reducing integrator tolerance (atol, rtol)"
                )
            infid_sum += infid
        self.mean_infid = jnp.mean(infid_sum)
        return self.mean_infid

    def grad_fun(self, params):
        grads = 0
        for g in self.joats:
            grads += g.gradient(params)
        return grads
