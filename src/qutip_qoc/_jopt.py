"""
This module contains functions that implement the JOPT algorithm to
calculate optimal parameters for analytical control pulse sequences.
"""
import qutip as qt
from qutip import Qobj, QobjEvo
from qutip_qoc.fidcomp import FidelityComputer

try:
    import jax
    import jax.numpy as jnp
    import qutip_jax  # noqa: F401
    import jaxlib  # noqa: F401
    from diffrax import Dopri5, PIDController
    _jax_available = True
except ImportError:
    _jax_available = False


class _JOPT:
    """
    Class for storing a control problem and calculating
    the fidelity error function and its gradient wrt the control parameters.
    Uses FidelityComputer for fidelity calculations with JAX optimization.
    """

    def __init__(
        self,
        objective,
        time_interval,
        time_options,
        control_parameters,
        alg_kwargs,
        guess_params,
        **integrator_kwargs,
    ):
        if not _jax_available:
            raise ImportError("The JOPT algorithm requires the modules jax, "
                           "jaxlib, and qutip_jax to be installed.")

        # Initialize FidelityComputer
        self._fid_type = alg_kwargs.get("fid_type", "PSU")
        self._fidcomp = FidelityComputer(self._fid_type)

        self._Hd = objective.H[0]
        self._Hc_lst = objective.H[1:]
        self._control_parameters = control_parameters
        self._guess_params = guess_params
        self._H = self._prepare_generator()

        # Convert to JAX format
        self._initial = objective.initial.to("jax")
        self._target = objective.target.to("jax")

        self._evo_time = time_interval.evo_time
        self._var_t = "guess" in time_options

        # integrator options
        self._integrator_kwargs = integrator_kwargs
        self._integrator_kwargs["method"] = "diffrax"

        self._rtol = self._integrator_kwargs.get("rtol", 1e-5)
        self._atol = self._integrator_kwargs.get("atol", 1e-5)

        self._integrator_kwargs.setdefault(
            "stepsize_controller", PIDController(rtol=self._rtol, atol=self._atol)
        )
        self._integrator_kwargs.setdefault("solver", Dopri5())

        # choose solver according to problem
        if self._Hd.issuper:
            self._solver = qt.MESolver(H=self._H, options=self._integrator_kwargs)
        else:
            self._solver = qt.SESolver(H=self._H, options=self._integrator_kwargs)

        # JIT-compiled functions
        self.infidelity = jax.jit(self._infid)
        self.gradient = jax.jit(jax.grad(self._infid))

    def _prepare_generator(self):
        """Prepare Hamiltonian call signature for JAX optimization"""
        def helper(control, lower, upper):
            return jax.jit(lambda t, p: control(t, p[lower:upper]))

        H = QobjEvo(self._Hd)
        idx = 0

        for Hc, p_opt in zip(self._Hc_lst, self._control_parameters.values()):
            hc, ctrl = Hc[0], Hc[1]
            guess = p_opt.get("guess")
            M = len(guess)

            evo = QobjEvo(
                [hc, helper(ctrl, idx, idx + M)], args={"p": self._guess_params}
            )
            H += evo
            idx += M

        return H.to("jax")

    def _infid(self, params):
        """Calculate infidelity using FidelityComputer with JAX support"""
        # Adjust integration time-interval if time is parameter
        evo_time = self._evo_time if not self._var_t else params[-1]

        # Run the solver
        evolved = self._solver.run(
            self._initial, [0.0, evo_time], args={"p": params}
        ).final_state

        # Handle conversion of evolved state
        if hasattr(evolved, 'to_array'):  # JAX array-backed Qobj
            evolved_qobj = qt.Qobj(evolved.to_array(), dims=self._target.dims)
        else:  # Regular Qobj
            evolved_qobj = evolved

        # Handle conversion of target state
        if hasattr(self._target, 'to_array'):  # JAX array-backed Qobj
            target_qobj = qt.Qobj(self._target.to_array(), dims=self._target.dims)
        else:  # Regular Qobj
            target_qobj = self._target

        # Handle conversion of initial state
        if hasattr(self._initial, 'to_array'):  # JAX array-backed Qobj
            initial_qobj = qt.Qobj(self._initial.to_array(), dims=self._initial.dims)
        else:  # Regular Qobj
            initial_qobj = self._initial

        # Calculate infidelity
        infid = self._fidcomp.compute_infidelity(initial_qobj, target_qobj, evolved_qobj)
        
        return jnp.array(infid, dtype=jnp.float64)