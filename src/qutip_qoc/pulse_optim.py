"""
This module is the entry point for the optimization of control pulses.
It provides the function `optimize_pulses` which prepares and runs the
GOAT, JOPT, GRAPE, CRAB or RL optimization.
"""
import numpy as np
from qutip_qoc.fidcomp import FidelityComputer  # Added import
from qutip_qoc._optimizer import _global_local_optimization
from qutip_qoc._time import _TimeInterval
import qutip as qt

try:
    from qutip_qoc._rl import _RL
    _rl_available = True
except ImportError:
    _rl_available = False

__all__ = ["optimize_pulses"]


def optimize_pulses(
    objectives,
    control_parameters,
    tlist,
    algorithm_kwargs=None,
    optimizer_kwargs=None,
    minimizer_kwargs=None,
    integrator_kwargs=None,
    optimization_type=None,
):
    """
    Run GOAT, JOPT, GRAPE, CRAB or RL optimization.

    Parameters
    ----------
    objectives : list of :class:`qutip_qoc.Objective`
        List of objectives to be optimized.
        Each objective is weighted by its weight attribute.

    control_parameters : dict
        Dictionary of options for the control pulse optimization.
        The keys of this dict must be a unique string identifier for each control Hamiltonian / function.
        For the GOAT and JOPT algorithms, the dict may optionally also contain the key "__time__".
        For each control function it must specify:

            control_id : dict
                - guess: ndarray, shape (n,)
                    For RL you don't need to specify the guess.
                    Initial guess. Array of real elements of size (n,),
                    where ``n`` is the number of independent variables.

                - bounds : sequence, optional
                    Sequence of ``(min, max)`` pairs for each element in
                    `guess`. None is used to specify no bound.

            __time__ : dict, optional
                Only supported by GOAT, JOPT (for RL use `algorithm_kwargs: 'shorter_pulses'`).
                If given the pulse duration is treated as optimization parameter.
                It must specify both:

                    - guess: ndarray, shape (n,)
                        Initial guess. Array of real elements of size (n,),
                        where ``n`` is the number of independent variables.

                    - bounds : sequence, optional
                        Sequence of ``(min, max)`` pairs for each element in `guess`.
                        None is used to specify no bound.

        GRAPE and CRAB bounds are only one pair of ``(min, max)`` limiting the amplitude of all tslots equally.

    tlist: List.
        Time over which system evolves.

    algorithm_kwargs : dict, optional
        Dictionary of options for the optimization algorithm.

            - alg : str
                Algorithm to use for the optimization.
                Supported are: "GRAPE", "CRAB", "GOAT", "JOPT" and "RL".

            - fid_err_targ : float, optional
                Fidelity error target for the optimization.

            - max_iter : int, optional
                Maximum number of iterations to perform.
                Referes to local minimizer steps or in the context of
                `alg: "RL"` to the max. number of episodes.
                Global steps default to 0 (no global optimization).
                Can be overridden by specifying in minimizer_kwargs.

        Algorithm specific keywords for GRAPE,CRAB can be found in
        :func:`qutip_qtrl.pulseoptim.optimize_pulse`.

    optimizer_kwargs : dict, optional
        Dictionary of options for the global optimizer.
        Only supported by GOAT and JOPT.

            - method : str, optional
                Algorithm to use for the global optimization.
                Supported are: "basinhopping", "dual_annealing"

            - max_iter : int, optional
                Maximum number of iterations to perform.
                Default is 0 (no global optimization).

        Full list of options can be found in
        :func:`scipy.optimize.basinhopping`
        and :func:`scipy.optimize.dual_annealing`.

    minimizer_kwargs : dict, optional
        Dictionary of options for the local minimizer.

            - method : str, optional
                Algorithm to use for the local optimization.
                Gradient driven methods are supported.

        Full list of options and methods can be found in
        :func:`scipy.optimize.minimize`.

    integrator_kwargs : dict, optional
        Dictionary of options for the integrator.
        Only supported by GOAT and JOPT.
        Options for the solver, see :obj:`MESolver.options` and
        `Integrator <./classes.html#classes-ode>`_ for a list of all options.

    optimization_type : str, optional
        Type of optimization. By default, QuTiP-QOC will try to automatically determine 
        whether this is a *state transfer* or a *gate synthesis* problem. Set this 
        flag to ``"state_transfer"`` or ``"gate_synthesis"`` to set the mode manually.

    Returns
    -------
    result : :class:`qutip_qoc.Result`
        Optimization result.
    """
    algorithm_kwargs = algorithm_kwargs or {}
    optimizer_kwargs = optimizer_kwargs or {}
    minimizer_kwargs = minimizer_kwargs or {}
    integrator_kwargs = integrator_kwargs or {}

    # Set default fidelity type if not specified
    algorithm_kwargs.setdefault("fid_type", "PSU")

    # create time interval
    time_interval = _TimeInterval(tslots=tlist)

    time_options = control_parameters.get("__time__", {})
    if time_options:  # convert to list of bounds if not already
        if not isinstance(time_options["bounds"][0], (list, tuple)):
            time_options["bounds"] = [time_options["bounds"]]

    alg = algorithm_kwargs.get("alg", "GRAPE")  # works with most input types

    Hd_lst, Hc_lst = [], []
    # Prepare objectives and extract Hamiltonians
    if not isinstance(objectives, list):
        objectives = [objectives]
    
    # Convert states based on optimization type
    for objective in objectives:
        H_list = objective.H if isinstance(objective.H, list) else [objective.H]
        if any(qt.issuper(H_i) for H_i in H_list):
            if isinstance(optimization_type, str) and optimization_type.lower() == "state_transfer":
                if qt.isket(objective.initial):
                    objective.initial = qt.operator_to_vector(qt.ket2dm(objective.initial))
                elif qt.isoper(objective.initial):
                    objective.initial = qt.operator_to_vector(objective.initial)
                if qt.isket(objective.target):
                    objective.target = qt.operator_to_vector(qt.ket2dm(objective.target))
                elif qt.isoper(objective.target):
                    objective.target = qt.operator_to_vector(objective.target)
            elif isinstance(optimization_type, str) and optimization_type.lower() == "gate_synthesis":
                objective.initial = qt.to_super(objective.initial)
                objective.target = qt.to_super(objective.target)
            elif optimization_type is None:
                if qt.isoper(objective.initial) and qt.isoper(objective.target):
                    if np.isclose((objective.initial).tr(), 1) and np.isclose((objective.target).tr(), 1):
                        objective.initial = qt.operator_to_vector(objective.initial)
                        objective.target = qt.operator_to_vector(objective.target)
                    else:
                        objective.initial = qt.to_super(objective.initial)
                        objective.target = qt.to_super(objective.target)
                if qt.isket(objective.initial):
                    objective.initial = qt.operator_to_vector(qt.ket2dm(objective.initial))
                if qt.isket(objective.target):
                    objective.target = qt.operator_to_vector(qt.ket2dm(objective.target))

    # extract guess and bounds for the control pulses
        x0, bounds = [], []
        for key in control_parameters.keys():
            if key != "__time__":
                x0.append(control_parameters[key].get("guess"))
                bounds.append(control_parameters[key].get("bounds"))
        try:  # GRAPE, CRAB format
            lbound = [b[0][0] for b in bounds]
            ubound = [b[0][1] for b in bounds]
        except Exception:
            lbound = [b[0] for b in bounds]
            ubound = [b[1] for b in bounds]

        # Set up minimizer options
        minimizer_kwargs.setdefault("options", {})
        minimizer_kwargs["options"].setdefault(
            "maxiter", algorithm_kwargs.get("max_iter", 1000)
        )
        minimizer_kwargs["options"].setdefault(
            "gtol", algorithm_kwargs.get("min_grad", 0.0 if alg == "CRAB" else 1e-8)
        )


        # Run the appropriate optimization algorithm
        if alg.upper() == "RL" and _rl_available:
            # Reinforcement learning optimization
            rl_optimizer = _RL(
                objectives=objectives,
                control_parameters=control_parameters,
                time_interval=time_interval,
                time_options=time_options,
                alg_kwargs=algorithm_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                minimizer_kwargs=minimizer_kwargs,
                integrator_kwargs=integrator_kwargs,
                qtrl_optimizers=None,
            )
            rl_optimizer.train()
            return rl_optimizer.result()
        else:
            # Standard optimization (GOAT, JOPT, GRAPE, CRAB)
            return _global_local_optimization(
                objectives=objectives,
                control_parameters=control_parameters,
                time_interval=time_interval,
                time_options=time_options,
                algorithm_kwargs=algorithm_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                minimizer_kwargs=minimizer_kwargs,
                integrator_kwargs=integrator_kwargs,
                qtrl_optimizers=None,
            )
    # prepare qtrl optimizers
    # Initialize empty list for optimizers (we'll use our own framework)
    def generate_pulse_from_params(params, n_tslots, num_coeffs=None, fix_frequency=False, 
                             init_coeff_scaling=1.0, crab_pulse_params=None):
        """
        Generate a CRAB pulse from Fourier coefficients or other parameters.
        
        Parameters:
        -----------
        params : array_like
            Array of control parameters (Fourier coefficients or other parameters)
        n_tslots : int
            Number of time slots in the pulse
        num_coeffs : int, optional
            Number of Fourier coefficients per control dimension
        fix_frequency : bool, optional
            Whether to use fixed frequencies
        init_coeff_scaling : float, optional
            Scaling factor for initial coefficients
        crab_pulse_params : dict, optional
            Additional parameters for CRAB pulse generation
        
        Returns:
        --------
        numpy.ndarray
            Generated pulse amplitudes for each time slot
        """
        # Default CRAB parameters
        if crab_pulse_params is None:
            crab_pulse_params = {}
        
        # Determine the number of coefficients per dimension
        if num_coeffs is None:
            if fix_frequency:
                num_coeffs = len(params) // 2  # amplitudes and phases
            else:
                num_coeffs = len(params) // 3  # amplitudes, phases, and frequencies
        
        # Reshape parameters based on CRAB type
        if fix_frequency:
            # Parameters are [A1, A2, ..., phi1, phi2, ...]
            amplitudes = params[:num_coeffs] * init_coeff_scaling
            phases = params[num_coeffs:2*num_coeffs]
            # Use fixed frequencies from crab_pulse_params or default
            frequencies = crab_pulse_params.get('frequencies', 
                                            np.linspace(1, 10, num_coeffs))
        else:
            # Parameters are [A1, A2, ..., phi1, phi2, ..., w1, w2, ...]
            amplitudes = params[:num_coeffs] * init_coeff_scaling
            phases = params[num_coeffs:2*num_coeffs]
            frequencies = params[2*num_coeffs:3*num_coeffs]
        
        # Time points
        t = np.linspace(0, 1, n_tslots)
        
        # Generate pulse using Fourier components
        pulse = np.zeros(n_tslots)
        for A, phi, w in zip(amplitudes, phases, frequencies):
            pulse += A * np.sin(w * t + phi)
        
        return pulse

    # Initialize empty list for optimizers (we'll use our own framework)
    optimizers = []

    if alg == "CRAB" or alg == "GRAPE":
        # Determine dynamics type (unitary or general matrix)
        dyn_type = "GEN_MAT"
        for objective in objectives:
            if any(qt.isoper(H_i) for H_i in (objective.H if isinstance(objective.H, list) else [objective.H])):
                dyn_type = "UNIT"

        # Algorithm-specific configurations
        if alg == "GRAPE":
            use_as_amps = True
            minimizer_kwargs.setdefault("method", "L-BFGS-B")  # gradient-based
            alg_params = algorithm_kwargs.get("alg_params", {})
            
        elif alg == "CRAB":
            minimizer_kwargs.setdefault("method", "Nelder-Mead")  # gradient-free
            use_as_amps = len(x0[0]) == time_interval.n_tslots
            num_coeffs = algorithm_kwargs.get("num_coeffs", None)
            fix_frequency = algorithm_kwargs.get("fix_frequency", False)

            if num_coeffs is None:
                if use_as_amps:
                    num_coeffs = 2  # default fourier coefficients
                else:
                    num_coeffs = len(x0[0]) // 2 if fix_frequency else len(x0[0]) // 3

            alg_params = {
                "num_coeffs": num_coeffs,
                "init_coeff_scaling": algorithm_kwargs.get("init_coeff_scaling", 1.0),
                "crab_pulse_params": algorithm_kwargs.get("crab_pulse_params", None),
                "fix_frequency": fix_frequency,
            }

        # Handle bounds
        if use_as_amps:
            lbound = lbound[0]
            ubound = ubound[0]

        # Prepare pulse parameters for each objective
        for i, objective in enumerate(objectives):
            # Generate initial pulses for each control
            init_amps = np.zeros((time_interval.n_tslots, len(Hc_lst[i])))
            
            for j in range(len(Hc_lst[i])):
                if use_as_amps:
                    # For amplitude-based optimization, use the initial guess directly
                    init_amps[:, j] = x0[j]
                else:
                    # For parameterized pulses, generate from coefficients
                    init_amps[:, j] = generate_pulse_from_params(
                        x0[j], 
                        n_tslots=time_interval.n_tslots,
                        num_coeffs=alg_params.get("num_coeffs"),
                        fix_frequency=alg_params.get("fix_frequency", False),
                        init_coeff_scaling=alg_params.get("init_coeff_scaling", 1.0),
                        crab_pulse_params=alg_params.get("crab_pulse_params")
                    )

            # Store the optimization problem configuration
            optimizers.append({
                "drift": Hd_lst[i],
                "ctrls": Hc_lst[i],
                "initial": objective.initial,
                "target": objective.target,
                "init_amps": init_amps,
                "bounds": (lbound, ubound),
                "alg_params": alg_params,
                "use_as_amps": use_as_amps,
                "dyn_type": dyn_type,
                "fid_type": algorithm_kwargs.get("fid_type", "PSU")  # Use our FidelityComputer
            })

            # Update control parameters if using parameterized pulses
            if not use_as_amps:
                num_params = len(x0[0])
                for key in control_parameters.keys():
                    if key != "__time__":
                        control_parameters[key]["bounds"] = [
                            (lbound, ubound) for _ in range(num_params)
                        ]

    elif alg == "RL":
        if not _rl_available:
            raise ImportError(
                "The required dependencies (gymnasium, stable-baselines3) for "
                "the reinforcement learning algorithm are not available."
            )

        rl_env = _RL(
            objectives,
            control_parameters,
            time_interval,
            time_options,
            algorithm_kwargs,
            optimizer_kwargs,
            minimizer_kwargs,
            integrator_kwargs,
            optimizers,  # Pass our optimizers list
        )
        rl_env.train()
        return rl_env.result()

    # Run the optimization using our custom framework
    return _global_local_optimization(
        objectives,
        control_parameters,
        time_interval,
        time_options,
        algorithm_kwargs,
        optimizer_kwargs,
        minimizer_kwargs,
        integrator_kwargs,
        optimizers,  # Pass our optimizers configuration
    )
