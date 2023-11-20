"""
This module contains the optimization routine
for optimizing analytical control functions
with GOAT resp. JOAT.
"""
import numpy as np
import scipy as sp
import qutip as qt

from qutip_qoc.result import Result
from qutip_qoc.joat import Multi_JOAT
from qutip_qoc.goat import Multi_GOAT


def optimize_pulses(
        objectives,
        pulse_options,
        time_interval,
        time_options,
        algorithm_kwargs,
        optimizer_kwargs,
        minimizer_kwargs,
        integrator_kwargs):
    """
    Optimize a pulse sequence to implement a given target unitary by optimizing
    the parameters of the pulse functions. The algorithm is a two-layered
    approach. The outer layer does a global optimization using basin-hopping or
    dual annealing. The inner layer does a local optimization using a gradient-
    based method. Gradients and error values are calculated in the GOAT/JOAT
    module.

    Parameters
    ----------
    objectives : list of :class:`qutip_qoc.Objective`
        List of objectives to be optimized.

    pulse_options : dict
        Dictionary of options for the control pulse optimization.
        For each control function it must specify:

            control_id : dict
                - guess: ndarray, shape (n,)
                    Initial guess. Array of real elements of size (n,),
                    where ``n`` is the number of independent variables.

                - bounds : sequence, optional
                    Sequence of ``(min, max)`` pairs for each element in
                    `guess`. None is used to specify no bound.

    time_interval : :class:`qutip_qoc.TimeInterval`
        Time interval for the optimization.

    time_options : dict, optional
        Only supported by GOAT and JOAT.
        Dictionary of options for the time interval optimization.
        It must specify both:

            - guess: ndarray, shape (n,)
                Initial guess. Array of real elements of size (n,),
                where ``n`` is the number of independent variables.

            - bounds : sequence, optional
                Sequence of ``(min, max)`` pairs for each element in `guess`.
                None is used to specify no bound.

    algorithm_kwargs : dict, optional
        Dictionary of options for the optimization algorithm.

            - alg : str
                Algorithm to use for the optimization.
                Supported are: "GOAT", "JOAT".

            - fid_err_targ : float, optional
                Fidelity error target for the optimization.

            - max_iter : int, optional
                Maximum number of global iterations to perform.
                Can be overridden by specifying in
                optimizer_kwargs/minimizer_kwargs.

    optimizer_kwargs : dict, optional
        Dictionary of options for the global optimizer.
        Only supported by GOAT and JOAT.

            - method : str, optional
                Algorithm to use for the global optimization.
                Supported are: "basinhopping", "dual_annealing"

            - max_iter : int, optional
                Maximum number of iterations to perform.

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
        Only supported by GOAT and JOAT.
        Options for the solver, see :obj:`MESolver.options` and
        `Integrator <./classes.html#classes-ode>`_ for a list of all options.

    Returns
    -------
    result : :class:`qutip_qoc.Result`
        Optimization result.
    """
    # integrator must not normalize output
    integrator_kwargs["normalize_output"] = False
    integrator_kwargs.setdefault("progress_bar", False)

    def helper(lst, input):
        # to extract initial and boundary values
        # of any kind and shape
        if input is None:
            return lst
        if isinstance(input, (list, np.ndarray)):
            lst.append(input)
        elif isinstance(input, (tuple)):
            lst.append([input])
        elif np.isscalar(input):
            lst.append([input])
        else:  # jax Array
            lst.append(np.array(input))
        return lst

    x0, bounds = [], []
    for key in pulse_options.keys():
        helper(x0, pulse_options[key].get("guess"))
        helper(bounds, pulse_options[key].get("bounds"))

    helper(x0, time_options.get("guess", None))
    helper(bounds, time_options.get("bounds", None))

    optimizer_kwargs.setdefault("x0", np.concatenate(x0))

    if algorithm_kwargs.get("alg") == "JOAT":
        with qt.CoreOptions(default_dtype="jaxdia"):
            multi_objective = Multi_JOAT(objectives, time_interval,
                                         time_options, pulse_options,
                                         algorithm_kwargs,
                                         guess_params=optimizer_kwargs["x0"],
                                         **integrator_kwargs)
    elif algorithm_kwargs.get("alg") == "GOAT":
        multi_objective = Multi_GOAT(objectives, time_interval, time_options,
                                     pulse_options, algorithm_kwargs,
                                     guess_params=optimizer_kwargs["x0"],
                                     **integrator_kwargs)

    max_wall_time = algorithm_kwargs.get("max_wall_time", 1e10)
    fid_err_targ = algorithm_kwargs.get("fid_err_targ", 1e-10)
    disp = algorithm_kwargs.get("disp", False)

    # optimizer specific settings
    opt_method = optimizer_kwargs.get(
        "method", algorithm_kwargs.get("method", "basinhopping"))

    if opt_method == "basinhopping":
        optimizer = sp.optimize.basinhopping

        # if not specified through optimizer_kwargs "niter"
        optimizer_kwargs.setdefault(  # or "max_iter"
            "niter", optimizer_kwargs.get(  # use algorithm_kwargs
                "max_iter", algorithm_kwargs.get("max_iter", 1000)))

        # realizes boundaries through minimizer
        minimizer_kwargs.setdefault("bounds", np.concatenate(bounds))

    elif opt_method == "dual_annealing":
        optimizer = sp.optimize.dual_annealing

        # if not specified through optimizer_kwargs "maxiter"
        optimizer_kwargs.setdefault(  # or "max_iter"
            "maxiter", optimizer_kwargs.get(  # use algorithm_kwargs
                "max_iter", algorithm_kwargs.get("max_iter", 1000)))

        # realizes boundaries through optimizer
        optimizer_kwargs.setdefault("bounds", np.concatenate(bounds))

    # remove overload from optimizer_kwargs
    optimizer_kwargs.pop("max_iter", None)
    optimizer_kwargs.pop("method", None)

    # define the result Krotov style
    result = Result(objectives,
                    time_interval,
                    guess_params=x0,
                    var_time=time_options.get("guess", False))

    # helper functions for callbacks
    def inside_bounds(x):
        idx = 0
        for bound in bounds:
            for b in bound:
                if not (b[0] <= x[idx] <= b[1]):
                    print("parameter out of bounds, continuing optimization")
                    return False
                idx += 1
        return True

    def min_callback(intermediate_result):
        terminate = False

        if intermediate_result.fun <= fid_err_targ:
            terminate = True
            reason = "fid_err_targ reached"
        elif result.time_elapsed() >= max_wall_time:
            terminate = True
            reason = "max_wall_time reached"

        if disp:
            message = "minimizer step, infidelity: %.5f" % intermediate_result.fun
            if terminate:
                message += "\n" + reason + ", terminating minimization"
            print(message)

        if terminate:  # manually save the result and exit
            if intermediate_result.fun < result.infidelity:
                if intermediate_result.fun > 0:
                    if inside_bounds(intermediate_result.x):
                        result.update(intermediate_result.fun,
                                      intermediate_result.x)
            raise StopIteration

    def opt_callback(x, f, accept):
        terminate = False

        if f <= fid_err_targ:
            terminate = True
            result.message = "fid_err_targ reached"
        elif result.time_elapsed() >= max_wall_time:
            terminate = True
            result.message = "max_wall_time reached"

        if disp:
            message = "optimizer step, infidelity: %.5f" % f +\
                ", took %.2f seconds" % result.time_iter()
            if terminate:
                message += "\n" + result.message + ", terminating optimization"
            print(message)

        if terminate:  # manually save the result and exit
            if f < result.infidelity:
                if f < 0:
                    print(
                        "WARNING: infidelity < 0 -> inaccurate integration, "
                        "try reducing integrator tolerance (atol, rtol), "
                        "continuing with global optimization")
                    terminate = False
                elif inside_bounds(x):
                    result.update(f, x)

        return terminate

    result.start_time()

    # run the optimization
    min_res = optimizer(
        func=multi_objective.goal_fun,
        minimizer_kwargs={
            'jac': multi_objective.grad_fun,
            'callback': min_callback,
            **minimizer_kwargs
        },
        callback=opt_callback,
        **optimizer_kwargs
    )

    result.end_time()

    # some global optimization methods do not return the minimum result
    # when terminated through StopIteration (see min_callback)
    if min_res.fun < result.infidelity:
        if inside_bounds(min_res.x):
            result.update(min_res.fun, min_res.x)

    result.iters = min_res.nit
    if result.message is None:
        result.message = min_res.message

    return result
