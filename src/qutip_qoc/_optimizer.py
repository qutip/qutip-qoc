"""
This module contains the optimization routine
to find the control parameters in a local and global search.
It also contains the callback class to keep track of the optimization process.
"""
import time
import numpy as np
import scipy as sp

from scipy.optimize import OptimizeResult
from qutip_qoc.result import Result
from qutip_qoc.objective import _MultiObjective


__all__ = ["_global_local_optimization"]


def _get_init_and_bounds_from_options(lst, input):
    """
    Extract initial and boundary values of any kind and shape
    from the control_parameters and time_options dictionary.
    """
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


class _Callback:
    """
    Callback functions for the local and global optimization algorithm.
    Keeps track of time and saves intermediate results.
    Terminates the optimization if the infidelity error target is reached.
    Class initialization starts the clock.
    """

    def __init__(self, result, fid_err_targ, max_wall_time, bounds, disp):
        self._result = result
        self._fid_err_targ = fid_err_targ
        self._max_wall_time = max_wall_time
        self._bounds = bounds
        self._disp = disp

        self._elapsed_time = 0
        self._iter_seconds = []
        self._start_time = self._iter_time = time.time()

    def stop_clock(self):
        """
        Stops the clock and saves the start-,end- and iterations- time in result.
        """
        self._end_time = time.time()

        self._result.start_local_time = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(self._start_time)
        )
        self._result.end_local_time = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(self._end_time)
        )

        self._result.iter_seconds = self._iter_seconds

    def _time_iter(self):
        """
        Calculates and stores the time for each iteration.
        """
        iter_time = time.time()
        diff = round(iter_time - self._iter_time, 4)
        self._iter_time = iter_time
        self._iter_seconds.append(diff)
        return diff

    def _time_elapsed(self):
        """
        Calculates and stores the elapsed time since the start of the optimization.
        """
        self._elapsed_time = round(time.time() - self._start_time, 4)
        return self._elapsed_time

    def inside_bounds(self, x):
        """
        Check if the current parameters are inside the boundaries
        used for the global and local optimization callback.
        """
        idx = 0
        for bound in self._bounds:
            for b in bound:
                if (b[0] and b[1]) and not (b[0] <= x[idx] <= b[1]):
                    if self._disp:
                        print("parameter out of bounds, continuing optimization")
                    return False
                idx += 1
        return True

    def min_callback(self, intermediate_result: OptimizeResult):
        """
        Callback function for the local minimizer,
        terminates if the infidelity target is reached or
        the maximum wall time is exceeded.
        """
        terminate = False

        if intermediate_result.fun <= self._fid_err_targ:
            terminate = True
            reason = "fid_err_targ reached"
        elif self._time_elapsed() >= self._max_wall_time:
            terminate = True
            reason = "max_wall_time reached"

        if self._disp:
            message = "minimizer step, infidelity: %.5f" % intermediate_result.fun
            if terminate:
                message += "\n" + reason + ", terminating minimization"
            print(message)

        if terminate:  # manually save the result and exit
            if intermediate_result.fun < self._result.infidelity:
                if intermediate_result.fun > 0:
                    if self.inside_bounds(intermediate_result.x):
                        self._result._update(
                            intermediate_result.fun, intermediate_result.x
                        )
            raise StopIteration

    def opt_callback(self, x, f, accept):
        """
        Callback function for the global optimizer,
        terminates if the infidelity target is reached or
        the maximum wall time is exceeded.
        """
        terminate = False
        global_step_seconds = self._time_iter()

        if f <= self._fid_err_targ:
            terminate = True
            self._result.message = "fid_err_targ reached"
        elif self._time_elapsed() >= self._max_wall_time:
            terminate = True
            self._result.message = "max_wall_time reached"

        if self._disp:
            message = (
                "optimizer step, infidelity: %.5f" % f
                + ", took %.2f seconds" % global_step_seconds
            )
            if terminate:
                message += "\n" + self._result.message + ", terminating optimization"
            print(message)

        if terminate:  # manually save the result and exit
            if f < self._result.infidelity:
                if f < 0:
                    print(
                        "WARNING: infidelity < 0 -> inaccurate integration, "
                        "try reducing integrator tolerance (atol, rtol), "
                        "continuing with global optimization"
                    )
                    terminate = False
                elif self.inside_bounds(x):
                    self._result._update(f, x)

        return terminate


def _global_local_optimization(
    objectives,
    control_parameters,
    time_interval,
    time_options,
    algorithm_kwargs,
    optimizer_kwargs,
    minimizer_kwargs,
    integrator_kwargs,
    qtrl_optimizers=None,
):
    """
    Optimize a pulse sequence to implement a given target unitary by optimizing
    the parameters of the pulse functions. The algorithm is a two-layered
    approach. The outer layer does a global optimization using basin-hopping or
    dual annealing. The inner layer does a local optimization using a gradient-
    based method (no gradient for CRAB).
    Gradients and error values are calculated in the MultiObjective module.

    Parameters
    ----------
    objectives : list of :class:`qutip_qoc.Objective`
        List of objectives to be optimized.

    control_parameters : dict
        Dictionary of options for the control pulse optimization.
        For each control function it must specify:

            control_id : dict
                - guess: ndarray, shape (n,)
                    Initial guess. Array of real elements of size (n,),
                    where ``n`` is the number of independent variables.

                - bounds : sequence, optional
                    Sequence of ``(min, max)`` pairs for each element in
                    `guess`. None is used to specify no bound.

    time_interval : :class:`qutip_qoc._TimeInterval`
        Time interval for the optimization.

    time_options : dict, optional
        Only supported by GOAT and JOPT.
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
                Supported are: "GOAT", "JOPT".

            - fid_err_targ : float, optional
                Fidelity error target for the optimization.

            - max_iter : int, optional
                Maximum number of global iterations to perform.
                Can be overridden by specifying in
                optimizer_kwargs/minimizer_kwargs.

    optimizer_kwargs : dict, optional
        Dictionary of options for the global optimizer.
        Only supported by GOAT and JOPT.

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
        Only supported by GOAT and JOPT.
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

    # extract initial and boundary values for global and local optimizer
    x0, bounds = [], []
    for key in control_parameters.keys():
        _get_init_and_bounds_from_options(x0, control_parameters[key].get("guess"))
        _get_init_and_bounds_from_options(bounds, control_parameters[key].get("bounds"))

    _get_init_and_bounds_from_options(x0, time_options.get("guess", None))
    _get_init_and_bounds_from_options(bounds, time_options.get("bounds", None))

    optimizer_kwargs["x0"] = np.concatenate(x0)

    multi_objective = _MultiObjective(
        objectives=objectives,
        qtrl_optimizers=qtrl_optimizers,
        time_interval=time_interval,
        time_options=time_options,
        control_parameters=control_parameters,
        alg_kwargs=algorithm_kwargs,
        guess_params=optimizer_kwargs["x0"],
        **integrator_kwargs,
    )

    # optimizer specific settings
    opt_method = optimizer_kwargs.get(
        "method", algorithm_kwargs.get("method", "basinhopping")
    )

    if opt_method == "basinhopping":
        optimizer = sp.optimize.basinhopping

        # if not specified through optimizer_kwargs "niter"
        optimizer_kwargs.setdefault(
            "niter",
            optimizer_kwargs.get("max_iter", algorithm_kwargs.get("glob_max_iter", 0)),
        )

        if len(bounds) != 0:  # realizes boundaries through minimizer
            minimizer_kwargs.setdefault("bounds", np.concatenate(bounds))

    elif opt_method == "dual_annealing":
        optimizer = sp.optimize.dual_annealing

        # if not specified through optimizer_kwargs "maxiter"
        keys = ["maxiter", "max_iter", "glob_max_iter", "niter"]
        value = next(
            (optimizer_kwargs.pop(key, None) or algorithm_kwargs.get(key))
            for key in keys
            if optimizer_kwargs.get(key) or algorithm_kwargs.get(key)
        )
        optimizer_kwargs.setdefault("maxiter", value)

        # remove remaining keys
        for key in keys[1:]:
            optimizer_kwargs.pop(key, None)

        if len(bounds) != 0:  # realizes boundaries through optimizer
            optimizer_kwargs.setdefault("bounds", np.concatenate(bounds))

    # remove overload from optimizer_kwargs
    optimizer_kwargs.pop("max_iter", None)
    optimizer_kwargs.pop("method", None)

    # should optimization include time (only for GOAT and JOPT)
    var_t = True if time_options.get("guess", False) else False

    # define the result object
    result = Result(
        objectives,
        time_interval,
        guess_params=x0,
        var_time=var_t,
        qtrl_optimizers=qtrl_optimizers,
    )

    # Callback instance for termination and logging
    max_wall_time = algorithm_kwargs.get("max_wall_time", 1e10)
    fid_err_targ = algorithm_kwargs.get("fid_err_targ", 1e-10)
    disp = algorithm_kwargs.get("disp", False)
    # start the clock
    cllbck = _Callback(result, fid_err_targ, max_wall_time, bounds, disp)

    # run the optimization
    min_res = optimizer(
        func=multi_objective.goal_fun,
        minimizer_kwargs={
            "jac": multi_objective.grad_fun,
            "callback": cllbck.min_callback,
            **minimizer_kwargs,
        },
        callback=cllbck.opt_callback,
        **optimizer_kwargs,
    )

    cllbck.stop_clock()  # stop the clock

    # some global optimization methods do not return the minimum result
    # when terminated through StopIteration (see min_callback)
    if min_res.fun < result.infidelity:
        if cllbck.inside_bounds(min_res.x):
            result._update(min_res.fun, min_res.x)

    # save runtime information in result
    result.n_iters = min_res.nit
    if result.message is None:
        result.message = (
            (
                "Local minimizer: " + min_res["lowest_optimization_result"].message
                if opt_method == "basinhopping"
                else ""  # dual_annealing does not return a local minimizer message
            )
            + " Global optimizer: "
            + min_res.message[0]
        )

    return result
