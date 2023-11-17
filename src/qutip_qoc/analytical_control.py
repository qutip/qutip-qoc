"""
This module contains functions that implement the JOAT algorithm to
calculate optimal parameters for analytical control pulse sequences.
"""
import numpy as np
import scipy as sp

import qutip as qt

from result import Result

from joat import Multi_JOAT
from goat import Multi_GOAT


def optimize_pulses(
        objectives,
        pulse_options,
        time_interval,
        time_options,
        algorithm_kwargs,
        **kwargs):
    """
    Optimize a pulse sequence to implement a given target unitary.

    Parameters
    ----------
    objectives : list of :class:`qutip.Qobj`
        List of objectives to be implemented.
    pulse_options : dict of dict
        Dictionary of options for each pulse.
        guess : list of floats
            Initial guess for the pulse parameters.
        bounds : list of pairs of floats
            [(lower, upper), ...]
            Bounds for the pulse parameters.
    tslots : array_like
        List of times for the calculataion of final pulse sequence.
        During integration only the first and last time are used.
    kwargs : dict of dict
        Dictionary keys are "optimizer", "minimizer" and "integrator".

        The "optimizer" dictionary contains keyword arguments for the optimizer:
            niter : integer, optional
                The number of basin-hopping iterations. There will be a total of
                ``niter + 1`` runs of the local minimizer.
            T : float, optional
                The "temperature" parameter for the acceptance or rejection criterion.
                Higher "temperatures" mean that larger jumps in function value will be
                accepted.  For best results `T` should be comparable to the
                separation (in function value) between local minima.
            stepsize : float, optional
                Maximum step size for use in the random displacement.
            take_step : callable ``take_step(x)``, optional
                Replace the default step-taking routine with this routine. The default
                step-taking routine is a random displacement of the coordinates, but
                other step-taking algorithms may be better for some systems.
                `take_step` can optionally have the attribute ``take_step.stepsize``.
                If this attribute exists, then `basinhopping` will adjust
                ``take_step.stepsize`` in order to try to optimize the global minimum
                search.
            accept_test : callable, ``accept_test(f_new=f_new, x_new=x_new, f_old=fold, x_old=x_old)``, optional
                Define a test which will be used to judge whether to accept the
                step. This will be used in addition to the Metropolis test based on
                "temperature" `T`. The acceptable return values are True,
                False, or ``"force accept"``. If any of the tests return False
                then the step is rejected. If the latter, then this will override any
                other tests in order to accept the step. This can be used, for example,
                to forcefully escape from a local minimum that `basinhopping` is
                trapped in.
            callback : callable, ``callback(x, f, accept)``, optional
                A callback function which will be called for all minima found. ``x``
                and ``f`` are the coordinates and function value of the trial minimum,
                and ``accept`` is whether that minimum was accepted. This can
                be used, for example, to save the lowest N minima found. Also,
                `callback` can be used to specify a user defined stop criterion by
                optionally returning True to stop the `basinhopping` routine.
            interval : integer, optional
                interval for how often to update the `stepsize`
            disp : bool, optional
                Set to True to print status messages
            niter_success : integer, optional
                Stop the run if the global minimum candidate remains the same for this
                number of iterations.
            seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional

                If `seed` is None (or `np.random`), the `numpy.random.RandomState`
                singleton is used.
                If `seed` is an int, a new ``RandomState`` instance is used,
                seeded with `seed`.
                If `seed` is already a ``Generator`` or ``RandomState`` instance then
                that instance is used.
                Specify `seed` for repeatable minimizations. The random numbers
                generated with this seed only affect the default Metropolis
                `accept_test` and the default `take_step`. If you supply your own
                `take_step` and `accept_test`, and these functions use random
                number generation, then those functions are responsible for the state
                of their random number generator.
            target_accept_rate : float, optional
                The target acceptance rate that is used to adjust the `stepsize`.
                If the current acceptance rate is greater than the target,
                then the `stepsize` is increased. Otherwise, it is decreased.
                Range is (0, 1). Default is 0.5.
            See scipy.optimize.basinhopping for more details.

        The "minimizer" dictionary contains keyword arguments for the minimizer:
            method : str or callable, optional
                Type of solver.  Should be one of
                    - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
                    - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
                    - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
                    - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
                    - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
                    - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
                    - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
                    - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
                    - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
                    - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
                    - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
                    - custom - a callable object, see below for description.
                If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
                depending on whether or not the problem has constraints or bounds.
            bounds : sequence or `Bounds`, optional
                Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell,
                trust-constr, and COBYLA methods. There are two ways to specify the
                bounds:
                    1. Instance of `Bounds` class.
                    2. Sequence of ``(min, max)`` pairs for each element in `x`. None
                    is used to specify no bound.
            constraints : {Constraint, dict} or List of {Constraint, dict}, optional
                Constraints definition. Only for COBYLA, SLSQP and trust-constr.

                Constraints for 'trust-constr' are defined as a single object or a
                list of objects specifying constraints to the optimization problem.
                Available constraints are:
                    - `LinearConstraint`
                    - `NonlinearConstraint`
                Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
                Each dictionary with fields:
                    type : str
                        Constraint type: 'eq' for equality, 'ineq' for inequality.
                    fun : callable
                        The function defining the constraint.
                    jac : callable, optional
                        The Jacobian of `fun` (only for SLSQP).
                    args : sequence, optional
                        Extra arguments to be passed to the function and Jacobian.
                Equality constraint means that the constraint function result is to
                be zero whereas inequality means that it is to be non-negative.
                Note that COBYLA only supports inequality constraints.
            tol : float, optional
                Tolerance for termination. When `tol` is specified, the selected
                minimization algorithm sets some relevant solver-specific tolerance(s)
                equal to `tol`. For detailed control, use solver-specific
                options.
            options : dict, optional
                A dictionary of solver options. All methods except `TNC` accept the
                following generic options:
                    maxiter : int
                        Maximum number of iterations to perform. Depending on the
                        method each iteration may use several function evaluations.
                        For `TNC` use `maxfun` instead of `maxiter`.
                    disp : bool
                        Set to True to print convergence messages.
                For method-specific options, see :func:`show_options()`.
            See scipy.optimize.minimize for more details.

        The "integrator" dictionary contains keyword arguments for the integrator options:
                - progress_bar : str {'text', 'enhanced', 'tqdm', ''}
                How to present the integrator progress.
                'tqdm' uses the python module of the same name and raise an error
                if not installed. Empty string or False will disable the bar.
                - progress_kwargs : dict
                kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
                - method : str ["adams", "bdf", "lsoda", "dop853", "vern9", etc.]
                Which differential equation integration method to use.
                - atol, rtol : float
                Absolute and relative tolerance of the ODE integrator.
                - nsteps : int
                Maximum number of (internally defined) steps allowed in one ``tslots``
                step.
                - max_step : float, 0
                Maximum lenght of one internal step. When using pulses, it should be
                less than half the width of the thinnest pulse.

                Other options could be supported depending on the integration method,
                see `Integrator <./classes.html#classes-ode>`_.

    Returns
    -------
    """
    optimizer_kwargs = kwargs.get("optimizer_kwargs", {})
    minimizer_kwargs = kwargs.get("minimizer_kwargs", {})
    integrator_kwargs = kwargs.get("integrator_kwargs", {})

    # integrator must not normalize output
    integrator_kwargs["normalize_output"] = False
    integrator_kwargs["progress_bar"] = integrator_kwargs.get(
        "progress_bar", False)

    def helper(lst, input):
        # to extract initial and boundary values
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

    optimizer_kwargs["x0"] = np.concatenate(x0)

    if algorithm_kwargs.get("alg") == "JOAT":
        with qt.CoreOptions(default_dtype="jaxdia"):
            multi_objective = Multi_JOAT(objectives, time_interval, time_options,
                                         pulse_options, algorithm_kwargs,
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
        minimizer_kwargs["bounds"] = np.concatenate(bounds)

    elif opt_method == "dual_annealing":
        optimizer = sp.optimize.dual_annealing

        # if not specified through optimizer_kwargs "maxiter"
        optimizer_kwargs.setdefault(  # or "max_iter"
            "maxiter", optimizer_kwargs.get(  # use algorithm_kwargs
                "max_iter", algorithm_kwargs.get("max_iter", 1000)))

        # realizes boundaries through optimizer
        optimizer_kwargs["bounds"] = np.concatenate(bounds)

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
                if inside_bounds(x):
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
