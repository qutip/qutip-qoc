import time
import numpy as np

import qutip_qtrl.logging_utils as logging
from qutip_qtrl.pulseoptim import optimize_pulse

from qutip_qoc.analytical_control import optimize_pulses as opt_pulses
from qutip_qoc.result import Result

__all__ = ["optimize_pulses"]


def optimize_pulses(objectives, pulse_options, time_interval, time_options={},
                    algorithm_kwargs={}, optimizer_kwargs={},
                    minimizer_kwargs={}, integrator_kwargs={}):
    """
    Wrapper to choose between GOAT/JOAT and GRAPE/CRAB optimization.

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

        GRAPE and CRAB have only one control function with n_tslots parameters.
        The bounds are only one pair of ``(min, max)`` limiting all tslots
        equally.

    time_interval : :class:`qutip_qoc.TimeInterval`
        Time interval for the optimization.
        GRAPE and CRAB require n_tslots attribute.

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
                Supported are: "GRAPE", "CRAB", "GOAT", "JOAT".

            - fid_err_targ : float, optional
                Fidelity error target for the optimization.

            - max_iter : int, optional
                Maximum number of iterations to perform.
                Referes to global steps for GOAT/JOAT and
                local minimizer steps for GRAPE/CRAB.
                Can be overridden by specifying in
                optimizer_kwargs/minimizer_kwargs.

        Algorithm specific keywords for GRAPE/CRAB can be found in
        :func:`qutip_qtrl.pulseoptim.optimize_pulse`.

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
    alg = algorithm_kwargs.get("alg", "GRAPE")

    if alg == "GOAT" or alg == "JOAT":
        return opt_pulses(
            objectives,
            pulse_options,
            time_interval,
            time_options,
            algorithm_kwargs,
            optimizer_kwargs,
            minimizer_kwargs,
            integrator_kwargs,
        )

    elif alg == "GRAPE" or alg == "CRAB":
        if len(objectives) != 1:
            raise TypeError(
                "GRAPE and CRAB optimization only supports one objective at a "
                "time. Please use GOAT or JOAT for multiple objectives."
            )
        objective = objectives[0]

        # extract drift and control Hamiltonians from the objective
        Hd = objective.H[0]
        Hc_lst = [H[0] for H in objective.H[1:]]

        # extract initial and target states/operators from the objective
        init = objective.initial
        targ = objective.target

        # extract guess and bounds for the control pulses
        x0, bounds = [], []
        for key in pulse_options.keys():
            x0.append(pulse_options[key].get("guess"))
            bounds.append(pulse_options[key].get("bounds"))

        try:
            lbound = [b[0][0] for b in bounds]
            ubound = [b[0][1] for b in bounds]
        except Exception:
            lbound = [b[0] for b in bounds]
            ubound = [b[1] for b in bounds]

        if alg == "CRAB":
            min_g = 0.
            algorithm_kwargs["alg_params"] = {
                "guess_pulse": x0,
            }

        elif alg == "GRAPE":
            # only alowes for scalar bound
            lbound = lbound[0]
            ubound = ubound[0]

            min_g = minimizer_kwargs.get("gtol", 1e-10)

            algorithm_kwargs["alg_params"] = {
                "init_amps": np.array(x0).T,
            }

        # default "log_level" if not specified
        if algorithm_kwargs.get("disp", False):
            log_level = logging.INFO
        else:
            log_level = logging.WARN

        # low level minimizer overrides high level algorithm kwargs
        max_iter = minimizer_kwargs.get("options", {}).get(
            "maxiter", algorithm_kwargs.get("max_iter", 1000))

        optim_method = minimizer_kwargs.get(
            "method", algorithm_kwargs.get("optim_method", "DEF"))

        result = Result(objectives, time_interval)

        start_time = time.time()

        res = optimize_pulse(
            drift=Hd,
            ctrls=Hc_lst,
            initial=init,
            target=targ,
            num_tslots=time_interval.n_tslots,
            evo_time=time_interval.evo_time,
            tau=None,  # implicitly derived from tslots
            amp_lbound=lbound,
            amp_ubound=ubound,
            fid_err_targ=algorithm_kwargs.get("fid_err_targ", 1e-10),
            min_grad=min_g,
            max_iter=max_iter,
            max_wall_time=algorithm_kwargs.get("max_wall_time", 180),
            alg=alg,
            optim_method=optim_method,
            method_params=minimizer_kwargs,

            optim_alg=None,  # deprecated
            max_metric_corr=None,  # deprecated
            accuracy_factor=None,  # deprecated
            alg_params=algorithm_kwargs.get("alg_params", None),
            optim_params=algorithm_kwargs.get("optim_params", None),
            dyn_type=algorithm_kwargs.get("dyn_type", "GEN_MAT"),
            dyn_params=algorithm_kwargs.get("dyn_params", None),
            prop_type=algorithm_kwargs.get("prop_type", "DEF"),
            prop_params=algorithm_kwargs.get("prop_params", None),
            fid_type=algorithm_kwargs.get("fid_type", "DEF"),
            fid_params=algorithm_kwargs.get("fid_params", None),
            phase_option=None,  # deprecated
            fid_err_scale_factor=None,  # deprecated
            tslot_type=algorithm_kwargs.get("tslot_type", "DEF"),
            tslot_params=algorithm_kwargs.get("tslot_params", None),
            amp_update_mode=None,  # deprecated
            init_pulse_type=algorithm_kwargs.get("init_pulse_type", "DEF"),
            init_pulse_params=algorithm_kwargs.get("init_pulse_params", None),
            pulse_scaling=algorithm_kwargs.get("pulse_scaling", 1.0),
            pulse_offset=algorithm_kwargs.get("pulse_offset", 0.0),
            ramping_pulse_type=algorithm_kwargs.get(
                "ramping_pulse_type", None),
            ramping_pulse_params=algorithm_kwargs.get(
                "ramping_pulse_params", None),
            log_level=algorithm_kwargs.get("log_level", log_level),
            out_file_ext=algorithm_kwargs.get("out_file_ext", None),
            gen_stats=algorithm_kwargs.get("gen_stats", False),
        )

        end_time = time.time()

        # extract runtime information
        result.start_local_time = time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        result.end_local_time = time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(end_time))
        total_seconds = end_time - start_time
        result.iter_seconds = [res.num_iter / total_seconds] * res.num_iter
        result.n_iters = res.num_iter
        result.message = res.termination_reason
        result.final_states = [res.evo_full_final]
        result.infidelity = res.fid_err
        result.guess_params = res.initial_amps.T
        result.optimized_params = res.final_amps.T

        # not present in analytical results
        result.stats = res.stats

        return result
    else:
        raise ValueError(
            "Unknown algorithm: %s; choose either GOAT, JOAT, GRAPE or CRAB." %
            alg)
