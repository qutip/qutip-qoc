import time
import numpy as np

import qutip_qtrl.logging_utils as logging
from qutip_qtrl.pulseoptim import optimize_pulse, create_pulse_optimizer

from qutip_qoc.analytical_control import optimize_pulses as opt_pulses
from qutip_qoc.result import Result

__all__ = ["optimize_pulses"]


def optimize_pulses(
    objectives,
    pulse_options,
    time_interval,
    time_options={},
    algorithm_kwargs={},
    optimizer_kwargs={},
    minimizer_kwargs={},
    integrator_kwargs={},
):
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

    if isinstance(objectives, list):
        if alg == "GRAPE" and len(objectives) != 1:
            raise TypeError(
                "GRAPE optimization only supports one objective at a time. Please use CRAB, GOAT or JOAT for multiple objectives."
            )
    else:
        objectives = [objectives]

    Hd_lst, Hc_lst = [], []
    for objective in objectives:
        # extract drift and control Hamiltonians from the objective
        Hd_lst.append(objective.H[0])
        Hc_lst.append([H[0] if isinstance(H, list) else H for H in objective.H[1:]])

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

    # default "log_level" if not specified
    if algorithm_kwargs.get("disp", False):
        log_level = logging.INFO
    else:
        log_level = logging.WARN

    # low level minimizer overrides high level algorithm kwargs
    max_iter = minimizer_kwargs.get("options", {}).get(
        "maxiter", algorithm_kwargs.get("max_iter", 1000)
    )

    optim_method = minimizer_kwargs.get(
        "method", algorithm_kwargs.get("optim_method", "BFGS")
    )

    result = Result(objectives, time_interval)

    start_time = time.time()

    if alg == "GRAPE":
        # only allow for scalar bound
        lbound = lbound[0]
        ubound = ubound[0]

        min_g = minimizer_kwargs.get("gtol", 1e-10)

        algorithm_kwargs["alg_params"] = {
            "init_amps": np.array(x0).T,
        }

        res = optimize_pulse(
            drift=Hd_lst[0],
            ctrls=Hc_lst[0],
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
            ramping_pulse_type=algorithm_kwargs.get("ramping_pulse_type", None),
            ramping_pulse_params=algorithm_kwargs.get("ramping_pulse_params", None),
            log_level=algorithm_kwargs.get("log_level", log_level),
            out_file_ext=algorithm_kwargs.get("out_file_ext", None),
            gen_stats=algorithm_kwargs.get("gen_stats", False),
        )
        end_time = time.time()

        # extract runtime information
        result.start_local_time = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(start_time)
        )
        result.end_local_time = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(end_time)
        )
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
        return result  # GRAPE result

    crab_optimizer = []
    if alg == "CRAB":
        # Check wether guess referes to amplitudes or parameters
        use_as_amps = len(x0[0]) == time_interval.n_tslots
        num_coeffs = algorithm_kwargs.get("num_coeffs", None)
        fix_frequency = algorithm_kwargs.get("fix_frequency", False)

        if num_coeffs is None:
            # default only two sets of fourier expansion coefficients
            if use_as_amps:
                num_coeffs = 2
            else:  # depending on the number of parameters given
                num_coeffs = len(x0[0]) // 2 if fix_frequency else len(x0[0]) // 3

        for i, objective in enumerate(objectives):
            alg_params = {
                "drift": Hd_lst[i],
                "ctrls": Hc_lst[i],
                "initial": init,
                "target": targ,
                "num_tslots": time_interval.n_tslots,
                "evo_time": time_interval.evo_time,
                "tau": None,  # implicitly derived from tslots
                "amp_lbound": lbound,
                "amp_ubound": ubound,
                "fid_err_targ": algorithm_kwargs.get("fid_err_targ", 1e-10),
                "min_grad": 0.0,
                "max_iter": max_iter,
                "max_wall_time": algorithm_kwargs.get("max_wall_time", 180),
                "alg": alg,
                "optim_method": optim_method,
                "method_params": minimizer_kwargs,
                "optim_alg": None,  # deprecated
                "max_metric_corr": None,  # deprecated
                "accuracy_factor": None,  # deprecated
                "alg_params": {
                    "num_coeffs": num_coeffs,
                    "init_coeff_scaling": algorithm_kwargs.get("init_coeff_scaling"),
                    "crab_pulse_params": algorithm_kwargs.get("crab_pulse_params"),
                    "fix_frequency": fix_frequency,
                },
                "optim_params": algorithm_kwargs.get("optim_params", None),
                "dyn_type": algorithm_kwargs.get("dyn_type", "GEN_MAT"),
                "dyn_params": algorithm_kwargs.get("dyn_params", None),
                "prop_type": algorithm_kwargs.get("prop_type", "FRECHET"),
                "prop_params": algorithm_kwargs.get("prop_params", None),
                "fid_type": algorithm_kwargs.get("fid_type", "DEF"),
                "fid_params": algorithm_kwargs.get("fid_params", None),
                "phase_option": None,  # deprecated
                "fid_err_scale_factor": None,  # deprecated
                "tslot_type": algorithm_kwargs.get("tslot_type", "DEF"),
                "tslot_params": algorithm_kwargs.get("tslot_params", None),
                "amp_update_mode": None,  # deprecated
                "init_pulse_type": algorithm_kwargs.get("init_pulse_type", "DEF"),
                "init_pulse_params": algorithm_kwargs.get(
                    "init_pulse_params", None
                ),  # wavelength, frequency, phase etc.
                "pulse_scaling": algorithm_kwargs.get("pulse_scaling", 1.0),
                "pulse_offset": algorithm_kwargs.get("pulse_offset", 0.0),
                "ramping_pulse_type": algorithm_kwargs.get("ramping_pulse_type", None),
                "ramping_pulse_params": algorithm_kwargs.get(
                    "ramping_pulse_params", None
                ),
                "log_level": algorithm_kwargs.get(
                    "log_level", log_level
                ),  # TODO: deprecate
                "gen_stats": algorithm_kwargs.get("gen_stats", False),
            }

            ###### code from qutip_qtrl.pulseoptim.optimize_pulse ######

            crab_optim = create_pulse_optimizer(
                drift=alg_params["drift"],
                ctrls=alg_params["ctrls"],
                initial=alg_params["initial"],
                target=alg_params["target"],
                num_tslots=alg_params["num_tslots"],
                evo_time=alg_params["evo_time"],
                tau=alg_params["tau"],
                amp_lbound=alg_params["amp_lbound"],
                amp_ubound=alg_params["amp_ubound"],
                fid_err_targ=alg_params["fid_err_targ"],
                min_grad=alg_params["min_grad"],
                max_iter=alg_params["max_iter"],
                max_wall_time=alg_params["max_wall_time"],
                alg=alg_params["alg"],
                optim_method=alg_params["optim_method"],
                method_params=alg_params["method_params"],
                optim_alg=alg_params["optim_alg"],
                max_metric_corr=alg_params["max_metric_corr"],
                accuracy_factor=alg_params["accuracy_factor"],
                alg_params=alg_params["alg_params"],
                optim_params=alg_params["optim_params"],
                dyn_type=alg_params["dyn_type"],
                dyn_params=alg_params["dyn_params"],
                prop_type=alg_params["prop_type"],
                prop_params=alg_params["prop_params"],
                fid_type=alg_params["fid_type"],
                fid_params=alg_params["fid_params"],
                phase_option=alg_params["phase_option"],
                fid_err_scale_factor=alg_params["fid_err_scale_factor"],
                tslot_type=alg_params["tslot_type"],
                tslot_params=alg_params["tslot_params"],
                amp_update_mode=alg_params["amp_update_mode"],
                init_pulse_type=alg_params["init_pulse_type"],
                init_pulse_params=alg_params["init_pulse_params"],
                pulse_scaling=alg_params["pulse_scaling"],
                pulse_offset=alg_params["pulse_offset"],
                ramping_pulse_type=alg_params["ramping_pulse_type"],
                ramping_pulse_params=alg_params["ramping_pulse_params"],
                log_level=alg_params["log_level"],
                gen_stats=alg_params["gen_stats"],
            )

            dyn = crab_optim.dynamics
            dyn.init_timeslots()

            # Generate initial pulses for each control through generator
            init_amps = np.zeros([dyn.num_tslots, dyn.num_ctrls])

            for j in range(dyn.num_ctrls):
                # Create the pulse generator for each control
                pgen = crab_optim.pulse_generator[j]

                if use_as_amps:
                    pgen.init_pulse()
                    init_amps[:, j] = x0[j]
                else:
                    pgen.set_optim_var_vals(np.array(x0[j]))
                    init_amps[:, j] = pgen.gen_pulse()

            # Initialise the starting amplitudes
            dyn.initialize_controls(init_amps)
            # And store the corresponding parameters
            init_params = crab_optim._get_optim_var_vals()

            if use_as_amps:  # For the global optimizer
                num_params = len(init_params) // len(pulse_options)
                for i, key in enumerate(pulse_options.keys()):
                    pulse_options[key]["guess"] = init_params[
                        i * num_params : (i + 1) * num_params
                    ]  # amplitude bounds are taken care of by pulse generator
                    pulse_options[key]["bounds"] = None

            crab_optimizer.append(crab_optim)

    return opt_pulses(
        objectives,
        pulse_options,
        time_interval,
        time_options,
        algorithm_kwargs,
        optimizer_kwargs,
        minimizer_kwargs,
        integrator_kwargs,
        crab_optimizer,
    )
