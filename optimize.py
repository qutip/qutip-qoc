import numpy as np

import qutip_qtrl.logging_utils as logging
from qutip_qtrl.pulseoptim import optimize_pulse

from analytical_control import optimize_pulses as opt_pulses
from result import Result


def optimize_pulses(objectives, pulse_options, time_interval, time_options={}, algorithm_kwargs={}, **kwargs):
    """
    Wrapper to choose between GOAT/JOAT and GRAPE/CRAB optimization.
    """
    alg = algorithm_kwargs.get("alg", "GRAPE")
    if alg == "GOAT" or alg == "JOAT":
        return opt_pulses(
            objectives,
            pulse_options,
            time_interval,
            time_options,
            algorithm_kwargs,
            **kwargs
        )
    else:  # GRAPE or CRAB
        if len(objectives) != 1:
            raise TypeError("GRAPE and CRAB optimization only supports one objective at a time. "
                            "Please use GOAT or JOAT for multiple objectives.")
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

        if alg == "GRAPE":  # only alowes for scalar bound
            lbound = lbound[0]
            ubound = ubound[0]

        minimizer_kwargs = kwargs.get("minimizer_kwargs", {})

        if alg == "CRAB":
            min_g = 0.
            algorithm_kwargs["alg_params"] = {
                "guess_pulse": x0,
            }
        elif alg == "GRAPE":
            min_g = minimizer_kwargs.get("gtol", 1e-10)
            algorithm_kwargs["alg_params"] = {
                "init_amps": np.array(x0).T,
            }

        # default "log_level" if not specified
        if algorithm_kwargs.get("disp", False):
            log_level = logging.INFO
        else:
            log_level = logging.WARN

        result = Result(objectives, time_interval)
        result.start_time()

        res = optimize_pulse(
            drift=Hd,
            ctrls=Hc_lst,
            initial=init,
            target=targ,
            num_tslots=time_interval.num_tslots,
            evo_time=time_interval.evo_time,
            tau=None,  # implicitly derived from tlist
            amp_lbound=lbound,
            amp_ubound=ubound,
            fid_err_targ=algorithm_kwargs.get("fid_err_targ", 1e-10),
            min_grad=min_g,
            max_iter=algorithm_kwargs.get("max_iter", 500),
            max_wall_time=algorithm_kwargs.get("max_wall_time", 180),
            alg=alg,
            optim_method=minimizer_kwargs.get("method", "DEF"),
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

        result.end_time()

        result.iters = res.num_iter
        result.iter_seconds = [res.num_iter / result.time_delta]
        result.message = res.termination_reason
        result.final_states = [res.evo_full_final]
        result.infidelity = res.fid_err

        # TODO: GRAPE looks good, can we make params of CRAB accessible?
        n_cntrls = res.initial_amps.shape[1]

        result.guess_params = res.initial_amps.T
        result.optimized_params = res.final_amps.T

        # not present in GOAT result
        result.stats = res.stats

        return result
