import numpy as np
import qutip_qtrl.logging_utils as logging
import qutip_qtrl.optimizer as opt
import types
import copy
logger = logging.get_logger()

class CRAB(opt.OptimizerCrab):
    def __init__(self, cfg, dyn, params, termination_conditions):
        super().__init__(cfg, dyn, params)
        self.init_optim(termination_conditions)


    # overwrite
def fid_err_func_wrapper(self, *args):
    """
    Get the fidelity error achieved using the ctrl amplitudes passed
    in as the first argument.

    This is called by generic optimisation algorithm as the
    func to the minimised. The argument is the current
    variable values, i.e. control amplitudes, passed as
    a flat array. Hence these are reshaped as [nTimeslots, n_ctrls]
    and then used to update the stored ctrl values (if they have changed)

    The error is checked against the target, and the optimisation is
    terminated if the target has been achieved.
    """
    self.num_fid_func_calls += 1
    # *** update stats ***
    if self.stats is not None:
        self.stats.num_fidelity_func_calls = self.num_fid_func_calls
        if self.log_level <= logging.DEBUG:
            logger.debug(
                "fidelity error call {}".format(
                    self.stats.num_fidelity_func_calls
                )
            )

    amps = self._get_ctrl_amps(args[0].copy())
    self.dynamics.update_ctrl_amps(amps)

    tc = self.termination_conditions
    err = self.dynamics.fid_computer.get_fid_err()

    if self.iter_summary:
        self.iter_summary.fid_func_call_num = self.num_fid_func_calls
        self.iter_summary.fid_err = err

    if self.dump and self.dump.dump_fid_err:
        self.dump.update_fid_err_log(err)

    # handeled in minimizer callback
    """
    if err <= tc.fid_err_targ:
        raise errors.GoalAchievedTerminate(err)
    
    if self.num_fid_func_calls > tc.max_fid_func_calls:
        raise errors.MaxFidFuncCallTerminate()
    """
    return err

# overwrite

def fid_err_grad_wrapper(self, *args):
    """
    Get the gradient of the fidelity error with respect to all of the
    variables, i.e. the ctrl amplidutes in each timeslot

    This is called by generic optimisation algorithm as the gradients of
    func to the minimised wrt the variables. The argument is the current
    variable values, i.e. control amplitudes, passed as
    a flat array. Hence these are reshaped as [nTimeslots, n_ctrls]
    and then used to update the stored ctrl values (if they have changed)

    Although the optimisation algorithms have a check within them for
    function convergence, i.e. local minima, the sum of the squares
    of the normalised gradient is checked explicitly, and the
    optimisation is terminated if this is below the min_gradient_norm
    condition
    """
    # *** update stats ***
    self.num_grad_func_calls += 1
    if self.stats is not None:
        self.stats.num_grad_func_calls = self.num_grad_func_calls
        if self.log_level <= logging.DEBUG:
            logger.debug(
                "gradient call {}".format(self.stats.num_grad_func_calls)
            )
    amps = self._get_ctrl_amps(args[0].copy())
    self.dynamics.update_ctrl_amps(amps)
    fid_comp = self.dynamics.fid_computer
    # gradient_norm_func is a pointer to the function set in the config
    # that returns the normalised gradients
    grad = fid_comp.get_fid_err_gradient()

    if self.iter_summary:
        self.iter_summary.grad_func_call_num = self.num_grad_func_calls
        self.iter_summary.grad_norm = fid_comp.grad_norm

    if self.dump:
        if self.dump.dump_grad_norm:
            self.dump.update_grad_norm_log(fid_comp.grad_norm)

        if self.dump.dump_grad:
            self.dump.update_grad_log(grad)

    # handeled in minimizer callback
    """
    tc = self.termination_conditions
    if fid_comp.grad_norm < tc.min_gradient_norm:
        raise errors.GradMinReachedTerminate(fid_comp.grad_norm)
    """
    return grad.flatten()


class Multi_CRAB():
    """
    Composite class for multiple GOAT instances
    to optimize multiple objectives simultaneously
    """

    def __init__(self, crab_optimizer, objectives, time_interval, time_options, pulse_options,
                 alg_kwargs, guess_params, **integrator_kwargs):

        self.crabs = []

        for obj in objectives:
            crab = copy.deepcopy(crab_optimizer)
            crab.fid_err_func_wrapper = types.MethodType(fid_err_func_wrapper, crab)
            # Stack for each objective
            self.crabs.append(crab)

        self.mean_infid = None

    def goal_fun(self, params):
        """
        Calculates the mean infidelity over all objectives
        """
        infid_sum = 0

        for crab in self.crabs:  # TODO: parallelize
            infid = crab.fid_err_func_wrapper(params)
            infid_sum += infid

        self.mean_infid = np.mean(infid_sum)
        return self.mean_infid

    def grad_fun(self, params):
        """
        Calculates the sum of gradients over all objectives
        """
        grads = 0

        for c in self.crabs:
            grads += c.fid_err_grad_wrapper(params)

        return grads