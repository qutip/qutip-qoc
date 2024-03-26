import numpy as np
import qutip_qtrl.logging_utils as logging
import qutip_qtrl.optimizer as opt
from qutip_qtrl.errors import GoalAchievedTerminate, MaxFidFuncCallTerminate
import types
import copy

__all__ = ["CRAB", "Multi_CRAB"]

logger = logging.get_logger()


class CRAB(opt.OptimizerCrab):
    def __init__(self, cfg, dyn, params, termination_conditions):
        super().__init__(cfg, dyn, params)
        self.init_optim(termination_conditions)


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
                "fidelity error call {}".format(self.stats.num_fidelity_func_calls)
            )

    amps = self._get_ctrl_amps(args[0].copy())
    self.dynamics.update_ctrl_amps(amps)

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


class Multi_CRAB:
    """
    Composite class for multiple GOAT instances
    to optimize multiple objectives simultaneously
    """

    grad_fun = None

    def __init__(
        self,
        qtrl_optimizers,
    ):
        self.crabs = []
        for optim in qtrl_optimizers:
            crab = copy.deepcopy(optim)
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
            try:
                infid = crab.fid_err_func_wrapper(params)
            except (GoalAchievedTerminate, MaxFidFuncCallTerminate):
                pass
            except Exception as ex:
                raise ex
            infid_sum += infid

        self.mean_infid = np.mean(infid_sum)
        return self.mean_infid
