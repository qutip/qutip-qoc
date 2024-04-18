"""
This module provides an interface to the CRAB optimization algorithm in qutip-qtrl.
It defines the _CRAB class, which uses a qutip_qtrl.optimizer.Optimizer object
to store the control problem and calculate the fidelity error function and its gradient
with respect to the control parameters, according to the CRAB algorithm.
"""

import qutip_qtrl.logging_utils as logging
import copy

logger = logging.get_logger()


class _CRAB:
    """
    Class to interface with the CRAB optimization algorithm in qutip-qtrl.
    It has an attribute `qtrl` that is a `qutip_qtrl.optimizer.Optimizer` object
    for storing the control problem and calculating the fidelity error function
    and its gradient wrt the control parameters, according to the CRAB algorithm.
    The class does only provide the infidelity method, as the CRAB algorithm is
    not a gradient-based optimization.
    """

    def __init__(self, qtrl_optimizer):
        self._qtrl = copy.deepcopy(qtrl_optimizer)
        self.gradient = None

    def infidelity(self, *args):
        """
        This method is adapted from the original
        `qutip_qtrl.optimizer.Optimizer.fid_err_func_wrapper`

        Get the fidelity error achieved using the ctrl amplitudes passed
        in as the first argument.

        This is called by generic optimisation algorithm as the
        func to the minimised. The argument is the current
        variable values, i.e. control amplitudes, passed as
        a flat array. Hence these are reshaped as [nTimeslots, n_ctrls]
        and then used to update the stored ctrl values (if they have changed)
        """
        self._qtrl.num_fid_func_calls += 1
        # *** update stats ***
        if self._qtrl.stats is not None:
            self._qtrl.stats.num_fidelity_func_calls = self._qtrl.num_fid_func_calls
            if self._qtrl.log_level <= logging.DEBUG:
                logger.debug(
                    "fidelity error call {}".format(
                        self._qtrl.stats.num_fidelity_func_calls
                    )
                )

        amps = self._qtrl._get_ctrl_amps(args[0].copy())
        self._qtrl.dynamics.update_ctrl_amps(amps)

        err = self._qtrl.dynamics.fid_computer.get_fid_err()

        if self._qtrl.iter_summary:
            self._qtrl.iter_summary.fid_func_call_num = self._qtrl.num_fid_func_calls
            self._qtrl.iter_summary.fid_err = err

        if self._qtrl.dump and self._qtrl.dump.dump_fid_err:
            self._qtrl.dump.update_fid_err_log(err)

        return err
