"""
This module provides an interface to the GRAPE optimization algorithm in qutip-qtrl.
It defines the _GRAPE class, which uses a qutip_qtrl.optimizer.Optimizer object
to store the control problem and calculate the fidelity error function and its gradient
with respect to the control parameters, according to the CRAB algorithm.
"""

import qutip_qtrl.logging_utils as logging
import copy


logger = logging.get_logger()


class _GRAPE:
    """
    Class to interface with the CRAB optimization algorithm in qutip-qtrl.
    It has an attribute `qtrl` that is a `qutip_qtrl.optimizer.Optimizer` object
    for storing the control problem and calculating the fidelity error function
    and its gradient wrt the control parameters, according to the GRAPE algorithm.
    The class does provide both infidelity and gradient methods.
    """

    def __init__(self, qtrl_optimizer):
        self._qtrl = copy.deepcopy(qtrl_optimizer)

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

    def gradient(self, *args):
        """
        This method is adapted from the original
        `qutip_qtrl.optimizer.Optimizer.fid_err_grad_wrapper`

        Get the gradient of the fidelity error with respect to all of the
        variables, i.e. the ctrl amplidutes in each timeslot

        This is called by generic optimisation algorithm as the gradients of
        func to the minimised wrt the variables. The argument is the current
        variable values, i.e. control amplitudes, passed as
        a flat array. Hence these are reshaped as [nTimeslots, n_ctrls]
        and then used to update the stored ctrl values (if they have changed)
        """
        # *** update stats ***
        self._qtrl.num_grad_func_calls += 1
        if self._qtrl.stats is not None:
            self._qtrl.stats.num_grad_func_calls = self._qtrl.num_grad_func_calls
            if self._qtrl.log_level <= logging.DEBUG:
                logger.debug(
                    "gradient call {}".format(self._qtrl.stats.num_grad_func_calls)
                )
        amps = self._qtrl._get_ctrl_amps(args[0].copy())
        self._qtrl.dynamics.update_ctrl_amps(amps)
        fid_comp = self._qtrl.dynamics.fid_computer
        # gradient_norm_func is a pointer to the function set in the config
        # that returns the normalised gradients
        grad = fid_comp.get_fid_err_gradient()

        if self._qtrl.iter_summary:
            self._qtrl.iter_summary.grad_func_call_num = self._qtrl.num_grad_func_calls
            self._qtrl.iter_summary.grad_norm = fid_comp.grad_norm

        if self._qtrl.dump:
            if self._qtrl.dump.dump_grad_norm:
                self._qtrl.dump.update_grad_norm_log(fid_comp.grad_norm)

            if self._qtrl.dump.dump_grad:
                self._qtrl.dump.update_grad_log(grad)

        return grad.flatten()
