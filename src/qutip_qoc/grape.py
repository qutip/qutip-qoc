import qutip_qtrl.logging_utils as logging
import copy

__all__ = ["GRAPE"]

logger = logging.get_logger()


class GRAPE:
    def __init__(self, qtrl_optimizer):
        self.qtrl = copy.deepcopy(qtrl_optimizer)

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
        self.qtrl.num_fid_func_calls += 1
        # *** update stats ***
        if self.qtrl.stats is not None:
            self.qtrl.stats.num_fidelity_func_calls = self.qtrl.num_fid_func_calls
            if self.qtrl.log_level <= logging.DEBUG:
                logger.debug(
                    "fidelity error call {}".format(
                        self.qtrl.stats.num_fidelity_func_calls
                    )
                )

        amps = self.qtrl._get_ctrl_amps(args[0].copy())
        self.qtrl.dynamics.update_ctrl_amps(amps)

        err = self.qtrl.dynamics.fid_computer.get_fid_err()

        if self.qtrl.iter_summary:
            self.qtrl.iter_summary.fid_func_call_num = self.qtrl.num_fid_func_calls
            self.qtrl.iter_summary.fid_err = err

        if self.qtrl.dump and self.qtrl.dump.dump_fid_err:
            self.qtrl.dump.update_fid_err_log(err)

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
        self.qtrl.num_grad_func_calls += 1
        if self.qtrl.stats is not None:
            self.qtrl.stats.num_grad_func_calls = self.qtrl.num_grad_func_calls
            if self.qtrl.log_level <= logging.DEBUG:
                logger.debug(
                    "gradient call {}".format(self.qtrl.stats.num_grad_func_calls)
                )
        amps = self.qtrl._get_ctrl_amps(args[0].copy())
        self.qtrl.dynamics.update_ctrl_amps(amps)
        fid_comp = self.qtrl.dynamics.fid_computer
        # gradient_norm_func is a pointer to the function set in the config
        # that returns the normalised gradients
        grad = fid_comp.get_fid_err_gradient()

        if self.qtrl.iter_summary:
            self.qtrl.iter_summary.grad_func_call_num = self.qtrl.num_grad_func_calls
            self.qtrl.iter_summary.grad_norm = fid_comp.grad_norm

        if self.qtrl.dump:
            if self.qtrl.dump.dump_grad_norm:
                self.qtrl.dump.update_grad_norm_log(fid_comp.grad_norm)

            if self.qtrl.dump.dump_grad:
                self.qtrl.dump.update_grad_log(grad)

        return grad.flatten()
