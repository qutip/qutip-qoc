"""
This module provides an interface to the GRAPE optimization algorithm in qutip-qtrl.
It defines the _GRAPE class, which uses a qutip_qtrl.optimizer.Optimizer object
to store the control problem and calculate the fidelity error function and its gradient
with respect to the control parameters, using our FidelityComputer class.
"""

import qutip_qtrl.logging_utils as logging
import copy
import numpy as np
from qutip_qoc.fidcomp import FidelityComputer

logger = logging.get_logger()


class _GRAPE:
    """
    Class to interface with the GRAPE optimization algorithm.
    Uses our FidelityComputer for fidelity calculations while maintaining
    the GRAPE-specific gradient calculations.
    """

    def __init__(self, qtrl_optimizer, fid_type="PSU"):
        self._qtrl = copy.deepcopy(qtrl_optimizer)
        self.fidcomp = FidelityComputer(fid_type)
        
        # Extract initial and target from dynamics
        self.initial = self._get_initial()
        self.target = self._get_target()

    def _get_initial(self):
        """Extract initial state/unitary/map from dynamics"""
        if hasattr(self._qtrl.dynamics, 'initial'):
            return self._qtrl.dynamics.initial
        elif hasattr(self._qtrl.dynamics, 'initial_state'):
            return self._qtrl.dynamics.initial_state
        return None
    
    def _get_target(self):
        """Extract target state/unitary/map from dynamics"""
        if hasattr(self._qtrl.dynamics, 'target'):
            return self._qtrl.dynamics.target
        elif hasattr(self._qtrl.dynamics, 'target_state'):
            return self._qtrl.dynamics.target_state
        return None

    def _get_evolved(self):
        """Get the evolved state/unitary/map from dynamics"""
        evolved = None
        
        # Try to get the evolved state/unitary/map using different methods
        if hasattr(self._qtrl.dynamics, 'get_final_state'):
            evolved = self._qtrl.dynamics.get_final_state()
        elif hasattr(self._qtrl.dynamics, 'get_final_unitary'):
            evolved = self._qtrl.dynamics.get_final_unitary()
        elif hasattr(self._qtrl.dynamics, 'get_final_super'):
            evolved = self._qtrl.dynamics.get_final_super()
        
        # If evolved is still None, log a warning
        if evolved is None:
            logger.warning("Could not get evolved state/unitary/map from dynamics. Check if dynamics has appropriate methods.")
            
        return evolved

    def infidelity(self, *args):
        """
        Get the fidelity error using our FidelityComputer.
        Maintains all original logging and statistics functionality.
        """
        self._qtrl.num_fid_func_calls += 1
        
        # Update stats
        if self._qtrl.stats is not None:
            self._qtrl.stats.num_fidelity_func_calls = self._qtrl.num_fid_func_calls
            if self._qtrl.log_level <= logging.DEBUG:
                logger.debug(
                    f"fidelity error call {self._qtrl.stats.num_fidelity_func_calls}"
                )

        # Update control amplitudes
        amps = self._qtrl._get_ctrl_amps(args[0].copy())
        self._qtrl.dynamics.update_ctrl_amps(amps)

        # Calculate fidelity using our FidelityComputer
        evolved = self._get_evolved()
        
        # Check if evolved is None and handle appropriately
        if evolved is None:
            logger.error("Evolved state/unitary/map is None. Cannot compute fidelity.")
            # Return a high error value to indicate failure
            err = 1.0  # Maximum infidelity
        else:
            err = self.fidcomp.compute_infidelity(self.initial, self.target, evolved)

        # Maintain logging and statistics
        if self._qtrl.iter_summary:
            self._qtrl.iter_summary.fid_func_call_num = self._qtrl.num_fid_func_calls
            self._qtrl.iter_summary.fid_err = err

        if self._qtrl.dump and self._qtrl.dump.dump_fid_err:
            self._qtrl.dump.update_fid_err_log(err)

        return err

    def gradient(self, *args):
        """
        Get the gradient of the fidelity error.
        Still uses qtrl's gradient calculation as it's GRAPE-specific,
        but uses our FidelityComputer for the fidelity part.
        """
        # Update stats
        self._qtrl.num_grad_func_calls += 1
        if self._qtrl.stats is not None:
            self._qtrl.stats.num_grad_func_calls = self._qtrl.num_grad_func_calls
            if self._qtrl.log_level <= logging.DEBUG:
                logger.debug(f"gradient call {self._qtrl.stats.num_grad_func_calls}")

        # Update control amplitudes
        amps = self._qtrl._get_ctrl_amps(args[0].copy())
        self._qtrl.dynamics.update_ctrl_amps(amps)

        # Calculate gradient (still using qtrl's implementation as it's GRAPE-specific)
        fid_comp = self._qtrl.dynamics.fid_computer
        
        # Verify that fid_comp is available
        if not hasattr(self._qtrl.dynamics, 'fid_computer') or self._qtrl.dynamics.fid_computer is None:
            logger.error("Fidelity computer not available in dynamics. Cannot compute gradient.")
            # Return a zero gradient to avoid further errors
            return np.zeros_like(args[0])
            
        grad = fid_comp.get_fid_err_gradient()

        # Maintain logging and statistics
        if self._qtrl.iter_summary:
            self._qtrl.iter_summary.grad_func_call_num = self._qtrl.num_grad_func_calls
            self._qtrl.iter_summary.grad_norm = fid_comp.grad_norm

        if self._qtrl.dump:
            if self._qtrl.dump.dump_grad_norm:
                self._qtrl.dump.update_grad_norm_log(fid_comp.grad_norm)
            if self._qtrl.dump.dump_grad:
                self._qtrl.dump.update_grad_log(grad)

        return grad.flatten()