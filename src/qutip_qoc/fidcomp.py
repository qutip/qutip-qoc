"""
Fidelity computations for quantum optimal control.
Implements PSU, SU, and TRACEDIFF fidelity types.
"""

import qutip as qt
import jax.numpy as jnp
import numpy as np
from typing import List, Union

__all__ = ["FidelityComputer"]

class FidelityComputer:
    """
    Computes fidelity between initial and target states/unitaries/maps.
    
    Parameters
    ----------
    fid_type : str
        Type of fidelity to compute. Options are:
        - 'PSU': Phase-insensitive state/unitary fidelity
        - 'SU': Phase-sensitive state/unitary fidelity
        - 'TRACEDIFF': Trace difference for maps
    """
    
    def __init__(self, fid_type: str = "PSU"):
        self.fid_type = fid_type.upper()
        if self.fid_type not in ["PSU", "SU", "TRACEDIFF"]:
            raise ValueError(f"Unknown fidelity type: {fid_type}")
    
    def compute_fidelity(
        self, 
        initial: Union[qt.Qobj, List[qt.Qobj]], 
        target: Union[qt.Qobj, List[qt.Qobj]], 
        evolved: Union[qt.Qobj, List[qt.Qobj]]
    ) -> float:
        """
        Compute fidelity between evolved and target states/unitaries/maps.
        
        Parameters
        ----------
        initial : Qobj or list of Qobj
            Initial state(s)/unitary(ies)/map(s)
        target : Qobj or list of Qobj
            Target state(s)/unitary(ies)/map(s)
        evolved : Qobj or list of Qobj
            Evolved state(s)/unitary(ies)/map(s)
            
        Returns
        -------
        float
            Fidelity between evolved and target
        """
        if isinstance(initial, list):
            return np.mean([self._single_fidelity(i, t, e) 
                          for i, t, e in zip(initial, target, evolved)])
        return self._single_fidelity(initial, target, evolved)
    
    def _single_fidelity(
        self,
        initial: qt.Qobj,
        target: qt.Qobj,
        evolved: qt.Qobj
    ) -> float:
        """
        Compute fidelity for a single initial/target/evolved pair.
        """
        if self.fid_type in ["PSU", "SU"]:
            if evolved.type == "oper" and target.type == "oper":
                return self._unitary_fidelity(evolved, target)
            elif evolved.type == "ket" and target.type == "ket":
                return self._state_fidelity(evolved, target)
            else:
                raise TypeError(f"For {self.fid_type} fidelity, evolved and target must both be states or unitaries")
        elif self.fid_type == "TRACEDIFF":
            # For TRACEDIFF, we can handle both superoperators and regular operators
            if evolved.type in ["super", "oper"] and target.type in ["super", "oper"]:
                return self._map_fidelity(evolved, target)
            else:
                raise TypeError("For TRACEDIFF fidelity, evolved and target must be operators or superoperators")
        else:
            raise ValueError(f"Unknown fidelity type: {self.fid_type}")
    
    def _state_fidelity(self, evolved: qt.Qobj, target: qt.Qobj) -> float:
        """
        Compute state fidelity between evolved and target states.
        """
        # Calculate the overlap
        overlap = target.dag() * evolved
        
        # Handle both cases where overlap might be a Qobj or complex number
        if isinstance(overlap, qt.Qobj):
            overlap_value = overlap.full().item()  # Extract complex number from Qobj
        else:
            overlap_value = overlap  # Already a complex number
        
        if self.fid_type == "PSU":
            # Phase-insensitive state fidelity (absolute value of overlap)
            fid = jnp.abs(overlap_value) ** 2
        else:  # SU
            # Phase-sensitive state fidelity
            fid = (overlap_value ** 2).real  # Take real part to ensure float return
        
        return jnp.float64(fid)

    def _unitary_fidelity(self, evolved: qt.Qobj, target: qt.Qobj) -> float:
        """
        Compute fidelity between evolved and target unitaries.
        """
        d = evolved.shape[0]

        if hasattr(evolved.data, '_jxa'):
            evolved_mat = evolved.data._jxa
            target_mat = target.data._jxa
        else:
            evolved_mat = jnp.array(evolved.full())
            target_mat = jnp.array(target.full())

        # Compute V†U (conjugate transpose of target multiplied by evolved)
        overlap = jnp.trace(jnp.matmul(jnp.conj(jnp.transpose(target_mat)), evolved_mat))

        if self.fid_type == "PSU":
            fid = (jnp.abs(overlap) / d) ** 2
        else:  # SU
            fid = (overlap / d) ** 2

        return jnp.real(fid)

    
    def _map_fidelity(self, evolved: qt.Qobj, target: qt.Qobj) -> float:
        """
        Compute trace difference fidelity between evolved and target maps.
        Handles both superoperators and regular operators using JAX-compatible operations.
        """
        if evolved.type == "super" and target.type == "super":
            # Superoperator case
            d = int(np.sqrt(evolved.shape[0]))  # Hilbert space dimension
        elif evolved.type == "oper" and target.type == "oper":
            # Regular operator case
            d = evolved.shape[0]
        else:
            raise TypeError("Both evolved and target must be of the same type (super or oper)")
        
        # Extract JAX arrays from Qobjs
        if hasattr(evolved.data, '_jxa'):
            evolved_mat = evolved.data._jxa
            target_mat = target.data._jxa
        else:
            evolved_mat = jnp.array(evolved.full())
            target_mat = jnp.array(target.full())
        
        # Calculate difference
        diff_mat = target_mat - evolved_mat
        
        # Calculate trace norm (sum of singular values)
        # This avoids using Schur decomposition which JAX can't differentiate
        s = jnp.linalg.svd(diff_mat, compute_uv=False)
        trace_norm = jnp.sum(s)
        
        fid = 1 - trace_norm / (2 * jnp.sqrt(d))
        return fid

    def compute_infidelity(
        self, 
        initial: Union[qt.Qobj, List[qt.Qobj]], 
        target: Union[qt.Qobj, List[qt.Qobj]], 
        evolved: Union[qt.Qobj, List[qt.Qobj]]
    ) -> float:
        """
        Compute infidelity (1 - fidelity) between evolved and target.
        """
        return 1 - self.compute_fidelity(initial, target, evolved)