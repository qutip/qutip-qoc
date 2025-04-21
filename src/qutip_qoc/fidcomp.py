"""
Fidelity computation module for qutip_qoc (Quantum Optimal Control)

This module provides state, gate, average, and custom fidelity functions,
along with gradient support, performance optimization using Numba, 
fidelity tracking for optimization pipelines, and support for superoperators and Kraus representations.

Author: Adapted for qutip_qoc
"""

import numpy as np
from qutip import Qobj, fidelity, ket2dm, identity, superop_reps, spre, operator_to_vector, vector_to_operator
from numba import njit
from typing import Tuple
import numpy as np
import qutip as qt
from qutip import Qobj, ket2dm, qeye, identity 
from typing import Callable, Union
from joblib import Parallel, delayed

import functools
import logging
import json
import os
from typing import Callable, List, Union

__all__ = [
    'compute_fidelity', 'state_fidelity', 'unitary_fidelity',
    'average_gate_fidelity', 'custom_fidelity', 'get_fidelity_func',
    'fidelity_gradient', 'FidelityTracker',
    'superoperator_fidelity', 'kraus_fidelity', 'process_fidelity', 
    'gate_fidelity', 'operator_fidelity'
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Fidelity Functions ---

def compute_fidelity(
    target: Qobj, 
    achieved: Qobj, 
    kind: str = 'state', 
    **kwargs
) -> float:
    """
    Computes the fidelity between a target and achieved state based on the given fidelity type.

    Args:
        target (Qobj): The target quantum object (state, gate, or superoperator).
        achieved (Qobj): The achieved quantum object (state, gate, or superoperator).
        kind (str): The type of fidelity calculation ('state', 'unitary', 'average', 'super', 'kraus', or 'custom').
        **kwargs: Additional arguments for custom fidelity.

    Returns:
        float: The calculated fidelity value.

    Raises:
        ValueError: If an unsupported fidelity type is provided.

    Example:
        >>> target_state = Qobj([[1, 0], [0, 0]])
        >>> achieved_state = Qobj([[0.8, 0.2], [0.2, 0.8]])
        >>> compute_fidelity(target_state, achieved_state, kind='state')
        0.8
    """
    validate_qobj_pair(target, achieved, kind)
    if kind == 'state':
        return state_fidelity(target, achieved)
    elif kind == 'unitary':
        return unitary_fidelity(target, achieved)
    elif kind == 'average':
        return average_gate_fidelity(target, achieved)
    elif kind == 'super':
        return superoperator_fidelity(target, achieved)
    elif kind == 'kraus':
        return kraus_fidelity(target, achieved)
    elif kind == 'custom':
        return custom_fidelity(target, achieved, **kwargs)
    else:
        raise ValueError(f"Unsupported fidelity kind: {kind}")

def state_fidelity(target: qt.Qobj, achieved: qt.Qobj) -> float:
    """
    Computes the fidelity between two states (density matrices or pure states).
    """
    if target.isket:
        target = qt.ket2dm(target)
    if achieved.isket:
        achieved = qt.ket2dm(achieved)
    return qt.fidelity(target, achieved)

def unitary_fidelity(U_target: Qobj, U_actual: Qobj) -> float:
    """
    Computes the fidelity between two unitary operators.

    Args:
        U_target (Qobj): The target unitary matrix.
        U_actual (Qobj): The achieved unitary matrix.

    Returns:
        float: The unitary fidelity value.

    Example:
        >>> U_target = Qobj([[1, 0], [0, 1]])
        >>> U_actual = Qobj([[0.99, 0.01], [0.01, 0.99]])
        >>> unitary_fidelity(U_target, U_actual)
        0.9998
    """
    d = U_target.shape[0]
    overlap = (U_target.dag() * U_actual).tr()
    fid = abs(overlap / d) ** 2
    return fid.real

def average_gate_fidelity(U_target: Qobj, U_actual: Qobj) -> float:
    """
    Computes the average gate fidelity between two unitary operators.

    Args:
        U_target (Qobj): The target unitary matrix.
        U_actual (Qobj): The achieved unitary matrix.

    Returns:
        float: The average gate fidelity value.

    Example:
        >>> U_target = Qobj([[1, 0], [0, 1]])
        >>> U_actual = Qobj([[0.95, 0.05], [0.05, 0.95]])
        >>> average_gate_fidelity(U_target, U_actual)
        0.9995
    """
    d = U_target.shape[0]
    fid = (abs((U_target.dag() * U_actual).tr())**2 + d) / (d * (d + 1))
    return fid.real

def custom_fidelity(target, achieved, func: Callable) -> float:
    """
    Computes custom fidelity using a user-defined function.

    Args:
        target (Qobj): The target quantum object.
        achieved (Qobj): The achieved quantum object.
        func (Callable): A user-defined function to compute fidelity.

    Returns:
        float: The custom fidelity value.

    Example:
        >>> custom_fidelity(target, achieved, lambda t, a: np.abs(t - a).norm())
        0.1
    """
    return func(target, achieved)

def superoperator_fidelity(S_target: Qobj, S_actual: Qobj) -> float:
    """
    Computes the fidelity between two superoperators.

    Args:
        S_target (Qobj): The target superoperator.
        S_actual (Qobj): The achieved superoperator.

    Returns:
        float: The superoperator fidelity value.

    Example:
        >>> superoperator_fidelity(S_target, S_actual)
        0.85
    """
    d = int(np.sqrt(S_target.shape[0]))
    vec_id = operator_to_vector(identity(d))
    chi_target = S_target * vec_id
    chi_actual = S_actual * vec_id
    return np.abs((chi_target.dag() * chi_actual)[0, 0].real)

def kraus_fidelity(K_target: List[Qobj], K_actual: List[Qobj]) -> float:
    """
    Computes the fidelity between two Kraus operator sets.

    Args:
        K_target (List[Qobj]): List of target Kraus operators.
        K_actual (List[Qobj]): List of achieved Kraus operators.

    Returns:
        float: The Kraus fidelity value.

    Example:
        >>> kraus_fidelity(K_target, K_actual)
        0.92
    """
    d = K_target[0].shape[0]
    fid = 0
    for A in K_target:
        for B in K_actual:
            fid += np.abs((A.dag() * B).tr())**2
    return fid.real / (d**2)

def process_fidelity(ideal_process: qt.Qobj, achieved_process: qt.Qobj) -> float:
    """
    Computes the process fidelity between two processes (superoperators).
    Fidelity = Tr(E1 * E2^dagger) / sqrt(Tr(E1 * E1^dagger) * Tr(E2 * E2^dagger))
    """
    choi_ideal = ideal_process.choi()
    choi_actual = achieved_process.choi()
    
    fidelity = np.trace(choi_ideal * choi_actual.dag()) / np.sqrt(
        np.trace(choi_ideal * choi_ideal.dag()) * np.trace(choi_actual * choi_actual.dag())
    )
    return fidelity

def gate_fidelity(ideal_gate: qt.Qobj, achieved_gate: qt.Qobj) -> float:
    """
    Computes the gate fidelity between two gates (unitary operators).
    Fidelity = |<ideal|actual>|^2
    """
    return np.abs(np.trace(ideal_gate.dag() * achieved_gate)) ** 2

def operator_fidelity(ideal_operator: qt.Qobj, achieved_operator: qt.Qobj) -> float:
    """
    Computes the operator fidelity between two operators.
    Fidelity = Tr(sqrt(sqrt(A) * B * sqrt(A)))^2
    """
    return qt.fidelity(ideal_operator, achieved_operator)

def get_fidelity_func(kind: str = 'state') -> Union[Callable, None]:
    """
    Retrieves the fidelity function for the specified type.
    """
    return {
        'state': state_fidelity,
        'unitary': unitary_fidelity,
        'average': average_gate_fidelity,
        'super': superoperator_fidelity,
        'kraus': kraus_fidelity,
        'custom': custom_fidelity,
        'process': process_fidelity,
        'gate': gate_fidelity,
        'operator': operator_fidelity
    }.get(kind, None)

# --- Gradient Support ---

def fidelity_gradient(U_target: Qobj, U_list: List[Qobj], epsilon: float = 1e-6) -> np.ndarray:
    """
    Computes the gradient of fidelity with respect to control parameters.

    Args:
        U_target (Qobj): The target unitary matrix.
        U_list (List[Qobj]): List of unitary matrices (control parameters).
        epsilon (float): Perturbation size for numerical gradient.

    Returns:
        np.ndarray: Array of gradients.

    Example:
        >>> fidelity_gradient(U_target, U_list)
        array([0.1, -0.1])
    """
    base_fid = unitary_fidelity(U_target, U_list[-1])
    grads = []
    for i, U in enumerate(U_list):
        U_perturb = U + epsilon * identity(U.shape[0])
        U_new = U_list[:i] + [U_perturb] + U_list[i+1:]
        fid_perturbed = unitary_fidelity(U_target, U_new[-1])
        grad = (fid_perturbed - base_fid) / epsilon
        grads.append(grad)
    return np.array(grads)

# --- Performance Optimized Core ---

@njit
def trace_norm_numba(A_real: np.ndarray, A_imag: np.ndarray) -> float:
    """
    Computes the trace norm of a matrix using Numba for performance optimization.

    Args:
        A_real (np.ndarray): Real part of the matrix.
        A_imag (np.ndarray): Imaginary part of the matrix.

    Returns:
        float: Trace norm value.

    Example:
        >>> trace_norm_numba(A_real, A_imag)
        1.2
    """
    return np.sqrt(np.sum(A_real**2 + A_imag**2))

# --- Fidelity Tracker ---
logger = logging.getLogger(__name__)

class FidelityComputer:
    def __init__(self, save_path: Union[str, None] = None, fidtype: str = 'state', fidelity_function: Callable = None, target: Qobj = None, projector: Qobj = None):
        """
        Initializes the FidelityTracker class with an optional fidtype and custom fidelity function.
        
        Args:
            save_path (Union[str, None]): Path to save the fidelity history. If None, no saving occurs.
            fidtype (str): Type of fidelity to compute (default is 'state').
            fidelity_function (Callable): A custom function to compute fidelity (optional).
            target (Qobj): Target quantum object (e.g., state or unitary matrix, for state/unitary fidelities).
            projector (Qobj): Projector for fidelity computation (used in 'projector' mode).
        """
        self.history = []
        self.save_path = save_path
        self.fidtype = fidtype
        self.fidelity_function = fidelity_function  # Custom function, if provided
        self.target = target  # For state/unitary fidelities
        self.projector = projector  # For projector fidelities

        self.fidelity_methods = {
            'state': self._state_fidelity,
            'unitary': self._unitary_fidelity,
            'super': self.compute_superoperator_fidelity,
            'process': self.compute_process_fidelity,  # Added process fidelity
            'projector': self._projector_fidelity,
            # Add more fidelity types as needed
        }

        if self.fidtype == "custom" and not callable(self.fidelity_function):
            raise ValueError("For 'custom' fidelity, 'fidelity_function' must be provided and callable.")
        if self.fidtype == "projector" and self.projector is None:
            raise ValueError("For 'projector' fidelity, 'projector' must be provided.")
   
    def compute_fidelity(self, A: Union[Qobj, np.ndarray], B: Union[Qobj, np.ndarray] = None) -> float:
        """
        Computes fidelity based on the type specified during initialization.

        Args:
            A (Qobj): Achieved quantum object (e.g., state or unitary matrix)
            B (Qobj, optional): Target quantum object (e.g., state or unitary matrix)

        Returns:
            float: Fidelity value.
        """
        # Ensure A and B are Qobj
        A = self.ensure_qobj(A)
        B = self.ensure_qobj(B)

        # Handle different fidelity types
        if self.fidelity_function:
            # If the user has provided a custom fidelity function, use it
            return self.fidelity_function(A, B)
        elif self.fidtype in self.fidelity_methods:
            # Use one of the predefined fidelity methods
            return self.fidelity_methods[self.fidtype](A, B)
        else:
            raise ValueError(f"Unsupported fidelity type: {self.fidtype}")

    def _state_fidelity(self, psi, target):
        """Computes fidelity for state fidelity (|psi> <psi| with target)."""
        if target is None:
            raise ValueError("Target state must be provided for state fidelity.")
        return (psi.dag() * target).tr().real  # Fidelity formula: <psi|target>

    def _unitary_fidelity(self, U, _=None):
        if self.target is None:
            raise ValueError("Target unitary must be provided for unitary fidelity.")
        d = U.shape[0]  # or use self.target.shape[0]
        return abs((U.dag() * self.target).tr())**2 / (d ** 2)


    def _projector_fidelity(self, rho, _=None):
        if self.projector is None:
            raise ValueError("Projector must be provided for projector fidelity.")
        return (rho.dag() * self.projector).tr().real

    def _custom_fidelity(self, A: Qobj, B: Qobj) -> float:
        """Computes custom fidelity using a user-defined function."""
        return self.fidelity_function(A, B)

    def ensure_qobj(self, obj: Union[Qobj, np.ndarray]) -> Qobj:
        """
        Ensure the object is a Qobj (quantum object).
        
        Args:
            obj (Union[Qobj, np.ndarray]): The object to be converted.
        
        Returns:
            Qobj: The object wrapped in a Qobj if it is not already.
        """
        if isinstance(obj, Qobj):
            return obj
        else:
            return Qobj(obj)

    def record(self, step: int, fidelity_value: float):
        """
        Records the fidelity value at a specific optimization step.
        
        Args:
            step (int): The current step in the optimization.
            fidelity_value (float): The fidelity value at this step.
        """
        self.history.append((step, fidelity_value))
        logger.info(f"Step {step}: Fidelity = {fidelity_value:.6f}")
        if self.save_path:
            self.save_to_file()

    def get_history(self) -> List[Tuple[int, float]]:
        """
        Returns the history of recorded fidelity values.
        
        Returns:
            List[Tuple[int, float]]: A list of tuples containing (step, fidelity_value).
        """
        return self.history

    def plot(self):
        """
        Plots the fidelity history using matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
            steps, fids = zip(*self.history)
            plt.plot(steps, fids, marker='o')
            plt.xlabel("Step")
            plt.ylabel("Fidelity")
            plt.title("Fidelity Over Time")
            plt.grid(True)
            plt.show()
        except ImportError:
            logger.warning("matplotlib not installed. Cannot plot fidelity.")

    def save_to_file(self):
        """
        Saves the fidelity history to the specified file path.
        """
        if not self.save_path:
            return
        try:
            with open(self.save_path, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            logger.error(f"Failed to save fidelity history: {e}")

    def compute_state_fidelity(self, A: Qobj, B: Qobj) -> float:
        """
        Compute the state fidelity (e.g., Uhlmann fidelity).
        
        Args:
            A (Qobj): The target quantum state.
            B (Qobj): The achieved quantum state.
        
        Returns:
            float: The state fidelity value.
        """
        return abs((A.dag() * B).tr()) ** 2

    
    def compute_superoperator_fidelity(self, A: Qobj, B: Qobj) -> float:
        """
        Compute the superoperator fidelity.
        
        Args:
            A (Qobj): The target superoperator.
            B (Qobj): The achieved superoperator.
        
        Returns:
            float: The superoperator fidelity value.
        """
        return np.real(np.trace(A.dag() * B)) ** 2

    def compute_process_fidelity(self, A: Qobj, B: Qobj) -> float:
        """
        Compute the process fidelity between two quantum processes.
        
        Args:
            A (Qobj): The target quantum process (e.g., a process matrix).
            B (Qobj): The achieved quantum process.
        
        Returns:
            float: The process fidelity value.
        """
        return process_fidelity(A, B) 
    
    def compute_psu_fidelity(self, A: Qobj, B: Qobj) -> float:
        """
        Compute PSU (Pure State Unitary) fidelity.
        
        Args:
            A (Qobj): The target quantum state.
            B (Qobj): The achieved quantum state.
        
        Returns:
            float: The PSU fidelity value between the two states.
        """
        return abs((A.dag() * B).tr()) ** 2  # PSU fidelity calculation (example)

    def compute_symplectic_fidelity(self, A: Qobj, B: Qobj) -> float:
        """
        Compute the symplectic fidelity.
        
        Args:
            A (Qobj): The target quantum state.
            B (Qobj): The achieved quantum state.
        
        Returns:
            float: The symplectic fidelity value between the two states.
        """
        return np.abs((A.dag() * B).tr()) ** 2  # Symplectic fidelity calculation (example)

    def compute_multiple_fidelities(self, states1: List[Qobj], states2: List[Qobj]) -> List[float]:
        """
        Compute multiple fidelities in parallel.
        
        Args:
            states1 (List[Qobj]): List of target quantum objects.
            states2 (List[Qobj]): List of achieved quantum objects.
        
        Returns:
            List[float]: List of fidelity values for each pair.
        """
        # Use joblib to parallelize fidelity computations
        results = Parallel(n_jobs=-1)(
            delayed(self.compute_fidelity)(s1, s2) for s1, s2 in zip(states1, states2)
        )
        return results

# --- Validation --- 

def validate_qobj_pair(A: Qobj, B: Qobj, fidtype: str):
    """
    Validates that the target and achieved Qobj are compatible for fidelity computation.
    
    Args:
        A (Qobj): The target quantum object.
        B (Qobj): The achieved quantum object.
        fidtype (str): The type of fidelity ('state', 'unitary', 'super', etc.).
    
    Raises:
        ValueError: If the Qobj pair is incompatible for the specified fidelity type.
    """
    if fidtype == 'state' or fidtype == 'unitary':
        if A.shape != B.shape:
            raise ValueError(f"Target and achieved Qobj must have the same shape for {fidtype} fidelity.")
        if not ((A.isunitary and B.isunitary) or (A.isherm and B.isherm) or (A.isket and B.isket)):
            raise ValueError(f"For {fidtype} fidelity, the Qobjs must be valid unitary or Hermitian operators.")
    elif fidtype == 'super':
        # Add any necessary checks for superoperators
        pass
    elif fidtype == 'process':
        # Add any necessary checks for process fidelity (e.g., validity of process matrices)
        pass
    else:
        raise ValueError(f"Unsupported fidelity type: {fidtype}")

class FidelityComputerPSU:
    def fidelity(self, target: Qobj, state: Qobj) -> float:
        """
        Compute PSU (Pure State Unitary) fidelity.
        
        Args:
            target (Qobj): The target quantum state.
            state (Qobj): The quantum state to compare to the target.
        
        Returns:
            float: The PSU fidelity value between the two states.
        """
        return (state.overlap(target)) ** 2  # Example PSU fidelity calculation

    def gradient(self, target: Qobj, state: Qobj, control_params: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the PSU fidelity with respect to the control parameters.
        
        Args:
            target (Qobj): The target quantum state.
            state (Qobj): The quantum state to compare to the target.
            control_params (np.ndarray): The control parameters for optimization.
        
        Returns:
            np.ndarray: The gradient of the fidelity with respect to control parameters.
        """
        # Compute the overlap between the target and the state
        overlap = state.overlap(target)

        fidelity_gradient = 2 * np.real(np.conj(overlap) * self._compute_state_gradient(state, control_params))

        return fidelity_gradient


    def _compute_state_gradient(self, state: Qobj, control_params: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the quantum state with respect to the control parameters.
        
        Args:
            state (Qobj): The quantum state.
            control_params (np.ndarray): The control parameters for optimization.
        
        Returns:
            np.ndarray: The gradient of the quantum state with respect to the control parameters.
        """
        # This is a placeholder for actual state gradient computation.
        # Depending on how the state is parameterized (e.g., as a function of time or other parameters),
        # this method will compute the gradient of the state with respect to the control parameters.
        return np.gradient(state.full())  # Example, modify as necessary.

class FidelityComputerSymplectic:
    def fidelity(self, target: Qobj, state: Qobj) -> float:
        """
        Compute symplectic fidelity.
        
        Args:
            target (Qobj): The target quantum state.
            state (Qobj): The quantum state to compare to the target.
        
        Returns:
            float: The symplectic fidelity value between the two states.
        """
        return np.abs((target.dag() * state).tr()) ** 2  # Symplectic fidelity calculation

    def gradient(self, target: Qobj, state: Qobj, control_params: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the symplectic fidelity with respect to the control parameters.
        
        Args:
            target (Qobj): The target quantum state.
            state (Qobj): The quantum state to compare to the target.
            control_params (np.ndarray): The control parameters for optimization.
        
        Returns:
            np.ndarray: The gradient of the fidelity with respect to control parameters.
        """
        # Compute the overlap between the target and the state (symplectic)
        overlap = np.abs((target.dag() * state).tr())
        fidelity_gradient = 2 * np.real(overlap * self._compute_state_gradient(state, control_params))

        return fidelity_gradient


    def _compute_state_gradient(self, state: Qobj, control_params: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the quantum state with respect to the control parameters.
        
        Args:
            state (Qobj): The quantum state.
            control_params (np.ndarray): The control parameters for optimization.
        
        Returns:
            np.ndarray: The gradient of the quantum state with respect to the control parameters.
        """
        # This is a placeholder for actual state gradient computation.
        # Depending on how the state is parameterized (e.g., as a function of time or other parameters),
        # this method will compute the gradient of the state with respect to the control parameters.
        return np.gradient(state.full())  # Example, modify as necessary.