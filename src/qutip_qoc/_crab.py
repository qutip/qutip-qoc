"""
This module provides an implementation of the CRAB optimization algorithm.
It defines the _CRAB class which calculates the fidelity error function
using the FidelityComputer class.
"""

import numpy as np
import qutip as qt
from qutip_qoc.fidcomp import FidelityComputer

class _CRAB:
    """
    Class implementing the CRAB optimization algorithm.
    Uses FidelityComputer for fidelity calculations and manages its own optimization state.
    """

    def __init__(self, objective, time_interval, time_options, 
                control_parameters, alg_kwargs, guess_params, **integrator_kwargs):
        """
        Initialize CRAB optimizer.
        
        Parameters:
        -----------
        objective : Objective
            The control objective containing initial/target states and Hamiltonians
        time_interval : _TimeInterval
            Time discretization for the optimization
        time_options : dict
            Options for time evolution
        control_parameters : dict
            Control parameters with bounds and initial guesses
        alg_kwargs : dict
            Algorithm-specific parameters including:
            - fid_type: Fidelity type ('PSU', 'SU', 'TRACEDIFF')
            - num_coeffs: Number of CRAB coefficients
            - fix_frequency: Whether frequencies are fixed
        guess_params : array
            Initial guess parameters
        integrator_kwargs : dict
            Options for the ODE integrator
        """
        self.objective = objective
        self.time_interval = time_interval
        self.control_parameters = control_parameters
        self.alg_kwargs = alg_kwargs
        self.guess_params = guess_params
        self.integrator_kwargs = integrator_kwargs

        # Initialize fidelity computer
        self.fidcomp = FidelityComputer(alg_kwargs.get("fid_type", "PSU"))

        # CRAB-specific parameters
        self.num_coeffs = alg_kwargs.get("num_coeffs", 2)
        self.fix_frequency = alg_kwargs.get("fix_frequency", False)
        self.init_coeff_scaling = alg_kwargs.get("init_coeff_scaling", 1.0)
        
        # Extract control bounds
        self.bounds = []
        for key, params in control_parameters.items():
            if key != "__time__":
                self.bounds.append(params.get("bounds"))

        # Statistics tracking
        self.num_fid_func_calls = 0
        self.stats = None
        self.iter_summary = None
        self.dump = None

    def _generate_crab_pulse(self, params, n_tslots):
        """
        Generate a CRAB pulse from Fourier coefficients.
        
        Parameters:
        -----------
        params : array
            CRAB parameters (amplitudes, phases, frequencies)
        n_tslots : int
            Number of time slots
            
        Returns:
        --------
        array
            Pulse amplitudes for each time slot
        """
        t = np.linspace(0, 1, n_tslots)
        pulse = np.zeros(n_tslots)
        
        if self.fix_frequency:
            # Parameters are [A1, A2, ..., phi1, phi2, ...]
            num_components = len(params) // 2
            amplitudes = params[:num_components] * self.init_coeff_scaling
            phases = params[num_components:2*num_components]
            # Use linearly spaced frequencies if fixed
            frequencies = np.linspace(1, 10, num_components)
        else:
            # Parameters are [A1, A2, ..., phi1, phi2, ..., w1, w2, ...]
            num_components = len(params) // 3
            amplitudes = params[:num_components] * self.init_coeff_scaling
            phases = params[num_components:2*num_components]
            frequencies = params[2*num_components:3*num_components]
        
        for A, phi, w in zip(amplitudes, phases, frequencies):
            pulse += A * np.sin(w * t + phi)
            
        return pulse

    def _get_hamiltonian(self, pulses):
        """
        Construct the time-dependent Hamiltonian from control pulses.
        
        Parameters:
        -----------
        pulses : list of arrays
            Control pulses for each control Hamiltonian
            
        Returns:
        --------
        QobjEvo
            Time-dependent Hamiltonian
        """
        H = [self.objective.H[0]]  # Drift Hamiltonian
        
        for i, Hc in enumerate(self.objective.H[1:]):
            # Create time-dependent control term
            H.append([Hc[0] if isinstance(Hc, list) else Hc, 
                     lambda t, args, i=i: args['pulses'][i][int(t/self.time_interval.evo_time * len(args['pulses'][i]))]])
        
        return qt.QobjEvo(H, args={'pulses': pulses})

    def infidelity(self, params):
        """
        Calculate the infidelity for given CRAB parameters.
        
        Parameters:
        -----------
        params : array
            CRAB optimization parameters
            
        Returns:
        --------
        float
            Infidelity value
        """
        self.num_fid_func_calls += 1
        
        # Generate pulses for each control from CRAB parameters
        pulses = []
        num_ctrls = len(self.objective.H) - 1
        params_per_ctrl = len(params) // num_ctrls
        
        for i in range(num_ctrls):
            ctrl_params = params[i*params_per_ctrl:(i+1)*params_per_ctrl]
            pulses.append(self._generate_crab_pulse(ctrl_params, self.time_interval.n_tslots))
        
        # Create Hamiltonian with generated pulses
        H = self._get_hamiltonian(pulses)
        
        # Evolve the system
        result = qt.mesolve(
            H,
            self.objective.initial,
            self.time_interval.tslots,
            options=qt.Options(**self.integrator_kwargs)
        )
        
        # Calculate infidelity
        evolved = result.states[-1]
        return self.fidcomp.compute_infidelity(self.objective.initial, self.objective.target, evolved)