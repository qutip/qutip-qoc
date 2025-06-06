"""
This module contains functions that implement the GOAT algorithm to
calculate optimal parameters for analytical control pulse sequences.
"""
import numpy as np
import qutip as qt
from qutip import Qobj, QobjEvo
from qutip_qoc.fidcomp import FidelityComputer  # Import the FidelityComputer


class _GOAT:
    """
    Class for storing a control problem and calculating
    the fidelity error function and its gradient wrt the control parameters,
    according to the GOAT algorithm.
    Uses FidelityComputer for fidelity calculations.
    """

    def __init__(
        self,
        objective,
        time_interval,
        time_options,
        control_parameters,
        alg_kwargs,
        guess_params,
        **integrator_kwargs,
    ):
        # calculated during optimization
        self._g = None  # normalized overlap of X and target
        self._X = None  # most recently calculated evolution operator
        self._dX = None  # derivative of X wrt control parameters

        # Initialize FidelityComputer
        self._fid_type = alg_kwargs.get("fid_type", "PSU")
        self._fidcomp = FidelityComputer(self._fid_type)

        # make superoperators conform with SESolver
        if objective.H[0].issuper:
            self._is_super = True
            # extract drift and control Hamiltonians from the objective
            self._Hd = Qobj(objective.H[0].data)  # super -> oper
            self._Hc_lst = [Qobj(Hc[0].data) for Hc in objective.H[1:]]
            # extract initial and target state or operator from the objective
            self._initial = Qobj(objective.initial.data)
            self._target = Qobj(objective.target.data)
        else:
            self._is_super = False
            self._Hd = objective.H[0]
            self._Hc_lst = [Hc[0] for Hc in objective.H[1:]]
            self._initial = objective.initial
            self._target = objective.target

        # extract control functions and gradients from the objective
        self._controls = [H[1] for H in objective.H[1:]]
        self._grads = [H[2].get("grad", None) for H in objective.H[1:]]
        if None in self._grads:
            raise KeyError(
                "No gradient function found for control function "
                "at index {}.".format(self._grads.index(None)))
        self._evo_time = time_interval.evo_time
        self._var_t = "guess" in time_options

        # num of params for each control function
        self._para_counts = [
            len(v["guess"]) for k, v in control_parameters.items() if k != "__time__"
        ]

        # inferred attributes
        self._tot_n_para = sum(self._para_counts)  # excl. time
        self._sys_size = self._Hd.shape[0]

        # Scale the system Hamiltonian and initial state
        # for coupled system (X, dX)
        self._H_dia, self._H = self._prepare_generator_dia()
        self._H_off_dia = self._prepare_generator_off_dia()
        self._psi0 = self._prepare_state()

        self._evo = QobjEvo(self._H_dia + self._H_off_dia, {"p": guess_params})
        if self._is_super:  # for SESolver
            self._evo = (1j) * self._evo

        if self._var_t:  # for time derivative
            self._H_evo = QobjEvo(self._H, {"p": guess_params})

        # initialize the solver
        self._solver = qt.SESolver(H=self._evo, options=integrator_kwargs)

    def _prepare_state(self):
        """Initial state for coupled system (X, dX)"""
        scale = qt.data.one_element_csr(
            position=(0, 0), shape=(1 + self._tot_n_para, 1))
        psi0 = Qobj(scale) & self._initial
        return psi0

    def _prepare_generator_dia(self):
        """Diagonal elements of coupled system Hamiltonian"""
        def helper(control, lower, upper):
            return lambda t, p: control(t, p[lower:upper])

        H = [self._Hd] if self._var_t else []
        dia = qt.qeye(1 + self._tot_n_para)
        H_dia = [dia & self._Hd]

        idx = 0
        for control, M, Hc in zip(self._controls, self._para_counts, self._Hc_lst):
            if self._var_t:
                H.append([Hc, helper(control, idx, idx + M)])

            hc_dia = dia & Hc
            H_dia.append([hc_dia, helper(control, idx, idx + M)])
            idx += M

        return H_dia, H

    def _prepare_generator_off_dia(self):
        """Off-diagonal elements of coupled system Hamiltonian"""
        def helper(grad, lower, upper, idx):
            return lambda t, p: grad(t, p[lower:upper], idx)

        csr_shape = (1 + self._tot_n_para, 1 + self._tot_n_para)
        dH = []

        idx = 0
        for grad, M, Hc in zip(self._grads, self._para_counts, self._Hc_lst):
            for grad_idx in range(M):
                i = 1 + idx + grad_idx
                csr = qt.data.one_element_csr(position=(i, 0), shape=csr_shape)
                hc = Qobj(csr) & Hc
                dH.append([hc, helper(grad, idx, idx + M, grad_idx)])
            idx += M

        return dH

    def _solve_EOM(self, evo_time, params):
        """Solve equations of motion for X and dX"""
        res = self._solver.run(self._psi0, [0.0, evo_time], args={"p": params})

        X = res.final_state[: self._sys_size, : self._sys_size]
        dX = res.final_state[self._sys_size :, : self._sys_size]

        return X, dX

    def infidelity(self, params):
        """Calculate infidelity using FidelityComputer"""
        # adjust integration time-interval if time is parameter
        evo_time = self._evo_time if self._var_t is False else params[-1]

        X, self._dX = self._solve_EOM(evo_time, params)
        self._X = Qobj(X, dims=self._target.dims)

        # Use FidelityComputer for fidelity calculation
        infid = self._fidcomp.compute_infidelity(self._initial, self._target, self._X)
        
        # Store overlap for gradient calculation
        if self._fid_type != "TRACEDIFF":
            self._g = self._target.overlap(self._X)

        return infid

    def gradient(self, params):
        """Calculate gradient of fidelity error function"""
        X, dX = self._X, self._dX  # calculated in infidelity()

        dX_lst = []  # collect derivatives for each parameter
        for i in range(self._tot_n_para):
            idx = i * self._sys_size
            dx = dX[idx : idx + self._sys_size, :]
            dX_lst.append(Qobj(dx))

        if self._var_t:
            H_T = self._H_evo(params[-1], p=params)
            dX_dT = -1j * H_T * X
            if self._is_super:
                dX_dT = (1j) * dX_dT
            dX_lst.append(dX_dT)

        if self._fid_type == "TRACEDIFF":
            # For TRACEDIFF, gradient is based on trace difference
            diff = X - self._target
            trc = [dx.overlap(diff) + diff.overlap(dx) for dx in dX_lst]
            grad = 0.5 * np.real(np.array(trc)) / self._target.norm()
        else:
            # For PSU/SU, gradient is based on overlap
            trc = [self._target.overlap(dx) for dx in dX_lst]
            
            if self._fid_type == "PSU":
                # Phase factor for PSU
                phase_fac = np.conj(self._g) / np.abs(self._g) if self._g != 0 else 0
            else:  # SU
                phase_fac = 1
            
            grad = -(phase_fac * np.array(trc)).real

        return grad