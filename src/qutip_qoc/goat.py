"""
This module contains functions that implement the GOAT algorithm to
calculate optimal parameters for analytical control pulse sequences.
"""
import numpy as np
import scipy as sp

import qutip as qt
from qutip import Qobj, QobjEvo

__all__ = ["GOAT"]


class GOAT:
    """
    Class for storing a control problem and calculating
    the fidelity error function and its gradient wrt the control parameters,
    according to the GOAT algorithm.

    Attributes
    ----------
    g: float
        Normalized overlap of X and target.
    X: Qobj
        Most recently calculated evolution operator.
    dX: list
        Derivative of X wrt control parameters.
    """

    # calculated during optimization
    g = None
    X = None
    dX = None

    def __init__(
        self,
        objective,
        time_interval,
        time_options,
        pulse_options,
        alg_kwargs,
        guess_params,
        **integrator_kwargs,
    ):
        # make superoperators conform with SESolver
        if objective.H[0].issuper:
            self.is_super = True

            # extract drift and control Hamiltonians from the objective
            self.Hd = Qobj(objective.H[0].data)  # super -> oper
            self.Hc_lst = [Qobj(Hc[0].data) for Hc in objective.H[1:]]

            # extract initial and target state or operator from the objective
            self.initial = Qobj(objective.initial.data)
            self.target = Qobj(objective.target.data)

            self.fid_type = alg_kwargs.get("fid_type", "TRACEDIFF")

        else:
            self.is_super = False
            self.Hd = objective.H[0]
            self.Hc_lst = [Hc[0] for Hc in objective.H[1:]]
            self.initial = objective.initial
            self.target = objective.target
            self.fid_type = alg_kwargs.get("fid_type", "PSU")

        # extract control functions and gradients from the objective
        self.controls = [H[1] for H in objective.H[1:]]
        self.grads = [H[2].get("grad", None) for H in objective.H[1:]]
        if None in self.grads:
            raise KeyError(
                "No gradient function found for control function "
                "at index {}.".format(self.grads.index(None))
            )

        self.evo_time = time_interval.evo_time
        self.var_t = "guess" in time_options

        # num of params for each control function
        self.para_counts = [len(v["guess"]) for v in pulse_options.values()]

        # inferred attributes
        self.tot_n_para = sum(self.para_counts)  # excl. time
        self.norm_fac = 1 / self.target.norm()
        self.sys_size = self.Hd.shape[0]

        # Scale the system Hamiltonian and initial state
        # for coupled system (X, dX)
        self.H_dia, self.H = self._prepare_generator_dia()
        self.H_off_dia = self._prepare_generator_off_dia()
        self.psi0 = self._prepare_state()

        self.evo = QobjEvo(self.H_dia + self.H_off_dia, {"p": guess_params})
        if self.is_super:  # for SESolver
            self.evo = (1j) * self.evo

        if self.var_t:  # for time derivative
            self.H_evo = QobjEvo(self.H, {"p": guess_params})

        # initialize the solver TODO: usage of other solvers
        self.solver = qt.SESolver(H=self.evo, options=integrator_kwargs)

    def _prepare_state(self):
        """
        inital state (t=0) for coupled system (X, dX):
        [[  X(0)], -> [[1],
         [d1X(0)], ->  [0],
         [d2X(0)], ->  [0],
         [  ... ]] ->  [0]]
        """
        # TODO: use qutip CSR
        scale = sp.sparse.csr_matrix(([1], ([0], [0])), shape=(1 + self.tot_n_para, 1))
        psi0 = Qobj(scale) & self.initial
        return psi0

    def _prepare_generator_dia(self):
        """
        Combines the scaled and parameterized Hamiltonian elements on the diagonal
        of the coupled system (X, dX) Hamiltonian, with associated pulses:
        [[  H, 0, 0, ...], [[  X],
         [d1H, H, 0, ...],  [d1X],
         [d2H, 0, H, ...],  [d2X],
         [...,         ]]   [...]]
        Additionlly, if the time is a parameter, the time-dependent
        parameterized Hamiltonian without scaling
        """

        def helper(control, lower, upper):
            # to fix parameter index in loop
            return lambda t, p: control(t, p[lower:upper])

        # H = [Hd, [H0, c0(t)], ...]
        H = [self.Hd] if self.var_t else []

        dia = qt.qeye(1 + self.tot_n_para)
        H_dia = [dia & self.Hd]

        idx = 0
        for control, M, Hc in zip(self.controls, self.para_counts, self.Hc_lst):
            if self.var_t:
                H.append([Hc, helper(control, idx, idx + M)])

            hc_dia = dia & Hc
            H_dia.append([hc_dia, helper(control, idx, idx + M)])
            idx += M

        return H_dia, H  # lists to construct QobjEvo

    def _prepare_generator_off_dia(self):
        """
        Combines the scaled and parameterized Hamiltonian off-diagonal elements
        for the coupled system (X, dX) with associated pulses:
        [[  H, 0, 0, ...], [[  X],
         [d1H, H, 0, ...],  [d1U],
         [d2H, 0, H, ...],  [d2U],
         [...,         ]]   [...]]
        The off-diagonal elements correspond to the derivative elements
        """

        def helper(grad, lower, upper, idx):
            # to fix parameter index in loop
            return lambda t, p: grad(t, p[lower:upper], idx)

        csr_shape = (1 + self.tot_n_para, 1 + self.tot_n_para)

        # dH = [[H1', dc1'(t)], [H1", dc1"(t)], ... , [H2', dc2'(t)], ...]
        dH = []

        idx = 0
        for grad, M, Hc in zip(self.grads, self.para_counts, self.Hc_lst):
            for grad_idx in range(M):
                i = 1 + idx + grad_idx
                csr = sp.sparse.csr_matrix(([1], ([i], [0])), csr_shape)
                hc = Qobj(csr) & Hc
                dH.append([hc, helper(grad, idx, idx + M, grad_idx)])

            idx += M

        return dH  # list to construct QobjEvo

    def _solve_EOM(self, evo_time, params):
        """
        Calculates X, and dX i.e. the derivative of the evolution operator X
        wrt the control parameters by solving the Schroedinger operator equation
        returns X as Qobj and dX as list of dense matrices
        """
        res = self.solver.run(self.psi0, [0.0, evo_time], args={"p": params})

        X = res.final_state[: self.sys_size, : self.sys_size]
        dX = res.final_state[self.sys_size :, : self.sys_size]

        return X, dX

    def infidelity(self, params):
        """
        returns the infidelity to be minimized
        store intermediate results for gradient calculation
        the normalized overlap, the current unitary and its gradient
        """
        # adjust integration time-interval, if time is parameter
        evo_time = self.evo_time if self.var_t is False else params[-1]

        X, self.dX = self._solve_EOM(evo_time, params)

        self.X = Qobj(X, dims=self.target.dims)

        if self.fid_type == "TRACEDIFF":
            diff = self.X - self.target
            self.g = 1 / 2 * diff.overlap(diff)
            infid = self.norm_fac * np.real(self.g)
        else:
            self.g = self.norm_fac * self.target.overlap(self.X)
            if self.fid_type == "PSU":  # f_PSU (drop global phase)
                infid = 1 - np.abs(self.g)
            elif self.fid_type == "SU":  # f_SU (incl global phase)
                infid = 1 - np.real(self.g)

        return infid

    def gradient(self, params):
        """
        Calculates the gradient of the fidelity error function
        wrt control parameters by solving the Schroedinger operator equation
        """
        X, dX, g = self.X, self.dX, self.g  # calculated before

        dX_lst = []  # collect for each parameter
        for i in range(self.tot_n_para):
            idx = i * self.sys_size  # row index for parameter set i
            dx = dX[idx : idx + self.sys_size, :]
            dX_lst.append(Qobj(dx))

        if self.var_t:
            H_T = self.H_evo(params[-1], p=params)
            dX_dT = -1j * H_T * X
            if self.is_super:
                dX_dT = (1j) * dX_dT
            dX_lst.append(dX_dT)

        if self.fid_type == "TRACEDIFF":
            diff = X - self.target
            # product rule
            trc = [dx.overlap(diff) + diff.overlap(dx) for dx in dX_lst]
            grad = self.norm_fac * 1 / 2 * np.real(np.array(trc))

        else:  # -Re(... * Tr(...)) NOTE: gradient will be zero at local maximum
            trc = [self.target.overlap(dx) for dx in dX_lst]

            if self.fid_type == "PSU":  # f_PSU (drop global phase)
                # phase_fac = exp(-i*phi)
                phase_fac = np.conj(g) / np.abs(g) if g != 0 else 0

            elif self.fid_type == "SU":  # f_SU (incl global phase)
                phase_fac = 1

            grad = -(self.norm_fac * phase_fac * np.array(trc)).real

        return grad
