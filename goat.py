"""
This module contains functions that implement the GOAT algorithm to
calculate optimal parameters for analytical control pulse sequences.
"""
import numpy as np
import scipy as sp

import qutip as qt
from qutip import Qobj, QobjEvo


class GOAT:
    """
    Class for storing a control problem and calculating
    the fidelity error function and its gradient wrt the control parameters.
    """
    # calculated during optimization
    g = None  # normalized overlap of X and target
    X = None  # current evolution operator
    dX = None  # derivative of X wrt control parameters
    infid = None  # infidelity

    def __init__(self, objective, time_interval, time_options, pulse_options, alg_kwargs, guess_params, **integrator_kwargs):

        # make superoperators conform with SESolver
        if objective.H_evo[0].issuper:
            self.is_super = True

            # extract drift and control Hamiltonians from the objective
            self.Hd = Qobj(objective.H_evo[0].data)  # super -> oper
            self.Hc_lst = [Qobj(Hc[0].data) for Hc in objective.H_evo[1:]]

            # extract initial and target state or operator from the objective
            self.initial = Qobj(objective.initial.data)
            self.target = Qobj(objective.target.data)

            self.fid_type = alg_kwargs.get("fid_type", "TRACEDIFF")

        else:
            self.is_super = False
            self.Hd = objective.H_evo[0]
            self.Hc_lst = [Hc[0] for Hc in objective.H_evo[1:]]
            self.initial = objective.initial
            self.target = objective.target
            self.fid_type = alg_kwargs.get("fid_type", "PSU")

        # extract control functions and gradients from the objective
        self.controls = [H[1] for H in objective.H_evo[1:]]
        self.grads = [H[2].get("grad", None) for H in objective.H_evo[1:]]
        if None in self.grads:
            raise KeyError("No gradient function found for control function "
                           "at index {}.".format(self.grads.index(None)))

        self.evo_time = time_interval.evo_time
        self.var_t = "guess" in time_options

        # num of params for each control function
        self.para_counts = [len(v["guess"]) for v in pulse_options.values()]
        if self.var_t:  # add one parameter for time if variable
            self.para_counts.append(1)

        # inferred attributes
        self.tot_n_para = sum(self.para_counts)  # incl. time if var_t==True
        self.norm_fac = 1 / self.target.norm()
        self.sys_size = self.Hd.shape[0]

        # Scale the system Hamiltonian and initial state
        # for coupled system (X, dX)
        self.H = self.prepare_H()
        self.dH = self.prepare_dH()
        self.psi0 = self.prepare_psi0()

        self.evo = QobjEvo(self.H + self.dH, {"p": guess_params})
        if self.is_super:  # for SESolver
            self.evo = (1j) * self.evo

        # initialize the solver
        self.solver = qt.SESolver(H=self.evo, options=integrator_kwargs)

    def prepare_psi0(self):
        """
        inital state for coupled system (X, dX):
        [[  H, 0, 0, ...], [[  X],
         [d1H, H, 0, ...],  [d1U],
         [d2H, 0, H, ...],  [d2U],
         [...,         ]]   [...]]
        """
        scale = sp.sparse.csr_matrix(
            ([1], ([0], [0])), shape=(1 + self.tot_n_para, 1)
        )
        psi0 = Qobj(scale) & self.initial
        return psi0

    def prepare_H(self):
        """
        Combines the scaled Hamiltonian diagonal elements
        for the coupled system (X, dX) with associated pulses:
        [[  H, 0, 0, ...], [[  X],
         [d1H, H, 0, ...],  [d1U],
         [d2H, 0, H, ...],  [d2U],
         [...,         ]]   [...]]
        """
        def helper(control, lower, upper):
            # to fix parameter index in loop
            return lambda t, p: control(t, p[lower:upper])

        diag = qt.qeye(1 + self.tot_n_para)
        H = [diag & self.Hd]
        idx = 0

        # H = [Hd, [H0, c0(t)], ...]

        for control, M, Hc in zip(self.controls, self.para_counts, self.Hc_lst):
            hc = diag & Hc
            H.append([hc, helper(control, idx, idx + M)])
            idx += M
        return H  # list to construct QobjEvo

    def prepare_dH(self):
        """
        Combines the scaled Hamiltonian off-diagonal elements
        for the coupled system (X, dX) with associated pulses:
        [[  H, 0, 0, ...], [[  X],
         [d1H, H, 0, ...],  [d1U],
         [d2H, 0, H, ...],  [d2U],
         [...,         ]]   [...]]
        """
        def helper(control, lower, upper, idx):
            # to fix parameter index in loop
            return lambda t, p: grad(t, p[lower:upper], idx)

        csr_shape = (1 + self.tot_n_para,  1 + self.tot_n_para)
        dH = []
        idx = 0

        # dH = [[H1', dc1'(t)], [H1", dc1"(t)], ... , [H2', dc2'(t)], ...]

        for grad, M, Hc in zip(self.grads, self.para_counts, self.Hc_lst):

            for grad_idx in range(M + int(self.var_t)):
                # grad_idx == M -> time parameter
                i = 1 + idx + grad_idx if grad_idx < M else self.tot_n_para
                csr = sp.sparse.csr_matrix(([1], ([i], [0])), csr_shape)
                hc = Qobj(csr) & Hc
                dH.append([hc, helper(grad, idx, idx + M, grad_idx)])

            idx += M

        return dH  # list to construct QobjEvo

    def solve_EOM(self, evo_time, params):
        """
        Calculates X, and dX i.e. the derivative of the evolution operator X
        wrt the control parameters by solving the Schrodinger operator equation
        """
        res = self.solver.run(self.psi0, [0., evo_time], args={'p': params})

        X = res.final_state[:self.sys_size, :self.sys_size]
        dX = res.final_state[self.sys_size:, :self.sys_size]
        return X, dX

    def infidelity(self, params):
        """
        returns the infidelity to be minimized
        store intermediate results for gradient calculation
        the normalized overlap, the current unitary and its gradient
        """
        # adjust integration time-interval, if time is parameter
        evo_time = self.evo_time if self.var_t == False else params[-1]

        X, self.dX = self.solve_EOM(evo_time, params)

        self.X = Qobj(X, dims=self.target.dims)

        if self.fid_type == "TRACEDIFF":
            diff = self.X - self.target
            self.g = 1/2 * diff.overlap(diff)
            self.infid = self.norm_fac * np.real(self.g)
        else:
            self.g = self.norm_fac * self.target.overlap(self.X)
            if self.fid_type == "PSU":  # f_PSU (drop global phase)
                self.infid = 1 - np.abs(self.g)
            elif self.fid_type == "SU":  # f_SU (incl global phase)
                self.infid = 1 - np.real(self.g)

        return self.infid

    def gradient(self):
        """
        Calculates the gradient of the fidelity error function
        wrt control parameters by solving the Schrodinger operator equation
        """
        X, dX, g = self.X, self.dX, self.g  # calculated before

        dX_lst = []  # collect for each parameter
        for i in range(self.tot_n_para):
            idx = i * self.sys_size  # row index for parameter set i
            dx = dX[idx: idx + self.sys_size, :]
            dX_lst.append(Qobj(dx))

        if self.fid_type == "TRACEDIFF":
            diff = X - self.target
            # product rule
            trc = [dx.overlap(diff) + diff.overlap(dx) for dx in dX_lst]
            grad = self.norm_fac * 1/2 * np.real(np.array(trc))

        else:  # -Re(... * Tr(...)) NOTE: gradient will be zero at local maximum
            trc = [self.target.overlap(dx) for dx in dX_lst]

            if self.fid_type == "PSU":  # f_PSU (drop global phase)
                # phase_fac = exp(-i*phi)
                phase_fac = np.conj(g) / np.abs(g) if g != 0 else 0

            elif self.fid_type == "SU":  # f_SU (incl global phase)
                phase_fac = 1

            grad = -(self.norm_fac * phase_fac * np.array(trc)).real

        return grad


class Multi_GOAT:
    """
    Composite class for multiple GOAT instances
    to optimize multiple objectives simultaneously
    """

    def __init__(self, objectives, time_interval, time_options, pulse_options, alg_kwargs, guess_params, **integrator_kwargs):
        self.goats = [GOAT(obj, time_interval, time_options, pulse_options, alg_kwargs, guess_params, ** integrator_kwargs)
                      for obj in objectives]
        self.mean_infid = None

    def goal_fun(self, params):
        infid_sum = 0
        for goat in self.goats:  # TODO: parallelize
            infid = goat.infidelity(params)
            if infid < 0:
                print(
                    "WARNING: infidelity < 0 -> inaccurate integration, "
                    "try reducing integrator tolerance (atol, rtol)"
                )
            infid_sum += infid
        self.mean_infid = np.mean(infid_sum)
        return self.mean_infid

    def grad_fun(self, params):
        grads = 0
        for g in self.goats:
            grads += g.gradient()
        return grads
