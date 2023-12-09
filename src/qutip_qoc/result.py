import jaxlib
import pickle
import textwrap
import numpy as np
from inspect import signature

import qutip as qt
from qutip_qoc.objective import Objective

__all__ = ["Result"]


class Result():
    """
    Class for storing the results of a pulse control optimization run.

    Attributes:
    ----------
    objectives : list of :class:`qutip_qoc.Objective`
        List of objectives to be optimized.

    time_interval : :class:`qutip_qoc.TimeInterval`
        Time interval for the optimization.

    start_local_time : struct_time
        Time when the optimization started.

    end_local_time : struct_time
        Time when the optimization ended.

    total_seconds : float
        Total time in seconds the optimization took.
        Equal to the sum of iter_seconds.
        Equal to difference between end_local_time and start_local_time.

    iters : int
        Number of iterations until convergence.
        Equal to the length of iter_seconds.

    iter_seconds : list of float
        Seconds between each iteration.

    message : str
        Reason for termination.

    guess_controls : list of ndarray
        List of guess control pulses used to initialize the optimization.

    optimized_controls : list of ndarray
        List of optimized control pulses.

    optimized_objectives : list of :class:`qutip_qoc.Objective`
        List of objectives with optimized control pulses.

    final_states : list of :class:`qutip.Qobj`
        List of final states after the optimization.
        One for each objective.

    infidelity : float
        Final infidelity error after the optimization.

    var_time : bool
        Whether the optimization was performed with variable time.
        If True, the last parameter in optimized_params is the evolution time.
    """

    def __init__(
            self,
            objectives=None,
            time_interval=None,
            start_local_time=None,
            end_local_time=None,
            total_seconds=None,
            n_iters=None,
            iter_seconds=None,
            message=None,
            guess_controls=None,
            optimized_controls=None,
            optimized_objectives=None,
            final_states=None,
            guess_params=None,
            new_params=None,
            optimized_params=None,
            infidelity=np.inf,
            var_time=False,
    ):
        self.time_interval = time_interval
        self.objectives = objectives
        self.start_local_time = start_local_time
        self.end_local_time = end_local_time
        self._total_seconds = total_seconds
        self.n_iters = n_iters
        self.iter_seconds = iter_seconds
        self.message = message
        self._guess_controls = guess_controls
        self._optimized_controls = optimized_controls
        self._optimized_objectives = optimized_objectives
        # not present in Krotov
        self.guess_params = guess_params
        self.new_params = new_params
        self._optimized_params = optimized_params
        self.final_states = final_states
        self.infidelity = infidelity
        self.var_time = var_time

    def __str__(self):
        return textwrap.dedent(
            r'''
        Control Optimization Result
        --------------------------
        - Started at {start_local_time}
        - Number of objectives: {n_objectives}
        - Final fidelity error: {final_infid}
        - Final parameters: {final_params}
        - Number of iterations: {n_iters}
        - Reason for termination: {message}
        - Ended at {end_local_time} ({time_delta}s)
        '''.format(
                start_local_time=self.start_local_time,
                n_objectives=len(self.objectives),
                final_infid=self.infidelity,
                final_params=self.optimized_params,
                n_iters=self.n_iters,
                end_local_time=self.end_local_time,
                time_delta=self.total_seconds,
                message=self.message)
        ).strip()

    def __repr__(self):
        return self.__str__()

    @property
    def total_seconds(self):
        if self._total_seconds is None:
            self._total_seconds = sum(self.iter_seconds)
        return self._total_seconds

    @property
    def optimized_params(self):
        if self._optimized_params is None:
            # reshape (optimized) new_parameters array to match
            # shape and type of the guess_parameters list
            opt_params = []

            idx = 0
            for guess in self.guess_params:
                opt = self.new_params[idx: idx + len(guess)]

                if isinstance(guess, list):
                    opt = opt.tolist()

                opt_params.append(opt)
                idx += len(guess)

            self._optimized_params = opt_params
        return self._optimized_params

    @optimized_params.setter
    def optimized_params(self, params):
        self._optimized_params = params

    @property
    def optimized_controls(self):
        if self._optimized_controls is None:
            opt_ctrl = []

            for Hc, xf in zip(self.objectives[0].H[1:], self.optimized_params):

                control = Hc[1]
                if callable(control): # continuous control as in JOAT/GOAT
                    cf = []
                    try:
                        tslots = self.time_interval.tslots
                    except Exception:
                        print(
                            "time_interval.tslots not specified "
                            "(probably missing n_tslots), defaulting to 100 "
                            "collocation points for result.optimized_controls")
                        tslots = np.linspace(
                            0., self.time_interval.evo_time, 100)

                    for t in tslots:
                        cf.append(control(t, xf))
                else: # discrete control as in GRAPE/CRAB
                    cf = xf
                opt_ctrl.append(cf)

            self._optimized_controls = opt_ctrl
        return self._optimized_controls

    @property
    def guess_controls(self):
        if self._guess_controls is None:
            gss_ctrl = []

            for Hc, x0 in zip(self.objectives[0].H[1:], self.guess_params):

                control = Hc[1]
                if callable(control):
                    c0 = []
                    try:
                        tslots = self.time_interval.tslots
                    except Exception:
                        print(
                            "time_interval.tslots not specified "
                            "(probably missing n_tslots), defaulting to 100 "
                            "collocation points for result.optimized_controls")
                        tslots = np.linspace(
                            0., self.time_interval.evo_time, 100)

                    for t in tslots:
                        c0.append(control(t, x0))
                else:
                    c0 = x0
                gss_ctrl.append(c0)

            self._guess_controls = gss_ctrl
        return self._guess_controls

    @property
    def optimized_objectives(self):
        """
        """
        if self._optimized_objectives is None:
            opt_obj = []

            for obj in self.objectives:
                optimized_H = [obj.H[0]]

                for Hc, cf in zip(obj.H[1:], self.optimized_controls):
                    control = Hc[1]

                    if callable(control):
                        optimized_H = obj.H
                        break
                    else:
                        optimized_H.append([Hc[0], cf])

                opt_obj.append(
                    Objective(obj.initial, optimized_H, obj.target)
                )

            self._optimized_objectives = opt_obj
        return self._optimized_objectives

    @property
    def final_states(self):
        if self._final_states is None:
            states = []

            if self.var_time:  # last parameter is optimized time
                evo_time = self.optimized_params[-1][0]
            else:
                evo_time = self.time_interval.evo_time

            # extract parameter names from control functions f(t, para_key)
            c_sigs = [signature(Hc[1]) for Hc in self.objectives[0].H[1:]]
            c_keys = [sig.parameters.keys() for sig in c_sigs]
            para_keys = [list(keys)[1] for keys in c_keys]

            args_dict = {}
            for key, val in zip(para_keys, self.optimized_params):
                args_dict[key] = val

            # choose solver method based on type of control function
            if isinstance(
                    self.optimized_objectives[0].H[1][1],
                    jaxlib.xla_extension.PjitFunction):
                method = "diffrax"  # for JAX defined contols
            else:
                method = "adams"

            for obj in self.optimized_objectives:

                H = qt.QobjEvo(obj.H, args=args_dict)
                solver = None

                if obj.H[0].issuper:  # choose solver
                    solver = qt.MESolver(
                        H, options={
                            'normalize_output': False,
                            'method': method,
                        }
                    )
                else:
                    solver = qt.SESolver(
                        H, options={
                            'normalize_output': False,
                            'method': method,
                        }
                    )

                states.append(  # compute evolution
                    solver.run(obj.initial, tlist=[0., evo_time]).final_state
                )

            self._final_states = states
        return self._final_states

    @final_states.setter
    def final_states(self, states):
        self._final_states = states

    def update(self, infidelity, parameters):
        self.infidelity = infidelity
        self.new_params = parameters

    def dump(self, filename):
        with open(filename, 'wb') as dump_fh:
            pickler = pickle.Pickler(dump_fh)
            pickler.dump(self)

    @classmethod
    def load(cls, filename, objectives=None):
        with open(filename, 'rb') as dump_fh:
            result = pickle.load(dump_fh)
        result.objectives = objectives
        return result
