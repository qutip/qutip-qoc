import time
import pickle
import textwrap
import numpy as np
from inspect import signature
import qutip as qt
from objective import Objective


class Result():
    def __init__(
            self,
            objectives=None,
            time_interval=None,
            start_local_time=None,
            end_local_time=None,
            iters=None,
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
            var_time=False
    ):
        self.time_interval = time_interval
        self.objectives = objectives
        self.start_local_time = start_local_time
        self.end_local_time = end_local_time
        self.iters = iters
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
                start_local_time=time.strftime(
                    '%Y-%m-%d %H:%M:%S', self.start_local_time),
                n_objectives=len(self.objectives),
                final_infid=self.infidelity,
                final_params=self.optimized_params,
                n_iters=self.iters,
                end_local_time=time.strftime(
                    '%Y-%m-%d %H:%M:%S', self.end_local_time),
                time_delta=self.time_delta,
                message=self.message)
        ).strip()

    def __repr__(self):
        return self.__str__()

    def start_time(self):
        self.start_local_time = self.iter_time = time.time()
        self.elapsed_time = 0
        self.iter_seconds = []

    def end_time(self):
        end_local_time = time.time()
        # prepare information for printing
        self.time_delta = round(end_local_time - self.start_local_time, 4)
        self.start_local_time = time.localtime(self.start_local_time)
        self.end_local_time = time.localtime(end_local_time)

    def time_iter(self):
        """
        Calculates and stores the time
        after each iteration. (optimizer callback)
        """
        iter_time = time.time()
        diff = round(iter_time - self.iter_time, 4)
        self.iter_time = iter_time
        self.iter_seconds.append(diff)
        return diff

    def time_elapsed(self):
        self.elapsed_time = round(time.time() - self.start_local_time, 4)
        return self.elapsed_time

    @property
    def optimized_params(self):
        if self._optimized_params is None:
            # reshape (optimized) new_parameters array to match
            # shape and type of the guess_parameters list
            opt_params = []
            for i, guess in enumerate(self.guess_params):
                opt_params.append(type(guess)(
                    self.new_params[i: i + len(guess)])
                )
            self._optimized_params = opt_params
        return self._optimized_params

    @optimized_params.setter
    def optimized_params(self, params):
        self._optimized_params = params

    @property
    def optimized_controls(self):
        """
        """
        if self._optimized_controls is None:
            opt_ctrl = []

            for Hc, xf in zip(self.objectives[0].H_evo[1:], self.optimized_params):

                control = Hc[1]
                if callable(control):
                    cf = []
                    for t in self.time_interval.tlist:
                        cf.append(control(t, xf))
                else:
                    cf = xf
                opt_ctrl.append(cf)
            self._optimized_controls = opt_ctrl

        return self._optimized_controls

    @property
    def guess_controls(self):
        """
        """
        if self._guess_controls is None:
            gss_ctrl = []

            for Hc, x0 in zip(self.objectives[0].H_evo[1:], self.guess_params):

                control = Hc[1]
                if callable(control):
                    c0 = []
                    for t in self.time_interval.tlist:
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
                optimized_H = [obj.H_evo[0]]

                for Hc, cf in zip(obj.H_evo[1:], self.optimized_controls):
                    control = Hc[1]

                    if callable(control):
                        optimized_H = obj.H_evo
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
                evo_time = self.optimized_params[-1]
            else:
                evo_time = self.time_interval.evo_time

            # extract parameter names from control functions f(t, para_key)
            c_sigs = [signature(Hc[1]) for Hc in self.objectives[0].H_evo[1:]]
            c_keys = [sig.parameters.keys() for sig in c_sigs]
            para_keys = [list(keys)[1] for keys in c_keys]

            args_dict = {}
            for key, val in zip(para_keys, self.optimized_params):
                args_dict[key] = val

            for obj in self.optimized_objectives:
                states.append(
                    qt.mesolve(
                        obj.H_evo,
                        obj.initial,
                        tlist=[0., evo_time],
                        args=args_dict,
                        options={'normalize_output': False}
                    ).final_state
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
        """Dump the :class:`Result` to a binary :mod:`pickle` file.

        The original :class:`Result` object can be restored from the resulting
        file using :meth:`load`. However, time-dependent control fields that
        are callables/functions will not be preserved, as they are not
        "pickleable".

        Args:
            filename (str): Name of file to which to dump the :class:`Result`.
        """
        with open(filename, 'wb') as dump_fh:
            pickler = pickle.Pickler(dump_fh)
            # slf = copy.deepcopy(self)
            # slf.objectives = None
            # slf.optimized_objectives = None
            pickler.dump(self)

    @classmethod
    def load(cls, filename, objectives=None):
        """Construct :class:`Result` object from a :meth:`dump` file

        Args:
            filename (str): The file from which to load the :class:`Result`.
                Must be in the format created by :meth:`dump`.
            objectives (None or list[Objective]): If given, after loading
                :class:`Result` from the given `filename`, overwrite
                :attr:`objectives` with the given `objectives`. This is
                necessary because :meth:`dump` does not preserve time-dependent
                controls that are Python functions.
        Returns:
            Result: The :class:`Result` instance loaded from `filename`
        """
        with open(filename, 'rb') as dump_fh:
            result = pickle.load(dump_fh)
        result.objectives = objectives
        return result
