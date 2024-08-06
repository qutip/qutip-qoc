"""
This module contains ...
"""
import qutip as qt
from qutip import Qobj, QobjEvo

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class _RL(gym.Env): # TODO: this should be similar to your GymQubitEnv(gym.Env) implementation
    """
    Class for storing a control problem and ...
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
        super().__init__() # TODO: super init your gym environment here

        # ------------------------------- copied from _GOAT class -------------------------------
        
        # TODO: you dont have to use (or keep them) if you don't need the following attributes
        # this is just an inspiration how to extract information from the input 

        self._Hd = objective.H[0]
        self._Hc_lst = objective.H[1:]

        self._control_parameters = control_parameters
        self._guess_params = guess_params
        self._H = self._prepare_generator()

        self._initial = objective.initial
        self._target = objective.target

        self._evo_time = time_interval.evo_time

        # inferred attributes
        self._norm_fac = 1 / self._target.norm()

        # integrator options
        self._integrator_kwargs = integrator_kwargs

        self._rtol = self._integrator_kwargs.get("rtol", 1e-5)
        self._atol = self._integrator_kwargs.get("atol", 1e-5)

        # choose solver and fidelity type according to problem
        if self._Hd.issuper:
            self._fid_type = alg_kwargs.get("fid_type", "TRACEDIFF")
            self._solver = qt.MESolver(H=self._H, options=self._integrator_kwargs)

        else:
            self._fid_type = alg_kwargs.get("fid_type", "PSU")
            self._solver = qt.SESolver(H=self._H, options=self._integrator_kwargs)

        self.infidelity = self._infid # TODO: should be used to calculate the reward

        # ----------------------------------------------------------------------------------------
        # TODO: set up your gym environment as you did correctly in post10
        self.max_episode_time = time_interval.evo_time                  # maximum time for an episode
        self.max_steps = time_interval.n_tslots                         # maximum number of steps in an episode
        self.step_duration = time_interval.tslots[-1] / time_interval.n_tslots  # step duration for mesvole()
        ...
        
        
        # ----------------------------------------------------------------------------------------
    
    def _infid(self, params):
        """
        Calculate infidelity to be minimized
        """
        X = self._solver.run(
            self._initial, [0.0, self._evo_time], args={"p": params}
        ).final_state

        if self._fid_type == "TRACEDIFF":
            diff = X - self._target
            # to prevent if/else in qobj.dag() and qobj.tr()
            diff_dag = Qobj(diff.data.adjoint(), dims=diff.dims)
            g = 1 / 2 * (diff_dag * diff).data.trace()
            infid = np.real(self._norm_fac * g)
        else:
            g = self._norm_fac * self._target.overlap(X)
            if self._fid_type == "PSU":  # f_PSU (drop global phase)
                infid = 1 - np.abs(g)
            elif self._fid_type == "SU":  # f_SU (incl global phase)
                infid = 1 - np.real(g)

        return infid

    # TODO: don't hesitate to add the required methods for your rl environment

    def step(self, action):
        ...

    def train(self):
        ...

    def result(self):
        # TODO: return qoc.Result object with the optimized pulse amplitudes
        ...