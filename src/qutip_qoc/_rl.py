"""
This module contains ...
"""
import qutip as qt
from qutip import Qobj, QobjEvo
from qutip_qoc import Result

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import time

class _RL(gym.Env): # TODO: this should be similar to your GymQubitEnv(gym.Env) implementation
    """
    Class for storing a control problem and ...
    """

    def __init__(
        self,
        objectives,
        control_parameters,
        time_interval,
        time_options,
        alg_kwargs,
        optimizer_kwargs,
        minimizer_kwargs,
        integrator_kwargs,
        qtrl_optimizers,
    ):
        super(_RL,self).__init__() # TODO:(ok) super init your gym environment here

        # ------------------------------- copied from _GOAT class -------------------------------
        
        # TODO: you dont have to use (or keep them) if you don't need the following attributes
        # this is just an inspiration how to extract information from the input 
        self._Hd_lst, self._Hc_lst = [], []
        if not isinstance(objectives, list):
            objectives = [objectives]
        for objective in objectives:
            # extract drift and control Hamiltonians from the objective
            self._Hd_lst.append(objective.H[0])
            self._Hc_lst.append([H[0] if isinstance(H, list) else H for H in objective.H[1:]])

        # create the QobjEvo with Hd, Hc and controls(args)
        self.args = {f"alpha{i+1}": (1) for i in range(len(self._Hc_lst[0]))}    # set the control parameters to 1 for all the Hc
        self._H_lst = [self._Hd_lst[0]]
        for i, Hc in enumerate(self._Hc_lst[0]):
            self._H_lst.append([Hc, lambda t, args: self.pulse(t, self.args, i+1)])
        self._H = qt.QobjEvo(self._H_lst, self.args)

        self._control_parameters = control_parameters
        # extract bounds for _control_parameters
        bounds = []
        for key in control_parameters.keys():
            bounds.append(control_parameters[key].get("bounds"))
        self.lbound = [b[0][0] for b in bounds]
        self.ubound = [b[0][1] for b in bounds]

        #self._H = self._prepare_generator()
        self._alg_kwargs = alg_kwargs

        self._initial = objective.initial
        self._target = objective.target
        self.state = None

        self._result = Result(
            objectives = objectives,
            time_interval = time_interval,
            start_local_time = time.localtime(),    # initial optimization time
            n_iters = 0,                            # Number of iterations(episodes) until convergence 
            iter_seconds = [],                      # list containing the time taken for each iteration(episode) of the optimization
            var_time = False,                       # Whether the optimization was performed with variable time
        )

        #for the reward
        self._step_penalty = 1

        # To check if it exceeds the maximum number of steps in an episode
        self.current_step = 0

        #self._evo_time = time_interval.evo_time
        self._fid_err_targ = alg_kwargs["fid_err_targ"]

        # inferred attributes
        self._norm_fac = 1 / self._target.norm()

        self.temp_actions = []                  # temporary list to save episode actions
        self.actions = []                       # list of actions(lists) of the last episode

        # integrator options
        self._integrator_kwargs = integrator_kwargs
        self._rtol = self._integrator_kwargs.get("rtol", 1e-5)
        self._atol = self._integrator_kwargs.get("atol", 1e-5)

        # ----------------------------------------------------------------------------------------
        # TODO: set up your gym environment as you did correctly in post10
        self.max_episode_time = time_interval.evo_time                  # maximum time for an episode
        self.max_steps = time_interval.n_tslots                         # maximum number of steps in an episode
        self.step_duration = time_interval.tslots[-1] / time_interval.n_tslots  # step duration for mesvole
        self.max_episodes = alg_kwargs["max_iter"]                      # maximum number of episodes for training
        self.total_timesteps = self.max_episodes * self.max_steps       # for learn() of gym
        
        # Define action and observation spaces (Gym)
        #self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Continuous action space from -1 to +1, as suggested from gym
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self._Hc_lst[0]),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # Observation space, |v> have 2 real and 2 imaginary numbers -> 4
        # ----------------------------------------------------------------------------------------
        
    def update_solver(self): 
        # choose solver and fidelity type according to problem
        if self._Hd_lst[0].issuper:
            self._fid_type = self._alg_kwargs.get("fid_type", "TRACEDIFF")
            self._solver = qt.MESolver(H=self._H, options=self._integrator_kwargs)
        else:
            self._fid_type = self._alg_kwargs.get("fid_type", "PSU")
            self._solver = qt.SESolver(H=self._H, options=self._integrator_kwargs)

        self.infidelity = self._infid # TODO: should be used to calculate the reward

    #def _pulse(self, t, p):
    #    return p
    def pulse(self, t, args, idx):
        return 1*args[f"alpha{idx}"]

    def _infid(self, params=None):
        """
        Calculate infidelity to be minimized
        """
        X = self._solver.run(
            self.state, [0.0, self.step_duration], args={"p": params}
        ).final_state
        self.state = X

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
        #infid = 1 - qt.fidelity(self.state, self._target)
        return infid

    # TODO: don't hesitate to add the required methods for your rl environment

    def step(self, action):
        alphas = [((action[i] + 1) / 2 * (self.ubound[0] - self.lbound[0])) + self.lbound[0] for i in range(len(action))]   #TODO: use ubound[i] lbound[i] 

        for i, value in enumerate(alphas):
            self.args[f"alpha{i+1}"] = value
        self._H = qt.QobjEvo(self._H_lst, self.args)

        self.update_solver()                # _H has changed
        infidelity = self.infidelity()

        self.current_step += 1
        self.temp_actions.append(alphas)
        self._result.infidelity = infidelity
        reward = (1 - infidelity) - self._step_penalty

        terminated = infidelity <= self._fid_err_targ                       # the episode ended reaching the goal
        truncated = self.current_step >= self.max_steps                     # if the episode ended without reaching the goal

        if terminated or truncated:
            time_diff = time.mktime(time.localtime()) - time.mktime(self._result.start_local_time)
            self._result.iter_seconds.append(time_diff)
            self.current_step = 0                                           # Reset the step counter
            self.actions = self.temp_actions.copy()

        observation = self._get_obs()
        return observation, reward, bool(terminated), bool(truncated), {}

    def _get_obs(self):
        rho = self.state.full().flatten()                                   # to have state vector as NumPy array and flatten into one dimensional array.[a+i*b c+i*d]
        obs = np.concatenate((np.real(rho), np.imag(rho)))
        return obs.astype(np.float32)                                       # Gymnasium expects the observation to be of type float32
    
    def reset(self, seed=None):
        self.temp_actions = []
        self.state = self._initial
        return self._get_obs(), {}
    
    def result(self):
        # TODO: return qoc.Result object with the optimized pulse amplitudes
        self._result.message = "Optimization finished!"
        self._result.end_local_time = time.localtime()
        self._result.n_iters = len(self._result.iter_seconds)  
        self._result.optimized_params = self.actions.copy()
        self._result._optimized_controls = self.actions.copy()
        self._result._final_states = (self._result._final_states if self._result._final_states is not None else []) + [self.state]
        self._result.start_local_time = time.strftime("%Y-%m-%d %H:%M:%S", self._result.start_local_time)       # Convert to a string
        self._result.end_local_time = time.strftime("%Y-%m-%d %H:%M:%S", self._result.end_local_time)           # Convert to a string
        self._result._guess_controls = []
        self._result._optimized_H = [self._H]
        self._result.guess_params = []
        return self._result

    def train(self):
        # Check if the environment follows Gym API
        check_env(self, warn=True)

        # Create the model
        model = PPO('MlpPolicy', self, verbose=1)       # verbose = 1 to show training in terminal

        # Train the model
        model.learn(total_timesteps = self.total_timesteps)