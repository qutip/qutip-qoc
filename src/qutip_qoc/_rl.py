"""
This module contains functions that implement quantum optimal control 
using reinforcement learning (RL) techniques, allowing for the optimization 
of control pulse sequences in quantum systems.
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

class _RL(gym.Env):
    """
    Class for storing a control problem and implementing quantum optimal 
    control using reinforcement learning. This class defines a custom 
    Gym environment that models the dynamics of quantum systems 
    under various control pulses, and uses RL algorithms to optimize the 
    parameters of these pulses.
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
        """
        Initialize the reinforcement learning environment for quantum 
        optimal control. Sets up the system Hamiltonian, control parameters, 
        and defines the observation and action spaces for the RL agent.
        """

        super(_RL,self).__init__()
        
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

        self._alg_kwargs = alg_kwargs

        self._initial = objectives[0].initial
        self._target = objectives[0].target
        self.state = None
        self.dim = self._initial.shape[0]

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

        self._fid_err_targ = alg_kwargs["fid_err_targ"]

        # inferred attributes
        self._norm_fac = 1 / self._target.norm()

        self.temp_actions = []                  # temporary list to save episode actions
        self.actions = []                       # list of actions(lists) of the last episode

        # integrator options
        self._integrator_kwargs = integrator_kwargs
        self._rtol = self._integrator_kwargs.get("rtol", 1e-5)
        self._atol = self._integrator_kwargs.get("atol", 1e-5)

        self.max_episode_time = time_interval.evo_time                  # maximum time for an episode
        self.max_steps = time_interval.n_tslots                         # maximum number of steps in an episode
        self.step_duration = time_interval.tslots[-1] / time_interval.n_tslots  # step duration for mesvole
        self.max_episodes = alg_kwargs["max_iter"]                      # maximum number of episodes for training
        self.total_timesteps = self.max_episodes * self.max_steps       # for learn() of gym
        
        # Define action and observation spaces (Gym)
        if self._initial.isket:
            obs_shape = (2 * self.dim,)
        else:   # for unitary operators 
            obs_shape = (2 * self.dim * self.dim,)
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self._Hc_lst[0]),), dtype=np.float32)     # Continuous action space from -1 to +1, as suggested from gym
        self.observation_space = spaces.Box(low=-1, high=1, shape=obs_shape, dtype=np.float32)              # Observation space
        
    def update_solver(self): 
        """
        Update the solver and fidelity type based on the problem setup.
        Chooses the appropriate solver (Schrödinger or master equation) and 
        prepares for infidelity calculation.
        """
        if self._Hd_lst[0].issuper:
            self._fid_type = self._alg_kwargs.get("fid_type", "TRACEDIFF")
            self._solver = qt.MESolver(H=self._H, options=self._integrator_kwargs)
        else:
            self._fid_type = self._alg_kwargs.get("fid_type", "PSU")
            self._solver = qt.SESolver(H=self._H, options=self._integrator_kwargs)

        self.infidelity = self._infid

    def pulse(self, t, args, idx):
        """
        Returns the control pulse value at time t for a given index.
        """
        return 1*args[f"alpha{idx}"]

    def _infid(self, params=None):
        """
        The agent performs a step, then calculate infidelity to be minimized of the current state against the target state.
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
        return infid

    def step(self, action):
        """
        Perform a single time step in the environment, applying the scaled action (control pulse) 
        chosen by the RL agent. Updates the system's state and computes the reward.
        """
        alphas = [((action[i] + 1) / 2 * (self.ubound[0] - self.lbound[0])) + self.lbound[0] for i in range(len(action))] 

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
        """
        Get the current state observation for the RL agent. Converts the system's 
        quantum state or matrix into a real-valued NumPy array suitable for RL algorithms.
        """
        rho = self.state.full().flatten()
        obs = np.concatenate((np.real(rho), np.imag(rho)))
        return obs.astype(np.float32)                                       # Gymnasium expects the observation to be of type float32
    
    def reset(self, seed=None):
        """
        Reset the environment to the initial state, preparing for a new episode.
        """
        self.temp_actions = []
        self.state = self._initial
        return self._get_obs(), {}
    
    def result(self):
        """
        Retrieve the results of the optimization process, including the optimized 
        pulse sequences, final states, and performance metrics.
        """
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
        """
        Train the RL agent on the defined quantum control problem using the specified 
        reinforcement learning algorithm. Checks environment compatibility with Gym API.
        """
        # Check if the environment follows Gym API
        check_env(self, warn=True)

        # Create the model
        model = PPO('MlpPolicy', self, verbose=1)       # verbose = 1 to display training progress and statistics in the terminal
        # Train the model
        model.learn(total_timesteps = self.total_timesteps)