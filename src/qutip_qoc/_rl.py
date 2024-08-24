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
from stable_baselines3.common.callbacks import BaseCallback

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

        # extract bounds for control_parameters
        bounds = []
        for key in control_parameters.keys():
            bounds.append(control_parameters[key].get("bounds"))
        self.lbound = [b[0][0] for b in bounds]
        self.ubound = [b[0][1] for b in bounds]

        self._alg_kwargs = alg_kwargs
        self.shorter_pulses = self._alg_kwargs.get("shorter_pulses", False)     # lengthen the training to look for pulses of shorter duration, therefore episodes with fewer steps

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
            var_time = True,                        # Whether the optimization was performed with variable time
        )

        #for the reward
        self._step_penalty = 1

        # To check if it exceeds the maximum number of steps in an episode
        self.current_step = 0

        self.terminated = False
        self.truncated = False
        self.episode_info = []                # to contain some information from the latest episode

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
        self.current_episode = 0                                        # To keep track of the current episode
        
        # Define action and observation spaces (Gym)
        if self._initial.isket:
            obs_shape = (2 * self.dim,)
        else:   # for unitary operators 
            obs_shape = (2 * self.dim * self.dim,)
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self._Hc_lst[0]),), dtype=np.float32)     # Continuous action space from -1 to +1, as suggested from gym
        self.observation_space = spaces.Box(low=-1, high=1, shape=obs_shape, dtype=np.float32)              # Observation space
        
        # create the solver
        if self._Hd_lst[0].issuper:
            self._fid_type = self._alg_kwargs.get("fid_type", "TRACEDIFF")
            self._solver = qt.MESolver(H=self._H, options=self._integrator_kwargs)
        else:
            self._fid_type = self._alg_kwargs.get("fid_type", "PSU")
            self._solver = qt.SESolver(H=self._H, options=self._integrator_kwargs)

    def pulse(self, t, args, idx):
        """
        Returns the control pulse value at time t for a given index.
        """
        return 1*args[f"alpha{idx}"]
    
    def save_episode_info(self):
        """
        Save the information of the last episode before resetting the environment.
        """
        episode_data = {
            "episode": self.current_episode,
            "final_infidelity": self._result.infidelity,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "steps_used": self.current_step,
            "elapsed_time": time.mktime(time.localtime())
        }
        self.episode_info.append(episode_data)

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

        infidelity = self._infid()

        self.current_step += 1
        self.temp_actions.append(alphas)
        self._result.infidelity = infidelity
        reward = (1 - infidelity) - self._step_penalty

        self.terminated = infidelity <= self._fid_err_targ                       # the episode ended reaching the goal
        self.truncated = self.current_step >= self.max_steps                     # if the episode ended without reaching the goal

        observation = self._get_obs()
        return observation, reward, bool(self.terminated), bool(self.truncated), {}

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
        self.save_episode_info()

        time_diff = self.episode_info[-1]["elapsed_time"] - (self.episode_info[-2]["elapsed_time"] if len(self.episode_info) > 1 else time.mktime(self._result.start_local_time))
        self._result.iter_seconds.append(time_diff)
        self.current_step = 0                                           # Reset the step counter
        self.current_episode += 1                                       # Increment episode counter
        self.actions = self.temp_actions.copy()
        self.terminated = False
        self.truncated = False
        self.temp_actions = []
        self.state = self._initial
        return self._get_obs(), {}
    
    def result(self):
        """
        Retrieve the results of the optimization process, including the optimized 
        pulse sequences, final states, and performance metrics.
        """
        self._result.end_local_time = time.localtime()
        self._result.n_iters = len(self._result.iter_seconds)  
        self._result.optimized_params = self.actions.copy() + [self._result.total_seconds]                    # If var_time is True, the last parameter is the evolution time
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
        
        stop_callback = EarlyStopTraining(verbose=1)
        
        # Train the model
        model.learn(total_timesteps = self.total_timesteps, callback=stop_callback)

class EarlyStopTraining(BaseCallback):
    """
    A callback to stop training based on specific conditions (steps, infidelity, max iterations)
    """
    def __init__(self, verbose: int = 0):
        super(EarlyStopTraining, self).__init__(verbose)
        self.stop_train = False

    def _on_step(self) -> bool:
        """
        This method is required by the BaseCallback class. We use it to stop the training.
        - Stop training if the maximum number of episodes is reached.
        - Stop training if it finds an episode with infidelity <= than target infidelity
        """
        env = self.training_env.envs[0].unwrapped

        # Check if we need to stop training
        if self.stop_train:
            return False  # Stop training
        elif env.current_episode >= env.max_episodes:
            env._result.message = f"Reached {env.max_episodes} episodes, stopping training."
            return False    # Stop training
        elif (env._result.infidelity <= env._fid_err_targ) and not(env.shorter_pulses):
            env._result.message = f"Stop training because an episode with infidelity <= target infidelity was found"
            return False  # Stop training
        return True  # Continue training

    def _on_rollout_start(self) -> None:
        """
        This method is called before the rollout starts (before collecting new samples).
        Checks:
        - If all of the last 100 episodes have infidelity below the target and use the same number of steps, stop training.
        """
        #could be moved to on_step
        
        env = self.training_env.envs[0].unwrapped
        #Only if specified in alg_kwargs, the algorithm will search for shorter pulses, resulting in episodes with fewer steps.
        if env.shorter_pulses:
            if len(env.episode_info) >= 100:
                last_100_episodes = env.episode_info[-100:]

                min_steps = min(info['steps_used'] for info in last_100_episodes)
                steps_condition = all(ep['steps_used'] == min_steps for ep in last_100_episodes)
                infid_condition = all(ep['final_infidelity'] <= env._fid_err_targ for ep in last_100_episodes)

                if steps_condition and infid_condition:
                    env._result.message = "Training finished. No episode in the last 100 used fewer steps and infidelity was below target infid."
                    self.stop_train = True  # Stop training