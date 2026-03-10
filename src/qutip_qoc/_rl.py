"""
This module contains functions that implement quantum optimal control
using reinforcement learning (RL) techniques, allowing for the optimization
of control pulse sequences in quantum systems.
"""
import qutip as qt
from qutip import Qobj
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

        super(_RL, self).__init__()

        self._Hd_lst, self._Hc_lst = [], []
        for objective in objectives:
            # extract drift and control Hamiltonians from the objective
            self._Hd_lst.append(objective.H[0])
            self._Hc_lst.append(
                [H[0] if isinstance(H, list) else H for H in objective.H[1:]]
            )

        def create_pulse_func(idx):
            """
            Create a control pulse lambda function for a given index.
            """
            return lambda t, args: self._pulse(t, args, idx + 1)

        # create the QobjEvo with Hd, Hc and controls(args)
        self._H_lst = [self._Hd_lst[0]]
        dummy_args = {f"alpha{i+1}": 1.0 for i in range(len(self._Hc_lst[0]))}
        for i, Hc in enumerate(self._Hc_lst[0]):
            self._H_lst.append([Hc, create_pulse_func(i)])
        self._H = qt.QobjEvo(self._H_lst, args=dummy_args)

        self.shorter_pulses = alg_kwargs.get(
            "shorter_pulses", False
        )  # lengthen the training to look for pulses of shorter duration, therefore episodes with fewer steps

        # extract bounds for control_parameters
        bounds = []
        for key in control_parameters.keys():
            bounds.append(control_parameters[key].get("bounds"))
        self._lbound = [b[0][0] for b in bounds]
        self._ubound = [b[0][1] for b in bounds]

        self._alg_kwargs = alg_kwargs

        self._initial = objectives[0].initial
        self._target = objectives[0].target
        self._state = None
        self._dim = self._initial.shape[0]

        self._result = Result(
            objectives=objectives,
            time_interval=time_interval,
            start_local_time=time.time(),  # initial optimization time
            n_iters=0,  # Number of iterations(episodes) until convergence
            iter_seconds=[],  # list containing the time taken for each iteration(episode) of the optimization
            var_time=True,  # Whether the optimization was performed with variable time
            guess_params=[],
        )

        self._backup_result = Result(  # used as a backup in case the algorithm with shorter_pulses does not find an episode with infid<target_infid
            objectives=objectives,
            time_interval=time_interval,
            start_local_time=time.time(),
            n_iters=0,
            iter_seconds=[],
            var_time=True,
            guess_params=[],
        )
        self._use_backup_result = (
            False  # if true, use self._backup_result as the final optimization result
        )

        # for the reward
        self._step_penalty = 1

        # To check if it exceeds the maximum number of steps in an episode
        self._current_step = 0

        self.terminated = False
        self.truncated = False
        self._episode_info = []  # to contain some information from the latest episode

        self._fid_err_targ = alg_kwargs["fid_err_targ"]

        # inferred attributes
        self._norm_fac = 1 / self._target.norm()

        self._temp_actions = []  # temporary list to save episode actions
        self._actions = []  # list of actions(lists) of the last episode

        # integrator options
        self._integrator_kwargs = integrator_kwargs
        self._rtol = self._integrator_kwargs.get("rtol", 1e-5)
        self._atol = self._integrator_kwargs.get("atol", 1e-5)

        self.max_episode_time = time_interval.evo_time  # maximum time for an episode
        self.max_steps = time_interval.n_tslots  # maximum number of steps in an episode
        self._step_duration = (
            time_interval.tslots[-1] / time_interval.n_tslots
        )  # step duration for mesvole
        self.max_episodes = alg_kwargs[
            "max_iter"
        ]  # maximum number of episodes for training
        self._total_timesteps = self.max_episodes * self.max_steps  # for learn() of gym
        self.current_episode = 0  # To keep track of the current episode

        # Define action and observation spaces (Gym)
        if self._initial.isket:
            obs_shape = (2 * self._dim,)
        else:  # for unitary operators
            obs_shape = (2 * self._dim * self._dim,)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(self._Hc_lst[0]),), dtype=np.float32
        )  # Continuous action space from -1 to +1, as suggested from gym
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=obs_shape, dtype=np.float32
        )  # Observation space

        # create the solver
        if self._Hd_lst[0].issuper:
            self._fid_type = self._alg_kwargs.get("fid_type", "TRACEDIFF")
            self._solver = qt.MESolver(H=self._H, options=self._integrator_kwargs)
        else:
            self._fid_type = self._alg_kwargs.get("fid_type", "PSU")
            self._solver = qt.SESolver(H=self._H, options=self._integrator_kwargs)

    def _pulse(self, t, args, idx):
        """
        Returns the control pulse value at time t for a given index.
        """
        alpha = args[f"alpha{idx}"]
        return alpha

    def _save_episode_info(self):
        """
        Save the information of the last episode before resetting the environment.
        """
        episode_data = {
            "episode": self.current_episode,
            "final_infidelity": self._result.infidelity,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "steps_used": self._current_step,
            "elapsed_time": time.time(),
        }
        self._episode_info.append(episode_data)

    def _infid(self, args):
        """
        The agent performs a step, then calculate infidelity to be minimized of the current state against the target state.
        """
        X = self._solver.run(
            self._state, [0.0, self._step_duration], args=args
        ).final_state
        self._state = X

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
        alphas = [
            ((action[i] + 1) / 2 * (self._ubound[0] - self._lbound[0]))
            + self._lbound[0]
            for i in range(len(action))
        ]

        args = {f"alpha{i+1}": value for i, value in enumerate(alphas)}
        _infidelity = self._infid(args)

        self._current_step += 1
        self._temp_actions.append(alphas)
        self._result.infidelity = _infidelity
        reward = (1 - _infidelity) - self._step_penalty

        self.terminated = (
            _infidelity <= self._fid_err_targ
        )  # the episode ended reaching the goal
        self.truncated = (
            self._current_step >= self.max_steps
        )  # if the episode ended without reaching the goal

        observation = self._get_obs()
        return observation, reward, bool(self.terminated), bool(self.truncated), {}

    def _get_obs(self):
        """
        Get the current state observation for the RL agent. Converts the system's
        quantum state or matrix into a real-valued NumPy array suitable for RL algorithms.
        """
        rho = self._state.full().flatten()
        obs = np.concatenate((np.real(rho), np.imag(rho)))
        return obs.astype(
            np.float32
        )  # Gymnasium expects the observation to be of type float32

    def reset(self, seed=None):
        """
        Reset the environment to the initial state, preparing for a new episode.
        """
        self._save_episode_info()

        time_diff = self._episode_info[-1]["elapsed_time"] - (
            self._episode_info[-2]["elapsed_time"]
            if len(self._episode_info) > 1
            else self._result.start_local_time
        )
        self._result.iter_seconds.append(time_diff)
        self._current_step = 0  # Reset the step counter
        self.current_episode += 1  # Increment episode counter
        self._actions = self._temp_actions.copy()
        self.terminated = False
        self.truncated = False
        self._temp_actions = []
        self._result._final_states = [self._state]
        self._state = self._initial
        return self._get_obs(), {}

    def _save_result(self):
        """
        Save the results of the optimization process, including the optimized
        pulse sequences, final states, and performance metrics.
        """
        result_obj = self._backup_result if self._use_backup_result else self._result

        if self._use_backup_result:
            self._backup_result.iter_seconds = self._result.iter_seconds.copy()
            self._backup_result._final_states = self._result._final_states.copy()
            self._backup_result.infidelity = self._result.infidelity

        result_obj.end_local_time = time.time()
        result_obj.n_iters = len(self._result.iter_seconds)
        result_obj.optimized_params = self._actions.copy() + [
            self._result.total_seconds
        ]  # If var_time is True, the last parameter is the evolution time
        result_obj._optimized_controls = self._actions.copy()
        result_obj._guess_controls = []
        result_obj._optimized_H = [self._H]

    def result(self):
        """
        Final conversions and return of optimization results
        """
        if self._use_backup_result:
            self._backup_result.start_local_time = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self._backup_result.start_local_time)
            )
            self._backup_result.end_local_time = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self._backup_result.end_local_time)
            )
            return self._backup_result
        else:
            self._save_result()
            self._result.start_local_time = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self._result.start_local_time)
            )
            self._result.end_local_time = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self._result.end_local_time)
            )
            return self._result

    def train(self):
        """
        Train the RL agent on the defined quantum control problem using the specified
        reinforcement learning algorithm. Checks environment compatibility with Gym API.
        """
        # Check if the environment follows Gym API
        check_env(self, warn=True)

        # Create the model
        model = PPO(
            "MlpPolicy", self, verbose=1
        )  # verbose = 1 to display training progress and statistics in the terminal

        stop_callback = EarlyStopTraining(verbose=1)

        # Train the model
        model.learn(total_timesteps=self._total_timesteps, callback=stop_callback)


class EarlyStopTraining(BaseCallback):
    """
    A callback to stop training based on specific conditions (steps, infidelity, max iterations)
    """

    def __init__(self, verbose: int = 0):
        super(EarlyStopTraining, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method is required by the BaseCallback class. We use it to stop the training.
        - Stop training if the maximum number of episodes is reached.
        - Stop training if it finds an episode with infidelity <= than target infidelity
        - If all of the last 100 episodes have infidelity below the target and use the same number of steps, stop training.
        """
        env = self.training_env.get_attr("unwrapped")[0]

        # Check if we need to stop training
        if env.current_episode >= env.max_episodes:
            if env._use_backup_result is True:
                env._backup_result.message = f"Reached {env.max_episodes} episodes, stopping training. Return the last founded episode with infid < target_infid"
            else:
                env._result.message = (
                    f"Reached {env.max_episodes} episodes, stopping training."
                )
            return False  # Stop training
        elif (env._result.infidelity <= env._fid_err_targ) and not (env.shorter_pulses):
            env._result.message = "Stop training because an episode with infidelity <= target infidelity was found"
            return False  # Stop training
        elif env.shorter_pulses:
            if (
                env._result.infidelity <= env._fid_err_targ
            ):  # if it finds an episode with infidelity lower than target infidelity, I'll save it in the meantime
                env._use_backup_result = True
                env._save_result()
            if len(env._episode_info) >= 100:
                last_100_episodes = env._episode_info[-100:]

                min_steps = min(info["steps_used"] for info in last_100_episodes)
                steps_condition = all(
                    ep["steps_used"] == min_steps for ep in last_100_episodes
                )
                infid_condition = all(
                    ep["final_infidelity"] <= env._fid_err_targ
                    for ep in last_100_episodes
                )

                if steps_condition and infid_condition:
                    env._use_backup_result = False
                    env._result.message = "Training finished. No episode in the last 100 used fewer steps and infidelity was below target infid."
                    return False  # Stop training
        return True  # Continue training
