from typing import Callable

import tensorflow as tf
from tensorflow import TensorSpec
from tf_agents.environments import TFPyEnvironment
from tf_agents.trajectories import TimeStep


class HumanAgent:
    def __init__(self,
                 env: TFPyEnvironment,
                 action_spec: TensorSpec = None,
                 observation_spec: TensorSpec = None,
                 action_fn: Callable = lambda action: action,
                 name: str = 'HumanAgent',
                 reward_fn: Callable = lambda time_step: time_step.reward,
                 **dqn_kwargs):
        self._action_spec = action_spec
        self._env = env
        self._action_fn = action_fn
        self._name = name
        self._reward_fn = reward_fn
        self._observation_spec = observation_spec or self._env.observation_spec()

    def _observation_fn(self, observation: tf.Tensor):
        """
            Takes a tensor with specification self._env.observation_spec
            and extracts a tensor with specification self._observation_spec.

            For example, consider an agent within an NxN maze environment.
            The env could expose the entire NxN integer matrix as an observation
            but we would prefer the agent to only see a 3x3 window around their
            current location. To do this we can override this method.

            This allows us to have different agents acting in the same environment
            with different observations.
        """
        return observation

    def _augment_time_step(self, time_step: TimeStep) -> TimeStep:
        reward = self._reward_fn(time_step)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        if reward.shape != time_step.reward.shape:
            reward = tf.reshape(reward, time_step.reward.shape)

        observation = self._observation_fn(time_step.observation)

        return TimeStep(
            step_type=time_step.step_type,
            reward=reward,
            discount=time_step.discount,
            observation=observation
        )

    def _step_environment(self, action) -> TimeStep:
        action = self._action_fn(action)
        time_step = self._env.step(action)
        time_step = self._augment_time_step(time_step)
        return time_step

    def act(self, action):
        next_time_step = self._step_environment(action)
        return 0, next_time_step.reward
