from typing import Callable

from tensorflow import TensorSpec
from tf_agents.environments import TFPyEnvironment
from tf_agents.trajectories import TimeStep


class HumanAgent:
    def __init__(self,
                 env: TFPyEnvironment,
                 action_spec: TensorSpec = None,
                 action_fn: Callable = lambda action: action,
                 name: str = 'HumanAgent',
                 **dqn_kwargs):
        self._action_spec = action_spec
        self._env = env
        self._action_fn = action_fn
        self._name = name

    def _step_environment(self, action) -> TimeStep:
        action = self._action_fn(action)
        time_step = self._env.step(action)
        return time_step

    def act(self, action):
        next_time_step = self._step_environment(action)
        return 0, next_time_step
