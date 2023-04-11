import copy
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep


class ChessEnvironment(py_environment.PyEnvironment):

    REWARD_WIN = np.asarray(1., dtype=np.float32)
    REWARD_LOSS = np.asarray(-1., dtype=np.float32)
    REWARD_DRAW_OR_NOT_FINAL = np.asarray(0., dtype=np.float32)
    # A very small number such that it does not affect the value calculation.
    REWARD_ILLEGAL_MOVE = np.asarray(-.001, dtype=np.float32)

    REWARD_WIN.setflags(write=False)
    REWARD_LOSS.setflags(write=False)
    REWARD_DRAW_OR_NOT_FINAL.setflags(write=False)

    def __init__(self, rng: np.random.RandomState = None, discount=1.0):
        """Initializes ChessEnvironment.

        Args:
          rng: If a random generator is provided, the opponent will choose a random
            empty space. If None is provided, the opponent will choose the first
            empty space.
          discount: Discount for reward.
        """
        super(ChessEnvironment, self).__init__()
        self._rng = rng
        self._discount = np.asarray(discount, dtype=np.float32)

        self._states = None
