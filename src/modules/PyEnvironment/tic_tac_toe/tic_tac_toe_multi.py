import numpy as np

from .tic_tac_toe_environment import TicTacToeEnvironment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType, TimeStep

from ..env_flags import REWARD_ILLEGAL_MOVE


class TicTacToeMultiAgentEnv(TicTacToeEnvironment):

    def __init__(self, rng: np.random.RandomState = None, discount=1.0):
        """Initializes TicTacToeEnvironment.

    Args:
      rng: If a random generator is provided, the opponent will choose a random
        empty space. If None is provided, the opponent will choose the first
        empty space.
      discount: Discount for reward.
    """
        super(TicTacToeEnvironment, self).__init__()
        self._rng = rng
        self._discount = np.asarray(discount, dtype=np.float32)

        self._states = None

    def action_spec(self):
        position_spec = BoundedArraySpec((), np.int32, minimum=0, maximum=8)
        value_spec = BoundedArraySpec((), np.int32, minimum=1, maximum=2)
        return {
            'position': position_spec,
            'value': value_spec
        }

    def observation_spec(self):
        state_spec = BoundedArraySpec((3, 3), np.int32, minimum=0, maximum=2)
        mask_spec = BoundedArraySpec((9,), np.int32, minimum=0, maximum=1)
        return {
            'state': state_spec,
            'mask': mask_spec
        }

    def _reset(self):
        self._states = np.zeros((3, 3), np.int32)
        mask = np.ones((9,), np.int32)
        observation = {'state': self._states, 'mask': mask}
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32),
                        self._discount, observation)

    def set_state(self, time_step: TimeStep):
        self._current_time_step = time_step
        self._states = time_step.observation['state']

    def get_legal_actions(self, state):
        legal_moves = np.zeros(self._states.shape, dtype=np.int32)
        for y in range(len(state)):
            for x in range(len(state[0])):
                if state[x, y] == 0:
                    legal_moves[x, y] = 1
        return legal_moves

    def _step(self, action: np.ndarray):
        if self._current_time_step.is_last():
            return self._reset()

        mask_square = self.get_legal_actions(self._states)
        mask = mask_square.reshape((9,))

        index_flat = np.array(range(9)) == action['position']
        index = index_flat.reshape(self._states.shape) == True
        if self._states[index] != 0:
            observation = {'state': self._states, 'mask': mask}
            print('illegal move')
            return TimeStep(StepType.LAST,
                            REWARD_ILLEGAL_MOVE,
                            self._discount,
                            observation)

        self._states[index] = action['value']

        is_final, reward = self._check_states(self._states)

        if np.all(self._states == 0):
            step_type = StepType.FIRST
        elif is_final:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        observation = {'state': self._states, 'mask': mask}
        print('legal move')
        return TimeStep(step_type, reward, self._discount, observation)

    def console_print(self):
        table_str = '''
        {} | {} | {}
        - + - + -
        {} | {} | {}
        - + - + -
        {} | {} | {}
        '''.format(*tuple(self.get_state().flatten()))
        table_str = table_str.replace('0', ' ')
        table_str = table_str.replace('1', 'X')
        table_str = table_str.replace('2', 'O')
        print(table_str)
