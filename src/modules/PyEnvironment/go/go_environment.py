import copy
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep

from ..env_flags import REWARD_ILLEGAL_MOVE, REWARD_LOSS, REWARD_WIN, REWARD_DRAW_OR_NOT_FINAL, REWARD_NOT_PASSED


class GoEnvironment(py_environment.PyEnvironment):

    def __init__(self, discount=1.0):
        """Initializes DraughtsEnvironment.

        Args:
          discount: Discount for reward.
        """
        super(GoEnvironment, self).__init__()
        self._discount = np.asarray(discount, dtype=np.float32)

        self._states = None

        self._mask = None

        self._num_moves = 0
        self._max_moves = 500
        self._passed = False

    def action_spec(self):
        position_spec = BoundedArraySpec((), np.int32, minimum=0, maximum=361)
        value_spec = BoundedArraySpec((), np.int32, minimum=1, maximum=2)
        return {
            'position': position_spec,
            'value': value_spec
        }

    def observation_spec(self):
        state_spec = BoundedArraySpec((19, 19), np.int32, minimum=0, maximum=2)
        mask_spec = BoundedArraySpec((362,), np.int32, minimum=0, maximum=1)
        return {
            'state': state_spec,
            'mask': mask_spec
        }

    def _reset(self):
        self._states = np.zeros((19, 19))
        self._mask = self.get_legal_moves(self._states, 1)
        observation = {'state': self._states, 'mask': self._mask}
        self._num_moves = 0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32),
                        self._discount, observation)

    def check_legal_tiles(self, state, position, player):
        if state[position] == 0:
            return True
        return False

    def check_legal_place(self, state, position, player):
        if self.check_legal_tiles(state, position, player):
            return True
        return False

    def get_state(self) -> TimeStep:
        # Returning an unmodifiable copy of the state.
        return copy.deepcopy(self._current_time_step)

    def set_state(self, time_step: TimeStep):
        self._current_time_step = time_step
        self._states = time_step.observation

    def get_legal_moves(self, state, player):
        legal_flat = np.zeros((362,), np.int32)
        # Players can always pass
        legal_flat[361] = 1
        # Loop through each position on the board checking for legal moves
        for y in range(len(state)):
            for x in range(len(state[0])):
                position = (y, x)
                if self.check_legal_place(state, position, player):
                    position_flat = (position[0] * 19) + position[1]
                    legal_flat[position_flat] = 1
        return legal_flat

    def _step(self, action: np.ndarray):
        if self._current_time_step.is_last():
            return self._reset()

        self._num_moves += 1

        next_player = action['value']
        illegal = False
        passed = False

        index_flat = (np.array(range(362)) == action['position']).reshape(1, 362)
        index_flat = index_flat / index_flat.sum()
        if np.isnan(index_flat).any():
            observation = {'state': self._states, 'mask': self._mask}
            return TimeStep(StepType.LAST,
                            REWARD_ILLEGAL_MOVE,
                            self._discount,
                            observation)
        index = np.random.choice(range(362), p=np.squeeze(index_flat))

        position = (index // 19, index % 19)
        if position[0] == 19:
            passed = True
        else:
            if self.check_legal_place(self._states, position, action['value']):
                self._states[position] = action['value']
            else:
                illegal = True

        is_final, reward = self._check_states(self._states, action['value'], passed)

        if np.all(self._states == 0):
            step_type = StepType.FIRST
        elif is_final:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        if not illegal:
            if action['value'] == 1:
                next_player = 2
            else:
                next_player = 1

        self._mask = self.get_legal_moves(self._states, next_player)

        observation = {'state': self._states, 'mask': self._mask}
        if illegal and (not is_final):
            return TimeStep(step_type, REWARD_ILLEGAL_MOVE, self._discount, observation)
        else:
            return TimeStep(step_type, reward, self._discount, observation)

    def _check_states(self, states: np.ndarray, player: int, player_pass):
        """Check if the given states are final and calculate reward.

        Args:
          states: states of the board.

        Returns:
          A tuple of (is_final, reward) where is_final means whether the states
          are final are not, and reward is the reward for stepping into the states
          The meaning of reward: 0 = not decided or draw, 1 = win, -1 = loss
        """
        if self._passed and player_pass:
            return True, REWARD_DRAW_OR_NOT_FINAL
        elif self._num_moves > self._max_moves:
            return True, REWARD_DRAW_OR_NOT_FINAL
        return False, REWARD_DRAW_OR_NOT_FINAL  # ongoing

    def console_print(self):
        table_str = '''
        {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {}
        '''.format(*tuple(self.get_state().observation['state'].flatten()))
        table_str = table_str.replace('0', ' ')
        table_str = table_str.replace('1', 'W')
        table_str = table_str.replace('2', 'B')
        print(table_str)
