import copy
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep

from ..env_flags import REWARD_ILLEGAL_MOVE, REWARD_LOSS, REWARD_WIN, REWARD_DRAW_OR_NOT_FINAL, REWARD_NOT_PASSED


class DraughtsEnvironment(py_environment.PyEnvironment):

    def __init__(self, discount=1.0):
        """Initializes DraughtsEnvironment.

        Args:
          discount: Discount for reward.
        """
        super(DraughtsEnvironment, self).__init__()
        self._discount = np.asarray(discount, dtype=np.float32)

        self._states = None

    def action_spec(self):
        position_spec = BoundedArraySpec((), np.int32, minimum=0, maximum=4096)
        value_spec = BoundedArraySpec((), np.int32, minimum=1, maximum=2)
        return {
            'position': position_spec,
            'value': value_spec
        }

    def observation_spec(self):
        return BoundedArraySpec((8, 8), np.int32, minimum=0, maximum=4)

    def _reset(self):
        self._states = np.asarray([[2, 0, 2, 0, 2, 0, 2, 0],
                                   [0, 2, 0, 2, 0, 2, 0, 2],
                                   [2, 0, 2, 0, 2, 0, 2, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 1, 0, 1, 0, 1, 0],
                                   [0, 1, 0, 1, 0, 1, 0, 1],
                                   [1, 0, 1, 0, 1, 0, 1, 0]])
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32),
                        self._discount, self._states)

    def check_legal_tiles(self, state, position, move, player):
        if (state[position] == player) or (state[position] == player + 2):
            if state[move] == 0:
                return True
        return False

    def check_legal_common(self, state, position, move, player):
        if self.check_legal_tiles(state, position, move, player):
            print('legal move to make')
            y_dif = move[0] - position[0]
            x_dif = move[1] - position[1]
            if x_dif in [-1, 1]:
                print('correct x dist', x_dif, y_dif)
                if player == 1 and ((state[position] == player) and (y_dif == 1)):
                    print('correct y dist')
                    return True
                elif player == 2 and ((state[position] == player) and (y_dif == -1)):
                    print('correct y dist')
                    return True
                elif ((state[position] == player + 2) and y_dif in [-1, 1]) and (x_dif in [-1, 1]):
                    print('correct y dist')
                    return True
        return False

    def check_legal_take(self, state, position, move, player):
        if self.check_legal_tiles(state, position, move, player):
            y_dif = move[0] - position[0]
            x_dif = move[1] - position[1]
            if player == 1:
                if x_dif in [-2, 2]:
                    if ((state[position] == player) and (y_dif == 2)) or ((state[position] == player + 2) and
                                                                          (y_dif in [-2, 2])):
                        middle = [position[0] + (y_dif / 2), position[1] + (x_dif / 2)]
                        if state[middle] in [2, 4]:
                            return True
            else:
                if x_dif in [-2, 2]:
                    if ((state[position] == player) and (y_dif == -2)) or ((state[position] == player + 2) and
                                                                           (y_dif in [-2, 2])):
                        middle = [position[0] + (y_dif / 2), position[1] + (x_dif / 2)]
                        if state[middle] in [1, 3]:
                            return True
        return False

    def check_extra_takes(self, state, position, player):
        x_dif = [-2, 2]
        if state[position] != player:
            y_dif = [-2, 2]
        elif player == 1:
            y_dif = 2
        else:
            y_dif = -2
        for x in x_dif:
            for y in y_dif:
                if self.check_legal_take(state, position, [position[0] + y, position[1] + x], player):
                    return True
        return False

    def get_state(self) -> TimeStep:
        # Returning an unmodifiable copy of the state.
        return copy.deepcopy(self._current_time_step)

    def set_state(self, time_step: TimeStep):
        self._current_time_step = time_step
        self._states = time_step.observation

    def get_legal_moves(self, state, agent):
        mask = 0
        return mask

    def _step(self, action: np.ndarray):
        if self._current_time_step.is_last():
            return self._reset()

        extra = False

        index_flat = (np.array(range(4096)) == action['position']).reshape(1, 4096)
        index_flat = index_flat / np.sum(index_flat)
        for n in range(len(index_flat[0])):
            if np.isnan(index_flat[0][n]):
                index_flat[0][n] = 0
                index_flat = index_flat / np.sum(index_flat)
        index = np.random.choice(range(4096), p=np.squeeze(index_flat))

        position_index = index // 64
        move_index = index % 64

        position = (position_index // 8, position_index % 8)
        move = (move_index // 8, move_index % 8)

        if self.check_legal_common(self._states, position, move, action['value']):
            self.states[move] = self._states[position]
            self._states[position] = 0
            print('legal move')
            print(self._states)

            if self._turn == 1:
                self._turn = 2
            else:
                self._turn = 1

        elif self.check_legal_take(self._states, position, move, action['value']):
            y_dif = move[0] - position[0]
            x_dif = move[1] - position[1]
            middle = [position[0] + (y_dif / 2), position[1] + (x_dif / 2)]
            self._states[middle] = 0
            self.states[move] = self._states[position]
            self._states[position] = 0
            print(action['value'], ' takes ', middle)

            if not self.check_extra_takes(self._states, move, action['value']):
                extra = True
                if self._turn == 1:
                    self._turn = 2
                else:
                    self._turn = 1

        else:
            return TimeStep(StepType.LAST,
                            REWARD_ILLEGAL_MOVE,
                            self._discount,
                            self._states)

        is_final, reward = self._check_states(self._states, extra)

        if np.all(self._states == 0):
            step_type = StepType.FIRST
        elif is_final:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        if extra:
            return TimeStep(step_type, reward, np.asarray(0, dtype=np.float32), self._states)
        else:
            return TimeStep(step_type, reward, self._discount, self._states)

    def _check_states(self, states: np.ndarray, extra_take: bool):
        """Check if the given states are final and calculate reward.

        Args:
          states: states of the board.

        Returns:
          A tuple of (is_final, reward) where is_final means whether the states
          are final are not, and reward is the reward for stepping into the states
          The meaning of reward: 0 = not decided or draw, 1 = win, -1 = loss
        """
        if not 1 or 3 in states:
            return True, REWARD_LOSS  # 1 player loss
        elif not 2 or 4 in states:
            return True, REWARD_WIN  # 1 player win
        elif extra_take:
            return False, REWARD_NOT_PASSED  # ongoing with extra take for current player

        return False, REWARD_DRAW_OR_NOT_FINAL  # ongoing
