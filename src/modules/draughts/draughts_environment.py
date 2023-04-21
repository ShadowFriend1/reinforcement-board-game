import copy
from typing import Optional, Text

import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.typing import types

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

        self._mask = None

        self._num_moves = 0
        self._max_moves = 200

    def action_spec(self):
        position_spec = BoundedArraySpec((), np.int32, minimum=0, maximum=4095)
        value_spec = BoundedArraySpec((), np.int32, minimum=1, maximum=2)
        return {
            'position': position_spec,
            'value': value_spec
        }

    def observation_spec(self):
        state_spec = BoundedArraySpec((8, 8), np.int32, minimum=0, maximum=4)
        mask_spec = BoundedArraySpec((4096,), np.int32, minimum=0, maximum=1)
        return {
            'state': state_spec,
            'mask': mask_spec
        }

    def _reset(self):
        self._states = np.asarray([[0, 2, 0, 2, 0, 2, 0, 2],
                                   [2, 0, 2, 0, 2, 0, 2, 0],
                                   [0, 2, 0, 2, 0, 2, 0, 2],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 1, 0, 1, 0, 1, 0],
                                   [0, 1, 0, 1, 0, 1, 0, 1],
                                   [1, 0, 1, 0, 1, 0, 1, 0]])
        self._mask = self.get_legal_moves(self._states, 1)
        observation = {'state': self._states, 'mask': self._mask}
        self._num_moves = 0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32),
                        self._discount, observation)

    def check_legal_tiles(self, state, position, move, player):
        if (state[position] == player) or (state[position] == player + 2):
            if state[move] == 0:
                return True
        return False

    def check_legal_common(self, state, position, move, player):
        if self.check_legal_tiles(state, position, move, player):
            y_dif = move[0] - position[0]
            x_dif = move[1] - position[1]
            if x_dif in [-1, 1]:
                if player == 1 and ((state[position] == player) and (y_dif == -1)):
                    return True
                elif player == 2 and ((state[position] == player) and (y_dif == 1)):
                    return True
                elif ((state[position] == player + 2) and y_dif in [-1, 1]) and (x_dif in [-1, 1]):
                    return True
        return False

    def check_legal_take(self, state, position, move, player):
        if self.check_legal_tiles(state, position, move, player):
            y_dif = move[0] - position[0]
            x_dif = move[1] - position[1]
            if player == 1:
                if x_dif in [-2, 2]:
                    if ((state[position] == player) and (y_dif == -2)) or ((state[position] == player + 2) and
                                                                           (y_dif in [-2, 2])):
                        middle = (int(position[0] + (y_dif / 2)), int(position[1] + (x_dif / 2)))
                        if state[middle] in [2, 4]:
                            return True, middle
            else:
                if x_dif in [-2, 2]:
                    if ((state[position] == player) and (y_dif == 2)) or ((state[position] == player + 2) and
                                                                          (y_dif in [-2, 2])):
                        middle = (int(position[0] + (y_dif / 2)), int(position[1] + (x_dif / 2)))
                        if state[middle] in [1, 3]:
                            return True, middle
        return False, None

    def check_extra_takes(self, state, position, player):
        x_dif = [-2, 2]
        if state[position] != player:
            y_dif = [-2, 2]
        elif player == 1:
            y_dif = [-2]
        else:
            y_dif = [2]
        for x in x_dif:
            for y in y_dif:
                move = (position[0] + y, position[1] + x)
                if (0 <= move[0] < 8) and (0 <= move[1] < 8):
                    legal, _ = self.check_legal_take(state, position, move, player)
                    if legal:
                        return True, None
        return False, None

    def get_state(self) -> TimeStep:
        # Returning an unmodifiable copy of the state.
        return copy.deepcopy(self._current_time_step)

    def set_state(self, time_step: TimeStep):
        self._current_time_step = time_step
        self._states = time_step.observation

    def get_legal_moves(self, state, player):
        legal_flat = np.zeros((4096,), np.int32)
        # Loop through each position on the board checking for legal normal moves
        y_dif = [-1, 1]
        x_dif = [-1, 1]
        y_dif_take = [-2, 2]
        x_dif_take = [-2, 2]
        for y in range(len(state)):
            for x in range(len(state[0])):
                # Get legal moves for position
                # For normal pieces
                position = (y, x)
                for y_m in y_dif:
                    for x_m in x_dif:
                        move = (position[0] + y_m, position[1] + x_m)
                        if (0 <= move[0] < 8) and (0 <= move[1] < 8):
                            if self.check_legal_common(state, position, move, player):
                                position_flat = (position[0] * 8) + position[1]
                                move_flat = (move[0] * 8) + move[1]
                                legal_flat[(position_flat * 64) + move_flat] = 1
        # Checks for legal takes, if there is a legal take forces a take by removing all no take legal moves
        legal_take = False
        for y in range(len(state)):
            for x in range(len(state[0])):
                position = (y, x)
                for y_m in y_dif_take:
                    for x_m in x_dif_take:
                        move = (position[0] + y_m, position[1] + x_m)
                        if (0 <= move[0] < 8) and (0 <= move[1] < 8):
                            legal, _ = self.check_legal_take(state, position, move, player)
                            if legal:
                                if not legal_take:
                                    legal_flat = np.zeros((4096,), np.int32)
                                    legal_take = True
                                position_flat = (position[0] * 8) + position[1]
                                move_flat = (move[0] * 8) + move[1]
                                legal_flat[(position_flat * 64) + move_flat] = 1
        return legal_flat

    def _step(self, action: np.ndarray):
        if self._current_time_step.is_last():
            return self._reset()

        self._num_moves += 1

        extra = False
        next_player = action['value']
        illegal = False

        index_flat = (np.array(range(4096)) == action['position']).reshape(1, 4096)
        index_flat = index_flat / index_flat.sum()
        if np.isnan(index_flat).any():
            observation = {'state': self._states, 'mask': self._mask}
            return TimeStep(StepType.LAST,
                            REWARD_ILLEGAL_MOVE,
                            self._discount,
                            observation)
        index = np.random.choice(range(4096), p=np.squeeze(index_flat))

        position_index = index // 64
        move_index = index % 64

        position = (position_index // 8, position_index % 8)
        move = (move_index // 8, move_index % 8)

        if self.check_legal_common(self._states, position, move, action['value']):
            self._states[move] = self._states[position]
            self._states[position] = 0

        else:
            take_legal, middle = self.check_legal_take(self._states, position, move, action['value'])
            if take_legal:
                self._states[middle] = 0
                self._states[move] = self._states[position]
                self._states[position] = 0
                extra_legal, _ = self.check_extra_takes(self._states, move, action['value'])
                if extra_legal:
                    extra = True
            else:
                illegal = True

        if ((action['value'] == 1) and (move[0] == 0)) or ((action['value'] == 2) and (move[0] == 7)):
            self._states[move] = action['value'] + 2

        is_final, reward = self._check_states(self._states, extra, action['value'])

        if np.all(self._states == 0):
            step_type = StepType.FIRST
        elif is_final:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        if (not extra) and (not illegal):
            if action['value'] == 1:
                next_player = 2
            else:
                next_player = 1

        self._mask = self.get_legal_moves(self._states, next_player)

        observation = {'state': self._states, 'mask': self._mask}
        if extra:
            return TimeStep(step_type, reward, self._discount, observation)
        elif illegal and (not is_final):
            return TimeStep(step_type, REWARD_ILLEGAL_MOVE, self._discount, observation)
        else:
            return TimeStep(step_type, reward, self._discount, observation)

    def _check_states(self, states: np.ndarray, extra_take: bool, player: int):
        """Check if the given states are final and calculate reward.

        Args:
          states: states of the board.

        Returns:
          A tuple of (is_final, reward) where is_final means whether the states
          are final are not, and reward is the reward for stepping into the states
          The meaning of reward: 0 = not decided or draw, 1 = win, -1 = loss
        """
        if not any(x in states for x in [1, 3]):
            return True, REWARD_LOSS  # 1 player loss
        elif not any(x in states for x in [2, 4]):
            return True, REWARD_WIN  # 1 player win
        elif self._mask.sum() == 0:
            if player == 1:
                return True, REWARD_LOSS
            else:
                return True, REWARD_WIN
        elif self._num_moves > self._max_moves:
            return True, REWARD_DRAW_OR_NOT_FINAL
        elif extra_take:
            return False, REWARD_NOT_PASSED  # ongoing with extra take for current player
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
        print(table_str)

    def render(self, mode: Text = 'rgb_array') -> Optional[types.NestedArray]:
        return np.copy(self._states)
