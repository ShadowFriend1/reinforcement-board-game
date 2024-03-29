import copy
from typing import Text, Optional

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.typing import types

from ..env_flags import REWARD_ILLEGAL_MOVE, REWARD_LOSS, REWARD_WIN, REWARD_DRAW_OR_NOT_FINAL


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

        self._killed_white = 0
        self._killed_black = 0

    # returns the specification of the environments action space
    def action_spec(self):
        position_spec = BoundedArraySpec((), np.int32, minimum=0, maximum=361)
        value_spec = BoundedArraySpec((), np.int32, minimum=1, maximum=2)
        return {
            'position': position_spec,
            'value': value_spec
        }

    # returns the specification of the environments observation space
    def observation_spec(self):
        state_spec = BoundedArraySpec((19, 19), np.int32, minimum=0, maximum=2)
        mask_spec = BoundedArraySpec((362,), np.int32, minimum=0, maximum=1)
        return {
            'state': state_spec,
            'mask': mask_spec
        }

    # resets the environment
    def _reset(self):
        self._states = np.zeros((19, 19), np.int32)
        self._mask = self.get_legal_moves(self._states, 1)
        observation = {'state': self._states, 'mask': self._mask}
        self._num_moves = 0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32),
                        self._discount, observation)

    # checks whether the placement position is empty
    def check_legal_tiles(self, state, position, player):
        if state[position] == 0:
            return True
        return False

    # Checks whether the piece can be placed in a location
    def check_legal_place(self, state, position, player):
        # Checks for legal placement (empty space)
        if self.check_legal_tiles(state, position, player):
            return True
        return False

    # Gets the environment state
    def get_state(self) -> TimeStep:
        # Returning an unmodifiable copy of the state.
        return copy.deepcopy(self._current_time_step)

    # Sets the environment state
    def set_state(self, time_step: TimeStep):
        self._current_time_step = time_step
        self._states = time_step.observation

    # returns a mask with all legal moves for the player
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

    # Updates the board, removing any take pieces
    def update_board(self, state, currently_played):
        next_state = np.copy(state)
        white_killed = 0
        black_killed = 0
        checked = np.zeros(state.shape)
        for y in range(len(next_state)):
            for x in range(len(next_state[0])):
                position = (y, x)
                if (next_state[position] != 0) and (checked[position] == 0):
                    # checks for adjacent empty space (saves processing time)
                    if self.get_adj_empty(next_state, position):
                        checked[position] = 1
                    # Checks for a connected piece with an adjacent empty space (long processing time)
                    else:
                        connected = self.get_connected(next_state, position)
                        if currently_played not in connected:
                            for pos in connected:
                                if self.get_adj_empty(next_state, pos):
                                    for n in connected:
                                        checked[n] = 1
                                    break
                            if checked[position] == 0:
                                for n in connected:
                                    if next_state[n] == 2:
                                        white_killed += 1
                                    else:
                                        black_killed += 1
                                    next_state[n] = 0
                elif next_state[position] == 0:
                    checked[position] = 1
        if self.get_adj_empty(next_state, currently_played):
            checked[currently_played] = 1
        else:
            connected = self.get_connected(next_state, currently_played)
            for pos in connected:
                if self.get_adj_empty(next_state, pos):
                    for n in connected:
                        checked[n] = 1
                    break
            if checked[currently_played] == 0:
                for n in connected:
                    if next_state[n] == 2:
                        white_killed += 1
                    else:
                        black_killed += 1
                    next_state[n] = 0
        killed_pieces = [white_killed, black_killed]
        return next_state, killed_pieces

    # Checks if piece has an adjacent empty spaces
    def get_adj_empty(self, state, position):
        x_dif = [1, -1]
        y_dif = [1, -1]
        for y_m in y_dif:
            for x_m in x_dif:
                check = (position[0] + y_m, position[1] + x_m)
                if (19 > check[0] > 0) and (19 > check[1] > 0):
                    if state[check] == 0:
                        return True
        return False

    # Gets all values that are adjacent to a position (used to determine whether pieces are surrounded)
    def get_adj_values(self, state, position):
        adjacent = []
        x_dif = [1, -1]
        y_dif = [1, -1]
        for y_m in y_dif:
            for x_m in x_dif:
                check = (position[0] + y_m, position[1] + x_m)
                if (19 > check[0] > 0) and (19 > check[1] > 0):
                    adjacent.append(state[check])
        return adjacent

    # Get adjacent positions with the same value
    def get_adjacent(self, state, position):
        adjacent = []
        piece = state[position]
        x_dif = [1, -1]
        y_dif = [1, -1]
        for y_m in y_dif:
            for x_m in x_dif:
                check = (position[0] + y_m, position[1] + x_m)
                if (19 > check[0] > 0) and (19 > check[1] > 0):
                    if state[check] == piece:
                        adjacent.append(check)
        return adjacent

    # Recursively checks for adjacent pieces
    def get_connected(self, state, position, connected=None):
        if connected is None:
            connected = [position]
        for pos in connected:
            for adj in self.get_adjacent(state, pos):
                if adj not in connected:
                    connected.append(adj)
                    for new in self.get_connected(state, adj, connected):
                        if new not in connected:
                            connected.append(new)
        return connected

    # Steps the environment forwards
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

        if index == 361:
            print('player ', action['value'], ' passed', self._passed)
            print(self._mask.sum())
            passed = True
        else:
            if self.check_legal_place(self._states, position, action['value']):
                self._states[position] = action['value']
                self._states, killed = self.update_board(self._states, position)
                self._killed_white += killed[0]
                self._killed_black += killed[1]
            else:
                illegal = True

        is_final, reward = self._check_states(passed)

        if passed:
            self._passed = True
        else:
            self._passed = False

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

    # Currently unimplemented so scoring is slightly wrong, but I couldn't find an effective way to achieve it
    def remove_dead(self, state):
        next_state = np.copy(state)
        killed_white = 0
        killed_black = 0
        killed = [killed_white, killed_black]
        return next_state, killed

    def get_score(self, state):
        # Gets the final score for the game and returns it
        end_state = np.copy(state)
        score_white = 0
        score_black = 0
        score_white += self._killed_white
        score_black += self._killed_black
        end_state, killed = self.remove_dead(end_state)
        score_white += killed[0]
        score_black += killed[1]
        checked = np.zeros(end_state.shape)
        for y in range(len(state)):
            for x in range(len(state[0])):
                position = (y, x)
                if (end_state[position] == 0) and (checked[position] == 0):
                    connected = self.get_connected(end_state, position)
                    surrounding = []
                    for n in connected:
                        checked[n] = 1
                        for m in self.get_adj_values(end_state, n):
                            if m not in surrounding:
                                surrounding.append(m)
                    if (2 in surrounding) and (1 not in surrounding):
                        score_white += len(connected)
                    elif (1 in surrounding) and (2 not in surrounding):
                        score_black += len(connected)
        return score_white, score_black

    def _check_states(self, player_pass):
        """Check if the given states are final and calculate reward.

        Args:
          player_pass: whether the current player has passed

        Returns:
          A tuple of (is_final, reward) where is_final means whether the states
          are final are not, and reward is the reward for stepping into the states
          The meaning of reward: 0 = not decided or draw, 1 = win, -1 = loss
        """

        if (self._passed and player_pass) or (self._mask.sum() == 0):
            self.console_print()
            score_white, score_black = self.get_score(self._states)
            if score_white > score_black:
                print('White Win', score_black, score_white)
                return True, REWARD_LOSS
            elif score_white < score_black:
                print('Black Win', score_black, score_white)
                return True, REWARD_WIN
            else:
                print('Draw', score_black, score_white)
                return True, REWARD_DRAW_OR_NOT_FINAL
        elif self._num_moves > self._max_moves:
            print('Time out')
            return True, REWARD_DRAW_OR_NOT_FINAL
        return False, REWARD_DRAW_OR_NOT_FINAL  # ongoing

    def console_print(self):
        table_str = '''
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
        {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
        '''.format(*tuple(self.get_state().observation['state'].flatten()))
        table_str = table_str.replace('0', ' ')
        table_str = table_str.replace('1', 'B')
        table_str = table_str.replace('2', 'W')
        print(table_str)

    def render(self, mode: Text = 'rgb_array') -> Optional[types.NestedArray]:
        return np.copy(self._states)
