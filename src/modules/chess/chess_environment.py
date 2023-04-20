import copy
from typing import Text, Optional

import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.typing import types

from ..env_flags import REWARD_ILLEGAL_MOVE, REWARD_LOSS, REWARD_WIN, REWARD_DRAW_OR_NOT_FINAL


class ChessEnvironment(py_environment.PyEnvironment):
    PAWN_W = 1
    KNIGHT_W = 2
    BISHOP_W = 3
    ROOK_W = 4
    QUEEN_W = 5
    KING_W = 6
    PIECES_W = [PAWN_W, BISHOP_W, KNIGHT_W, ROOK_W, QUEEN_W, KING_W]

    PAWN_B = 7
    KNIGHT_B = 8
    BISHOP_B = 9
    ROOK_B = 10
    QUEEN_B = 11
    KING_B = 12
    PIECES_B = [PAWN_B, BISHOP_B, KNIGHT_B, ROOK_B, QUEEN_B, KING_B]

    REWARD_CHECK = np.asarray(0.1, dtype=np.float32)
    REWARD_CHECK.setflags(write=False)

    def __init__(self, discount=1.0):
        """Initializes DraughtsEnvironment.

        Args:
          discount: Discount for reward.
        """
        super(ChessEnvironment, self).__init__()
        self._discount = np.asarray(discount, dtype=np.float32)

        self._states = None

        self._past_states = []

        self._mask = None

        self._num_moves = 0
        self._max_moves = 400

        # Stores the column of a pawn that double moves to allow for en passant
        self._double_move = 9

        # Player currently in check (3 means none)
        self._check = 3

        # Stores whether pieces have moved (to check for castling)
        self._moved = []

    def action_spec(self):
        position_spec = BoundedArraySpec((), np.int32, minimum=0, maximum=4097)
        value_spec = BoundedArraySpec((), np.int32, minimum=1, maximum=2)
        return {
            'position': position_spec,
            'value': value_spec
        }

    def observation_spec(self):
        state_spec = BoundedArraySpec((8, 8), np.int32, minimum=0, maximum=12)
        mask_spec = BoundedArraySpec((4098,), np.int32, minimum=0, maximum=1)
        return {
            'state': state_spec,
            'mask': mask_spec
        }

    def _reset(self):
        self._states = np.asarray([[self.ROOK_B, self.KNIGHT_B, self.BISHOP_B, self.QUEEN_B, self.KING_B,
                                    self.BISHOP_B, self.KNIGHT_B, self.ROOK_B],
                                   [self.PAWN_B, self.PAWN_B, self.PAWN_B, self.PAWN_B, self.PAWN_B, self.PAWN_B,
                                    self.PAWN_B, self.PAWN_B],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [self.PAWN_W, self.PAWN_W, self.PAWN_W, self.PAWN_W, self.PAWN_W, self.PAWN_W,
                                    self.PAWN_W, self.PAWN_W],
                                   [self.ROOK_W, self.KNIGHT_W, self.BISHOP_W, self.QUEEN_W, self.KING_W, self.BISHOP_W,
                                    self.KNIGHT_W, self.ROOK_W]])
        self._mask = self.get_legal_moves(self._states, 1)
        self._past_states = []
        self._past_states.append(self._states)
        observation = {'state': self._states, 'mask': self._mask}
        self._num_moves = 0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32),
                        self._discount, observation)

    def check_legal_tiles(self, state, position, move, player):
        if ((player == 1) and (state[position] in self.PIECES_W)) and (state[move] not in self.PIECES_W):
            return True
        elif ((player == 2) and (state[position] in self.PIECES_B)) and (state[move] not in self.PIECES_B):
            return True
        else:
            return False

    def castle(self, state, left_right, player):
        next_state = np.copy(state)
        if player == 1:
            row = 7
        else:
            row = 0
        if left_right == 0:
            state[(row, 2)] = state[(row, 4)]
            state[(row, 4)] = 0
            state[(row, 3)] = state[(row, 0)]
            state[(row, 0)] = 0
        else:
            state[(row, 6)] = state[(row, 4)]
            state[(row, 4)] = 0
            state[(row, 5)] = state[(row, 7)]
            state[(row, 7)] = 0
        return next_state

    def move(self, state, position, move, player):
        next_state = np.copy(state)
        if ((next_state[position] in [self.PAWN_W, self.PAWN_B]) and (move[1] == self._double_move)) and \
                (((player == 1) and (next_state[(position[0], self._double_move)] == self.PAWN_B))
                 or ((player == 2) and (next_state[(position[0], self._double_move)] == self.PAWN_W))):
            next_state[move] = next_state[position]
            next_state[position] = 0
            next_state[(position[0], self._double_move)] = 0
        else:
            next_state[move] = next_state[position]
            next_state[position] = 0
        return next_state

    def check_legal_castle(self, state, left_right, player):
        if self._check == player:
            return False
        if (player == 1) and ((7, 4) not in self._moved):
            row = 7
        elif (player == 2) and ((0, 4) not in self._moved):
            row = 0
        else:
            return False
        if (left_right == 0) and ((row, 0) not in self._moved):
            req_empty = [state[(row, 1)], state[(row, 2)], state[(row, 3)]]
        elif (left_right == 1) and ((row, 7) not in self._moved):
            req_empty = [state[(row, 5)], state[(row, 6)]]
        else:
            return False
        if all(n == 0 for n in req_empty):
            next_state = self.castle(state, left_right, player)
            if not self.check_check(next_state, player):
                return True
        return False

    def check_legal_move(self, state, position, move, player):
        # Checks to see if the piece being moved belongs to the player and isnt moving onto one of their pieces
        if self.check_legal_tiles(state, position, move, player):
            # Checks if the move is legal for the type of piece
            match state[position]:
                case self.PAWN_W:
                    if state[move] == 0:
                        if position[1] == move[1]:
                            if move[0] == (position[0] - 1):
                                return True
                            if ((move[0] == (position[0] - 2)) and (position[0] == 6)) and \
                                    (state[(move[0]+1, move[1])] == 0):
                                return True
                        elif (position[1] - self._double_move in [1, -1]) and (move[1] == self._double_move):
                            if (move[0] == (position[0] + 1)) and \
                                    (state[(position[0], self._double_move)] == self.PAWN_B):
                                return True
                    else:
                        if (move[1] - position[1]) in [1, -1]:
                            if move[0] == (position[0] - 1):
                                return True

                case self.KNIGHT_W | self.KNIGHT_B:
                    x_dif = abs(move[1] - position[1])
                    y_dif = abs(move[0] - position[0])
                    if (x_dif == 1) and (y_dif == 2):
                        return True
                    elif (x_dif == 2) and (y_dif == 1):
                        return True

                case self.BISHOP_W | self.BISHOP_B:
                    x_dif = move[1] - position[1]
                    y_dif = move[0] - position[0]
                    if abs(x_dif) == abs(y_dif):
                        if x_dif < 0:
                            step_val_x = -1
                        else:
                            step_val_x = 1
                        if y_dif < 0:
                            step_val_y = -1
                        else:
                            step_val_y = 1
                        for y in range(0, y_dif, step_val_y):
                            for x in range(0, x_dif, step_val_x):
                                if abs(y) == abs(x):
                                    if (state[(position[0] + y, position[1] + x)] != 0) and (
                                            ((position[0] + y, position[0] + x) != position) and
                                            ((position[0] + y, position[1] + x) != move)):
                                        return False
                        return True

                case self.ROOK_W | self.ROOK_B:
                    if move[0] == position[0]:
                        x_dif = move[1] - position[1]
                        if x_dif < 0:
                            step_val = -1
                        else:
                            step_val = 1
                        row = move[0]
                        for x in range(0, x_dif, step_val):
                            if (state[(row, position[1] + x)] != 0) and (((row, position[1] + x) != position) and
                                                                         ((row, position[1] + x) != move)):
                                return False
                        return True
                    if move[1] == position[1]:
                        y_dif = move[0] - position[0]
                        if y_dif < 0:
                            step_val = -1
                        else:
                            step_val = 1
                        col = move[1]
                        for y in range(0, y_dif, step_val):
                            if (state[(position[0] + y, col)] != 0) and (((position[0] + y, col) != position) and
                                                                         ((position[0] + y, col) != move)):
                                return False
                        return True

                case self.QUEEN_W | self.QUEEN_B:
                    x_dif = move[1] - position[1]
                    y_dif = move[0] - position[0]
                    if abs(x_dif) == abs(y_dif):
                        if x_dif < 0:
                            step_val_x = -1
                        else:
                            step_val_x = 1
                        if y_dif < 0:
                            step_val_y = -1
                        else:
                            step_val_y = 1
                        for y in range(0, y_dif, step_val_y):
                            for x in range(0, x_dif, step_val_x):
                                if abs(y) == abs(x):
                                    if (state[(position[0] + y, position[1] + x)] != 0) and (
                                            ((position[0] + y, position[0] + x) != position) and
                                            ((position[0] + y, position[1] + x) != move)):
                                        return False
                        return True
                    elif move[0] == position[0]:
                        if x_dif < 0:
                            step_val = -1
                        else:
                            step_val = 1
                        row = move[0]
                        for x in range(0, x_dif, step_val):
                            if (state[(row, position[1] + x)] != 0) and (((row, x) != position) and ((row, x) != move)):
                                return False
                        return True
                    elif move[1] == position[1]:
                        if y_dif < 0:
                            step_val = -1
                        else:
                            step_val = 1
                        col = move[1]
                        for y in range(0, y_dif, step_val):
                            if (state[(position[0] + y, col)] != 0) and (((y, col) != position) and ((y, col) != move)):
                                return False
                        return True

                case self.KING_W | self.KING_B:
                    x_dif = abs(move[1] - position[1])
                    y_dif = abs(move[0] - position[0])
                    if (x_dif <= 1) and (y_dif <= 1):
                        return True

                case self.PAWN_B:
                    if state[move] == 0:
                        if position[1] == move[1]:
                            if move[0] == (position[0] + 1):
                                return True
                            if ((move[0] == (position[0] + 2)) and (position[0] == 1)) and \
                                    (state[(move[0]-1, move[1])] == 0):
                                return True
                        elif (position[1] - self._double_move in [1, -1]) and (move[1] == self._double_move):
                            if (move[0] == (position[0] + 1)) and \
                                    (state[(position[0], self._double_move)] == self.PAWN_W):
                                return True
                    else:
                        if (move[1] - position[1]) in [1, -1]:
                            if move[0] == (position[0] + 1):
                                return True

                case _:
                    return False
        return False

    def check_check(self, state, player):
        if player == 1:
            other_player = 2
            king = self.KING_W
        else:
            other_player = 1
            king = self.KING_B
        other_moves = self.get_legal_moves(state, other_player, False)
        for n in range(len(other_moves)):
            if other_moves[n] == 1:
                move_index = n % 64
                move = (move_index // 8, move_index % 8)
                if state[move] == king:
                    return True
        return False

    def check_legal(self, state, position, move, player, check_check=True):
        if 8 in [position[0], position[1], move[0], move[1]]:
            return False
        if self.check_legal_move(state, position, move, player):
            if not check_check:
                return True
            next_state = self.move(state, position, move, player)
            if not self.check_check(next_state, player):
                return True
        return False

    def get_state(self) -> TimeStep:
        # Returning an unmodifiable copy of the state.
        return copy.deepcopy(self._current_time_step)

    def set_state(self, time_step: TimeStep):
        self._current_time_step = time_step
        self._states = time_step.observation

    def set_position(self, position, value):
        self._states[position] = value

    def get_legal_moves(self, state, player, check_check=True):
        legal_flat = np.zeros((4098,), np.int32)
        if self.check_legal_castle(state, 0, player):
            legal_flat[4096] = 1
        if self.check_legal_castle(state, 1, player):
            legal_flat[4097] = 1
        # Loop through each position on the board checking for legal normal moves
        for y in range(len(state)):
            for x in range(len(state[0])):
                # Get legal moves for position
                # For normal pieces
                position = (y, x)
                for y_m in range(len(state)):
                    for x_m in range(len(state[0])):
                        move = (y_m, x_m)
                        if self.check_legal(state, position, move, player, check_check):
                            position_flat = (position[0] * 8) + position[1]
                            move_flat = (move[0] * 8) + move[1]
                            legal_flat[((position_flat * 64) + move_flat)] = 1
        return legal_flat

    def _step(self, action: np.ndarray):
        if self._current_time_step.is_last():
            return self._reset()

        next_player = action['value']
        illegal = False
        castle = None
        check = False
        self._num_moves += 1

        index_flat = (np.array(range(4098)) == action['position']).reshape(1, 4098)
        index_flat = index_flat / index_flat.sum()
        if np.isnan(index_flat).any():
            observation = {'state': self._states, 'mask': self._mask}
            return TimeStep(StepType.LAST,
                            REWARD_ILLEGAL_MOVE,
                            self._discount,
                            observation)
        index = np.random.choice(range(4098), p=np.squeeze(index_flat))

        position_index = index // 64
        move_index = index % 64

        position = (position_index // 8, position_index % 8)
        move = (move_index // 8, move_index % 8)

        if position_index == 64:
            castle = move_index

        if (castle is not None) and self.check_legal_castle(self._states, castle, action['value']):
            self._states = self.castle(self._states, castle, action['value'])
        elif castle is None:
            if self.check_legal_move(self._states, position, move, action['value']):
                next_state = self.move(self._states, position, move, action['value'])
                if (self._states[position] in [self.PAWN_W, self.PAWN_B]) and (abs(move[0] - position[0]) == 2):
                    self._double_move = move[1]
                else:
                    self._double_move = 9
                if position not in self._moved:
                    self._moved.append(position)
                self._states = next_state
        else:
            illegal = True

        if (action['value'] == 1) and (move[0] == 0):
            self._states[move] = self.QUEEN_W
        elif (action['value'] == 2) and (move[0] == 7):
            self._states[move] = self.QUEEN_B

        is_final, reward = self._check_states(action['value'])

        if np.all(self._states == 0):
            step_type = StepType.FIRST
        elif is_final:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        if not illegal:
            self._past_states.append(self._states)
            if action['value'] == 1:
                next_player = 2
            else:
                next_player = 1
            if self.check_check(self._states, next_player):
                check = True
                self._check = next_player

        self._mask = self.get_legal_moves(self._states, next_player)

        observation = {'state': self._states, 'mask': self._mask}
        if illegal and (not is_final):
            return TimeStep(step_type, REWARD_ILLEGAL_MOVE, self._discount, observation)
        elif check and (not is_final):
            return TimeStep(step_type, self.REWARD_CHECK, self._discount, observation)
        else:
            return TimeStep(step_type, reward, self._discount, observation)

    def _check_states(self, player: int):
        """Check if the given states are final and calculate reward.

        Args:
          player: the current player

        Returns:
          A tuple of (is_final, reward) where is_final means whether the states
          are final are not, and reward is the reward for stepping into the states
          The meaning of reward: 0 = not decided or draw, 1 = win, -1 = loss
        """
        if self._mask.sum() == 0:
            self.console_print()
            if player == 1:
                if self._check == player:
                    print('Black Wins')
                    return True, REWARD_LOSS
                else:
                    print('Stalemate')
                    return True, REWARD_DRAW_OR_NOT_FINAL
            else:
                if self._check == player:
                    print('White Wins')
                    return True, REWARD_WIN
                else:
                    print('Draw by Stalemate')
                    return True, REWARD_DRAW_OR_NOT_FINAL
        elif self._num_moves > self._max_moves:
            print('Time Out')
            return True, REWARD_DRAW_OR_NOT_FINAL
        count = 0
        states = []
        for n in self._past_states:
            if np.array_equal(n, self._states):
                states.append(n)
                count += 1
        if count >= 3:
            print('Draw by Repetition')
            return True, REWARD_DRAW_OR_NOT_FINAL
        else:
            return False, REWARD_DRAW_OR_NOT_FINAL  # ongoing

    def console_print(self):
        table_str = '''
        {} | {} | {} | {} | {} | {} | {} | {}
        -- + -- + -- + -- + -- + -- + -- + --
        {} | {} | {} | {} | {} | {} | {} | {}
        -- + -- + -- + -- + -- + -- + -- + --
        {} | {} | {} | {} | {} | {} | {} | {}
        -- + -- + -- + -- + -- + -- + -- + --
        {} | {} | {} | {} | {} | {} | {} | {}
        -- + -- + -- + -- + -- + -- + -- + --
        {} | {} | {} | {} | {} | {} | {} | {}
        -- + -- + -- + -- + -- + -- + -- + --
        {} | {} | {} | {} | {} | {} | {} | {}
        -- + -- + -- + -- + -- + -- + -- + --
        {} | {} | {} | {} | {} | {} | {} | {}
        -- + -- + -- + -- + -- + -- + -- + --
        {} | {} | {} | {} | {} | {} | {} | {}
        '''.format(*tuple(self._states.flatten()))
        table_str = table_str.replace('10', 'BR')
        table_str = table_str.replace('11', 'BQ')
        table_str = table_str.replace('12', 'BK')
        table_str = table_str.replace('0', '  ')
        table_str = table_str.replace('1', 'WP')
        table_str = table_str.replace('2', 'WN')
        table_str = table_str.replace('3', 'WB')
        table_str = table_str.replace('4', 'WR')
        table_str = table_str.replace('5', 'WQ')
        table_str = table_str.replace('6', 'WK')
        table_str = table_str.replace('7', 'BP')
        table_str = table_str.replace('8', 'BN')
        table_str = table_str.replace('9', 'BB')
        print(table_str)

    def render(self, mode: Text = 'rgb_array') -> Optional[types.NestedArray]:
        return np.copy(self._states)
