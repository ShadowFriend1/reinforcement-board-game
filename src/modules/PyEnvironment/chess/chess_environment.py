import copy
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep

from ..env_flags import REWARD_ILLEGAL_MOVE, REWARD_LOSS, REWARD_WIN, REWARD_DRAW_OR_NOT_FINAL, REWARD_NOT_PASSED


class ChessEnvironment(py_environment.PyEnvironment):

    PAWN_W = 1
    BISHOP_W = 2
    KNIGHT_W = 3
    ROOK_W = 4
    QUEEN_W = 5
    KING_W = 6
    PIECES_W = [PAWN_W, BISHOP_W, KNIGHT_W, ROOK_W, QUEEN_W, KING_W]

    PAWN_B = 7
    BISHOP_B = 8
    KNIGHT_B = 9
    ROOK_B = 10
    QUEEN_B = 11
    KING_B = 12
    PIECES_B = [PAWN_B, BISHOP_B, KNIGHT_B, ROOK_B, QUEEN_B, KING_B]

    def __init__(self, discount=1.0):
        """Initializes DraughtsEnvironment.

        Args:
          discount: Discount for reward.
        """
        super(ChessEnvironment, self).__init__()
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
        state_spec = BoundedArraySpec((8, 8), np.int32, minimum=0, maximum=12)
        mask_spec = BoundedArraySpec((4096,), np.int32, minimum=0, maximum=1)
        return {
            'state': state_spec,
            'mask': mask_spec
        }

    def _reset(self):
        self._states = np.asarray([[9, 8, 7, 11, 12, 7, 8, 9],
                                   [6, 6, 6, 6, 6, 6, 6, 6],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 1, 1, 1, 1, 1, 1, 1],
                                   [4, 3, 2, 5, 6, 2, 3, 4]])
        self._mask = self.get_legal_moves(self._states, 1)
        observation = {'state': self._states, 'mask': self._mask}
        self._num_moves = 0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32),
                        self._discount, observation)

    def check_legal_tiles(self, state, position, player):
        if (player == 1) and (state[position] in self.PIECES_W):
            return True
        elif (player == 2) and (state[position] in self.PIECES_B):
            return True
        else:
            return False

    def check_legal_move(self, state, position, move, player):
        if self.check_legal_tiles(state, position, player):
            match state[position]:
                case 1:
                    return True
                case 2:
                    return True
                case 3:
                    return True
                case 4:
                    return True
                case 5:
                    return True
                case 6:
                    return True
                case 7:
                    return True
                case 8:
                    return True
                case 9:
                    return True
                case 10:
                    return True
                case 11:
                    return True
                case 12:
                    return True
                case _:
                    return False
        return False

    def get_state(self) -> TimeStep:
        # Returning an unmodifiable copy of the state.
        return copy.deepcopy(self._current_time_step)

    def set_state(self, time_step: TimeStep):
        self._current_time_step = time_step
        self._states = time_step.observation

    def get_legal_moves(self, state, player):
        legal_flat = np.zeros((4096,), np.int32)
        # Loop through each position on the board checking for legal normal moves
        for y in range(len(state)):
            for x in range(len(state[0])):
                # Get legal moves for position
                # For normal pieces
                position = (y, x)
                for y_m in range(len(state)):
                    for x_m in range(len(state[0])):
                        move = (y_m, x_m)
                        if self.check_legal_move(state, position, move, player):
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
                self._states[move] = self._states[position]
                self._states[position] = 0

                if self.check_extra_takes(self._states, move, action['value']):
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
