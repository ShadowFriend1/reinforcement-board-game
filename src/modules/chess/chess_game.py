# A class representing the chess game

import numpy as np

class ChessEnv(gym.Env):
    def __init__(self):
        self.board = np.array([])
