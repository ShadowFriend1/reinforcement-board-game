# A class representing the GymChess game

import gym
import numpy as np

class ChessEnv(gym.Env):
    def __init__(self):
        self.board = np.array([])
