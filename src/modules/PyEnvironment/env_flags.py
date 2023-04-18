import numpy as np

REWARD_WIN = np.asarray(1., dtype=np.float32)
REWARD_LOSS = np.asarray(-1., dtype=np.float32)
REWARD_DRAW_OR_NOT_FINAL = np.asarray(0., dtype=np.float32)
REWARD_ILLEGAL_MOVE = np.asarray(-0.01, dtype=np.float32)
REWARD_NOT_PASSED = np.asarray(0.02, dtype=np.float32)

REWARD_WIN.setflags(write=False)
REWARD_LOSS.setflags(write=False)
REWARD_DRAW_OR_NOT_FINAL.setflags(write=False)
REWARD_NOT_PASSED.setflags(write=False)
