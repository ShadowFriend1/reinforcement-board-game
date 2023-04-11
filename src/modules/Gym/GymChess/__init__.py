from gymnasium.envs.registration import register

register(
    id='GymChess-v0',
    entry_point='GymChess:ChessEnv',
)
