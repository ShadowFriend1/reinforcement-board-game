from gymnasium.envs.registration import register

register(
    id='chess-v0',
    entry_point='chess:ChessEnv',
)
