from gymnasium.envs.registration import register

register(
    id='draughts-v0',
    entry_point='draughts:DraughtsEnv',
)
