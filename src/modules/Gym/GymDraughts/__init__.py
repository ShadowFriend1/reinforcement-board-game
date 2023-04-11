from gymnasium.envs.registration import register

register(
    id='GymDraughts-v0',
    entry_point='GymDraughts:DraughtsEnv',
)
