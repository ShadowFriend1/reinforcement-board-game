import gym
import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.environments import suite_gym

if __name__ == "__main__":
    # Chess = GymChess:GymChess-v0, Draughts = GymDraughts:GymDraughts-v0
    env_name = input("Please input environment name in format '<folder_name>:<environment_name>'")
    env = suite_gym.load(env_name)