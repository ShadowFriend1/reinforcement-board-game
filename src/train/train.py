import gym
import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.environments import suite_gym


def train_model(env, net):
    print()

def build_model(env):
    q_net = q_network.QNetwork(

    )


if __name__ == "__main__":
    # Chess = chess:chess-v0, draughts = draughts:draughts-v0
    env_name = input("Please input environment name in format '<folder_name>:<environment_name>'")
    env = suite_gym.load(env_name)
    print()
