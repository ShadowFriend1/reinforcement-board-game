import gym
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

def train_model(env, agent):
    print()

def build_model(env, fc_layer_params=(100,), learning_rate=1e-3):
    q_net = q_network.QNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=fc_layer_params
    )

    train_step_counter = tf.Variable(0)

    optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimiser,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter
    )

    agent.initialize()

    return agent
