import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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
from src.modules import *

num_iterations = 20000  # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_test_episodes = 10  # @param {type:"integer"}
test_interval = 1000  # @param {type:"integer"}


def train_model(train_env, test_env, agent, replay_buffer, iterator):
    # Wraps the code in a graph using TF function
    agent.train = common.function(agent.train)

    # Resets the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(test_env, agent.policy, num_test_episodes)
    returns = [avg_return]

    for n in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % test_interval == 0:
            avg_return = compute_avg_return(test_env, agent.policy, num_test_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

    iterations = range(0, num_iterations + 1, test_interval)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=250)
    plt.show()


def build_model(env, fc_layer_params=(100,), learning_rate=1e-3):
    q_net = q_network.QNetwork(env.observation_spec(),
                               env.action_spec(),
                               fc_layer_params=fc_layer_params)

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

    print("-- Model Build Completed")
    return agent


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)

    # Chess = chess:chess-v0, draughts = draughts:draughts-v0, go = gym-go:go-v0
    # env_name = input("Please input environment name in format '<install_name>:<environment_name>'")
    env_name = 'gym_go:go-v0'
    train_py_env = suite_gym.load(env_name, gym_kwargs={'size': 19, 'komi': 0, 'reward_method': 'heuristic'})
    test_py_env = suite_gym.load(env_name, gym_kwargs={'size': 19, 'komi': 0, 'reward_method': 'heuristic'})
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    test_env = tf_py_environment.TFPyEnvironment(test_py_env)
    agent = build_model(train_env)
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                                   batch_size=train_env.batch_size,
                                                                   max_length=replay_buffer_max_length)

    collect_data(train_env, agent.policy, replay_buffer, steps=100)

    dataset = replay_buffer.as_dataset(num_parallel_calls=3,
                                       sample_batch_size=batch_size,
                                       num_steps=2).prefetch(3)

    iterator = iter(dataset)

    train_model(train_env, test_env, agent, replay_buffer, iterator)
