import os.path
from functools import partial
from itertools import cycle
import random

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import clear_output
from tf_agents.policies import policy_saver

from src.modules.PyEnvironment.draughts.draughts_environment import DraughtsEnvironment
from src.modules.PyEnvironment.env_flags import REWARD_WIN, REWARD_ILLEGAL_MOVE, REWARD_NOT_PASSED
from src.modules.PyEnvironment.tic_tac_toe.tic_tac_toe_environment import TicTacToeEnvironment
from src.modules.PyEnvironment.tic_tac_toe.tic_tac_toe_multi import TicTacToeMultiAgentEnv
from src.train.multi_agent import MultiDQNAgent

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.utils import common

from src.train.network import MaskedNetwork

sns.set()


def training_episode(tf_env, player_1, player_2):
    ts = tf_env.reset()
    player_1.reset()
    player_2.reset()
    time_steps = []
    players = cycle([player_1, player_2])
    reward = None
    player = None
    while not ts.is_last():
        if reward not in [REWARD_NOT_PASSED, REWARD_ILLEGAL_MOVE]:
            player = next(players)
        _, reward = player.act(collect=True)
        ts = tf_env.current_time_step()
        time_steps.append(ts)
    return time_steps


def collect_training_data(env, player_1, player_2):
    for game in range(episodes_per_iteration):
        training_episode(env, player_1, player_2)

        p1_return = player_1.episode_return()
        p2_return = player_2.episode_return()

        if REWARD_WIN in p1_return:
            outcome = 'p1_win'
        elif REWARD_WIN in p2_return:
            outcome = 'p2_win'
        else:
            outcome = 'time_out'

        games.append({
            'iteration': iteration,
            'game': game,
            'p1_return': np.sum(p1_return),
            'p2_return': np.sum(p2_return),
            'outcome': outcome,
            'final_step': tf_env.current_time_step()
        })


def train(player1, player2):
    for _ in range(train_steps_per_iteration):
        p1_train_info = player1.train_iteration()
        p2_train_info = player2.train_iteration()
        print('step complete')

        loss_infos.append({
            'iteration': iteration,
            'p1_loss': p1_train_info.loss.numpy(),
            'p2_loss': p2_train_info.loss.numpy()
        })


def action_fn(player, action):
    return {'position': action, 'value': player}


def observation_and_action_constraint_splitter(observation):
    return observation['state'], observation['mask']


def plot_history():
    games_data = pd.DataFrame.from_records(games)
    loss_data = pd.DataFrame.from_records(loss_infos)
    loss_data['Player 1'] = np.log(loss_data.p1_loss)
    loss_data['Player 2'] = np.log(loss_data.p2_loss)

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    loss_melted = pd.melt(loss_data,
                          id_vars=['iteration'],
                          value_vars=['Player 1', 'Player 2'])
    smoothing = iteration // 50
    loss_melted.iteration = smoothing * (loss_melted.iteration // smoothing)

    sns.lineplot(ax=axs[0][0],
                 x='iteration', hue='variable',
                 y='value', data=loss_melted)
    axs[0][0].set_title('Loss History')
    axs[0][0].set_ylabel('log-loss')

    returns_melted = pd.melt(games_data,
                             id_vars=['iteration'],
                             value_vars=['p1_return', 'p2_return'])
    returns_melted.iteration = smoothing * (returns_melted.iteration // smoothing)
    sns.lineplot(ax=axs[0][1],
                 x='iteration', hue='variable',
                 y='value', data=returns_melted)
    axs[0][1].set_title('Return History')
    axs[0][1].set_ylabel('return')

    games_data['p1_win'] = games_data.outcome == 'p1_win'
    games_data['p2_win'] = games_data.outcome == 'p2_win'
    games_data['time out'] = games_data.outcome == 'time_out'
    grouped_games_data = games_data.groupby('iteration')
    cols = ['game', 'p1_win', 'p2_win', 'time_out']
    grouped_games_data = grouped_games_data[cols]
    game_totals = grouped_games_data.max()['game'] + 1
    summed_games_data = grouped_games_data.sum()
    summed_games_data['p1_win_rate'] = summed_games_data.p1_win / game_totals
    summed_games_data['p2_win_rate'] = summed_games_data.p2_win / game_totals
    summed_games_data['time_out_rate'] = summed_games_data.time_out / game_totals
    summed_games_data['iteration'] = smoothing * (summed_games_data.index // smoothing)

    sns.lineplot(ax=axs[1][0],
                 x='iteration',
                 y='p1_win_rate',
                 data=summed_games_data,
                 label='Player 1 Win Rate')
    sns.lineplot(ax=axs[1][0],
                 x='iteration',
                 y='p2_win_rate',
                 data=summed_games_data,
                 label='Player 2 Win Rate')
    sns.lineplot(ax=axs[1][0],
                 x='iteration',
                 y='time_out_rate',
                 data=summed_games_data,
                 label='Time Out Ending Rate')
    axs[1][0].set_title('Outcomes History')
    axs[1][0].set_ylabel('ratio')

    plt.show()


def p2_reward_fn(ts: TimeStep) -> float:
    if ts.reward == -REWARD_WIN:
        return REWARD_WIN
    if ts.reward == REWARD_WIN:
        return REWARD_WIN
    return ts.reward


if __name__ == "__main__":
    num_iterations = 2000
    initial_collect_episodes = 10
    episodes_per_iteration = 10
    train_steps_per_iteration = 1
    training_batch_size = 512
    training_num_steps = 2
    replay_buffer_size = 5 * episodes_per_iteration * 4096
    learning_rate = 1e-3
    plot_interval = 1

    iteration = 1
    games = []
    loss_infos = []

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUS,", len(logical_gpus), "Logical GPUS")
        except RuntimeError as e:
            print(e)

    env = DraughtsEnvironment()

    tf_env = TFPyEnvironment(env)

    player_1_q_network = MaskedNetwork(
        action_spec=tf_env.action_spec()['position'],
        observation_spec=tf_env.observation_spec(),
        name='Player1QNet'
    )

    player_1 = MultiDQNAgent(
        tf_env,
        action_spec=tf_env.action_spec()['position'],
        action_fn=partial(action_fn, 1),
        name='Player1',
        learning_rate=learning_rate,
        training_batch_size=training_batch_size,
        training_num_steps=training_num_steps,
        replay_buffer_max_length=replay_buffer_size,
        td_errors_loss_fn=common.element_wise_squared_loss,
        q_network=player_1_q_network,
    )

    player_2_q_network = MaskedNetwork(
        action_spec=tf_env.action_spec()['position'],
        observation_spec=tf_env.observation_spec(),
        name='Player2QNet'
    )

    player_2 = MultiDQNAgent(
        tf_env,
        action_spec=tf_env.action_spec()['position'],
        action_fn=partial(action_fn, 2),
        reward_fn=p2_reward_fn,
        name='Player2',
        learning_rate=learning_rate,
        training_batch_size=training_batch_size,
        training_num_steps=training_num_steps,
        replay_buffer_max_length=replay_buffer_size,
        td_errors_loss_fn=common.element_wise_squared_loss,
        q_network=player_2_q_network,
    )

    player_1_policy_saver = policy_saver.PolicySaver(player_1.policy)

    player_2_policy_saver = policy_saver.PolicySaver(player_2.policy)

    save_dir = "..\models"

    player_1_policy_saver.save(os.path.join(save_dir, 'player_1_draughts'))

    player_2_policy_saver.save(os.path.join(save_dir, 'player_2_draughts'))

    print('Collecting Initial Training Sample...')
    for _ in range(initial_collect_episodes):
        training_episode(tf_env, player_1, player_2)
    print('Samples collected')

    if iteration > 1:
        plot_history()
        clear_output(wait=True)
    while iteration < num_iterations:
        collect_training_data(tf_env, player_1, player_2)
        train(player_1, player_2)
        print('iteration: ', iteration, ' completed')
        iteration += 1
        if iteration % plot_interval == 0:
            plot_history()
            clear_output(wait=True)
            player_1_policy_saver.save(os.path.join(save_dir, 'player_1_draughts'))
            player_2_policy_saver.save(os.path.join(save_dir, 'player_2_draughts'))

