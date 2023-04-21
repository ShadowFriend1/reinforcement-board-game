import os
import string
from functools import partial
from itertools import cycle

import PySimpleGUI as sg
import numpy as np
import tensorflow as tf
from tf_agents.environments import TFPyEnvironment

from src.modules.draughts.draughtsGUI_pygame import DraughtsGUI_pygame
from src.modules.env_flags import REWARD_NOT_PASSED, REWARD_ILLEGAL_MOVE
from src.modules.go.go_environment import GoEnvironment
from src.play.human_agent import HumanAgent
from src.train.multi_agent import MultiDQNAgent
from src.train.network import MaskedNetwork
from src.train.train_py_env import action_fn


def play():
    layout = [[sg.Text('Choose whether to play 1st or 2nd')],
              [sg.Radio('Player 1', 'RADIO1', default=True, key="-PLAYER1-")],
              [sg.Radio('Player 2', 'RADIO1', default=False)],
              [sg.Text('Choose a model to play against: ')],
              [sg.Text('Import an Environment: '), sg.Input(),
               sg.FolderBrowse(initial_folder=os.path.join('..', 'models', 'draughts', 'saved'), key='-IN-')],
              [sg.Text('Note, playing against a player 1 AI as player 1 or vice versa may cause issues')],
              [sg.Button('Ok'), sg.Button('Back')]]

    # Create the window
    window = sg.Window('Draughts against AI', layout)

    # Display and interact with the Window using an Event Loop
    while True:
        event, values = window.read()
        # See if user wants to quit or window was closed
        if event == 'Ok':
            try:
                # Load indicated policy and set it as the policy for an agent
                saved_policy = tf.saved_model.load(values['-IN-'])
                window.close()
                break
            except IOError:
                print('Invalid File')
        if event == sg.WINDOW_CLOSED or event == 'Back':
            window.close()
            return True

    env = GoEnvironment()

    tf_env = TFPyEnvironment(env)

    if values['-PLAYER1-']:
        ai_player = 2
        human_player = 1
    else:
        ai_player = 1
        human_player = 2

    temp_q_net = MaskedNetwork(
        action_spec=tf_env.action_spec()['position'],
        observation_spec=tf_env.observation_spec(),
        name='Player1QNet'
    )

    agent_ai = MultiDQNAgent(
        tf_env,
        action_spec=tf_env.action_spec()['position'],
        action_fn=partial(action_fn, ai_player),
        name='PlayerAI',
        q_network=temp_q_net
    )

    agent_ai.set_policy(saved_policy)

    agent_human = HumanAgent(
        tf_env,
        action_spec=tf_env.action_spec()['position'],
        action_fn=partial(action_fn, human_player),
        name='PlayerHuman'
    )

    if human_player == 1:
        players = cycle([agent_human, agent_ai])
    else:
        players = cycle([agent_ai, agent_human])

    ts = tf_env.reset()

    reward = None
    player = None
    while not ts.is_last():
        board_state = np.squeeze(tf_env.render().numpy())
        table_str = '''
                  a | b | c | d | e | f | g | h | i | j | k | l | m | n | o | p | q | r | s
                a {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                b {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                c {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                d {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                e {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                f {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                g {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                h {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                i {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                j {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                k {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                l {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                m {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                n {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                o {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                p {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                q {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                r {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                - - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -
                s {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}
                '''.format(*tuple(board_state.flatten()))
        table_str = table_str.replace('0', ' ')
        table_str = table_str.replace('1', 'B')
        table_str = table_str.replace('2', 'W')
        print(table_str)

        if reward not in [REWARD_NOT_PASSED, REWARD_ILLEGAL_MOVE]:
            player = next(players)

        if player == agent_human:
            legal_moves = []
            mask = env.get_legal_moves(board_state, human_player)
            for n in range(len(mask)):
                if mask[n] == 1:
                    position = (n // 19, n % 19)
                    legal_moves.append(position)
            legal = False
            choice = (19, 0)
            while not legal:
                choice = input("Input a move in the format row, column e.g. (a, g) or pass by typing pass")
                if choice == 'pass':
                    choice = (19, 0)
                    legal = True
                else:
                    try:
                        choice = tuple(choice.split(', '))
                        choice = (string.ascii_lowercase.index(choice[0]), string.ascii_lowercase.index(choice[1]))
                        if choice in legal_moves:
                            legal = True
                        else:
                            print('not a legal move')
                    except (ValueError, IndexError):
                        print('not a legal move')

            position_flat = (choice[0] * 19) + choice[1]
            human_action = tf.convert_to_tensor(position_flat)

            _, reward = agent_human.act(human_action)

        else:
            _, reward = player.act()
        ts = tf_env.current_time_step()
    return True
