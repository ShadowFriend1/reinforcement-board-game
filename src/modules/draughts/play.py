import os
from functools import partial

import PySimpleGUI as sg
import tensorflow as tf
from tf_agents.environments import TFPyEnvironment

from src.modules.draughts.draughts_environment import DraughtsEnvironment
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
            window.close()
            break
        if event == sg.WINDOW_CLOSED or event == 'Back':
            window.close()
            return True
        # Output a message to the window
    print(values['-IN-'])

    saved_policy = tf.saved_model.load(values['-IN-'])

    env = DraughtsEnvironment()

    tf_env = TFPyEnvironment(env)
    tf_env.reset()

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

    player_ai = MultiDQNAgent(
        tf_env,
        action_spec=tf_env.action_spec()['position'],
        action_fn=partial(action_fn, ai_player),
        name='PlayerAI',
        q_network=temp_q_net
    )

    player_ai.set_policy(saved_policy)

    player_human = HumanAgent(
        tf_env,
        action_spec=tf_env.action_spec()['position'],
        action_fn=partial(action_fn, human_player),
        name='PlayerHuman'
    )
