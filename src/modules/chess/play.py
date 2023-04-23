import os
from functools import partial
from itertools import cycle

import PySimpleGUI as sg
import numpy as np
import tensorflow as tf
from tf_agents.environments import TFPyEnvironment

from src.modules.chess.ChessGUI_pygame import ChessGUI_pygame
from src.modules.chess.chess_environment import ChessEnvironment
from src.modules.env_flags import REWARD_NOT_PASSED, REWARD_ILLEGAL_MOVE
from src.play.human_agent import HumanAgent
from src.train.multi_agent import MultiDQNAgent
from src.train.network import MaskedNetwork
from src.train.train_py_env import action_fn

# Plays against the AI
def play():
    # Creates the first window where the user chooses the model to play against and whether to play first or second
    layout = [[sg.Text('Choose whether to play 1st or 2nd')],
              [sg.Radio('Player 1', 'RADIO1', default=True, key="-PLAYER1-")],
              [sg.Radio('Player 2', 'RADIO1', default=False)],
              [sg.Text('Choose a model to play against: ')],
              [sg.Text('Import an Environment: '), sg.Input(),
               sg.FolderBrowse(initial_folder=os.path.join('..', 'models', 'chess', 'saved'), key='-IN-')],
              [sg.Text('Note, playing against a player 1 AI as player 1 or vice versa may cause issues')],
              [sg.Button('Ok'), sg.Button('Back')]]

    # Create the window
    window = sg.Window('Chess against AI', layout)

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

    # creates the environment and human and model agents which will interact with it
    env = ChessEnvironment()

    tf_env = TFPyEnvironment(env)

    if values['-PLAYER1-']:
        ai_player = 2
        human_player = 1
    else:
        ai_player = 1
        human_player = 2

    # temporary q_net instantiated to create basic policy for agent to be replaced
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

    # sets the models policy to the loaded policy
    agent_ai.set_policy(saved_policy)

    # A human agent that can interact with the actual environment
    agent_human = HumanAgent(
        tf_env,
        action_spec=tf_env.action_spec()['position'],
        action_fn=partial(action_fn, human_player),
        name='PlayerHuman'
    )

    # creates a pygame GUI for the game that will handle rendering
    gui = ChessGUI_pygame(image_dir=os.path.join('.', 'modules', 'chess', 'images'))

    # loops the player agents using a cycle iterator
    if human_player == 1:
        players = cycle([agent_human, agent_ai])
    else:
        players = cycle([agent_ai, agent_human])

    # Resets the environment
    ts = tf_env.reset()

    reward = None
    player = None
    # Game loop, uses pygame to render but that actual game and all rules occur in the environment which both agents
    # act on
    while not ts.is_last():
        board_state = np.squeeze(tf_env.render().numpy())
        board_str = board_state.astype(str)
        board_str[board_state == 0] = 'e'
        board_str[board_state == 1] = 'wP'
        board_str[board_state == 2] = 'wT'
        board_str[board_state == 3] = 'wB'
        board_str[board_state == 4] = 'wR'
        board_str[board_state == 5] = 'wQ'
        board_str[board_state == 6] = 'wK'
        board_str[board_state == 7] = 'bP'
        board_str[board_state == 8] = 'bT'
        board_str[board_state == 9] = 'bB'
        board_str[board_state == 10] = 'bR'
        board_str[board_state == 11] = 'bQ'
        board_str[board_state == 12] = 'bK'
        board_str = board_str.tolist()
        gui.Draw(board_str)

        if reward not in [REWARD_NOT_PASSED, REWARD_ILLEGAL_MOVE]:
            player = next(players)

        if player == agent_human:
            legal_moves = []
            mask = env.get_legal_moves(board_state, human_player)
            for n in range(len(mask)):
                if mask[n] == 1:
                    position_index = n // 64
                    move_index = n % 64
                    position = (position_index // 8, position_index % 8)
                    move = (move_index // 8, move_index % 8)
                    legal_moves.append((position, move))

            player_move = gui.GetPlayerInput(board_str, legal_moves)
            position = player_move[0]
            move = player_move[1]
            position_flat = (position[0] * 8) + position[1]
            move_flat = (move[0] * 8) + move[1]
            human_action = tf.convert_to_tensor((position_flat * 64) + move_flat)

            _, reward = agent_human.act(human_action)

        else:
            _, reward = player.act()
        ts = tf_env.current_time_step()

    gui.close()
    return True
