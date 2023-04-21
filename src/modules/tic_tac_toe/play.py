import os
from functools import partial
from itertools import cycle

import PySimpleGUI as sg
import numpy as np
import pygame as pg
import tensorflow as tf
from tf_agents.environments import TFPyEnvironment

from src.modules.draughts.draughtsGUI_pygame import DraughtsGUI_pygame
from src.modules.env_flags import REWARD_NOT_PASSED, REWARD_ILLEGAL_MOVE
from src.modules.tic_tac_toe.tic_tac_toe_multi import TicTacToeMultiAgentEnv
from src.play.human_agent import HumanAgent
from src.train.multi_agent import MultiDQNAgent
from src.train.network import MaskedNetwork
from src.train.train_py_env import action_fn

"""
drawXO and user_click functions adapted from https://www.geeksforgeeks.org/tic-tac-toe-gui-in-python-using-pygame/
"""
def drawXO(row, col, width, height, screen, x_img, o_img):
    global board, XO

    # for the first row, the image
    # should be pasted at a x coordinate
    # of 30 from the left margin
    if row == 1:
        posx = 30

    # for the second row, the image
    # should be pasted at a x coordinate
    # of 30 from the game line
    if row == 2:
        # margin or width / 3 + 30 from
        # the left margin of the window
        posx = width / 3 + 30

    if row == 3:
        posx = width / 3 * 2 + 30

    if col == 1:
        posy = 30

    if col == 2:
        posy = height / 3 + 30

    if col == 3:
        posy = height / 3 * 2 + 30

    # setting up the required board
    # value to display
    board[row - 1][col - 1] = XO

    if (XO == 'x'):

        # pasting x_img over the screen
        # at a coordinate position of
        # (pos_y, posx) defined in the
        # above code
        screen.blit(x_img, (posy, posx))
        XO = 'o'

    else:
        screen.blit(o_img, (posy, posx))
        XO = 'x'
    pg.display.update()


def user_click(width, height, ):
    # get coordinates of mouse click
    x, y = pg.mouse.get_pos()

    # get column of mouse click (1-3)
    if (x < width / 3):
        col = 1

    elif (x < width / 3 * 2):
        col = 2

    elif (x < width):
        col = 3

    else:
        col = None

    # get row of mouse click (1-3)
    if (y < height / 3):
        row = 1

    elif (y < height / 3 * 2):
        row = 2

    elif (y < height):
        row = 3

    else:
        row = None

    # after getting the row and col,
    # we need to draw the images at
    # the desired positions
    if (row and col and board[row - 1][col - 1] is None):
        global XO
        drawXO(row, col)

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
    window = sg.Window('Nought and Crosses against AI', layout)

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

    env = TicTacToeMultiAgentEnv()

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

    gui = DraughtsGUI_pygame(image_dir=os.path.join('.', 'modules', 'draughts', 'images'))

    ts = tf_env.reset()

    # sets up a display window with pygame and renders the board state
    pg.init()

    screen = pg.display.set_mode((400, 500), 0, 32)
    pg.display.set_caption("My Tic Tac Toe")
    initiating_window = pg.image.load("modified_cover.png")
    x_img = pg.image.load("X_modified.png")
    y_img = pg.image.load("o_modified.png")
    initiating_window = pg.transform.scale(
        initiating_window, (400, 500))
    x_img = pg.transform.scale(x_img, (80, 80))
    o_img = pg.transform.scale(y_img, (80, 80))

    pg.display.set_caption("My Tic Tac Toe")

    reward = None
    player = None
    while not ts.is_last():
        board_state = np.squeeze(tf_env.render().numpy())

        if reward not in [REWARD_NOT_PASSED, REWARD_ILLEGAL_MOVE]:
            player = next(players)

        if player == agent_human:
            mask = env.get_legal_actions(board_state)

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
