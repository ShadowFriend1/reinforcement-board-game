import os
import time
from functools import partial
from itertools import cycle

import PySimpleGUI as sg
import numpy as np
import pygame as pg
import tensorflow as tf
from pygame.locals import *
from tf_agents.environments import TFPyEnvironment

from src.modules.env_flags import REWARD_NOT_PASSED, REWARD_ILLEGAL_MOVE, REWARD_DRAW_OR_NOT_FINAL
from src.modules.tic_tac_toe.tic_tac_toe_multi import TicTacToeMultiAgentEnv
from src.play.human_agent import HumanAgent
from src.train.multi_agent import MultiDQNAgent
from src.train.network import MaskedNetwork
from src.train.train_py_env import action_fn

"""
game_initiating_window, draw_status, drawXO and user_click functions adapted from:
https://www.geeksforgeeks.org/tic-tac-toe-gui-in-python-using-pygame/
"""
draw = None


def game_initiating_window(screen, initiating_window, width, height, winner):
    # displaying over the screen
    screen.blit(initiating_window, (0, 0))

    white = (255, 255, 255)
    line_color = (0, 0, 0)

    # updating the display
    pg.display.update()
    time.sleep(3)
    screen.fill(white)

    # drawing vertical lines
    pg.draw.line(screen, line_color, (width / 3, 0), (width / 3, height), 7)
    pg.draw.line(screen, line_color, (width / 3 * 2, 0),
                 (width / 3 * 2, height), 7)

    # drawing horizontal lines
    pg.draw.line(screen, line_color, (0, height / 3), (width, height / 3), 7)
    pg.draw.line(screen, line_color, (0, height / 3 * 2),
                 (width, height / 3 * 2), 7)
    draw_status(winner, screen, width, 1, False)


def draw_status(winner, screen, width, player, draw):
    # getting the global variable draw
    # into action
    if player == 1:
        XO = 'x'
    else:
        XO = 'o'

    if winner is None:
        message = XO.upper() + "'s Turn"
    else:
        message = XO.upper() + " won !"
    if draw:
        message = "Game Draw !"

    # setting a font object
    font = pg.font.Font(None, 30)

    # setting the font properties like
    # color and width of the text
    text = font.render(message, True, (255, 255, 255))

    # copy the rendered message onto the board
    # creating a small block at the bottom of the main display
    screen.fill((0, 0, 0), (0, 400, 500, 100))
    text_rect = text.get_rect(center=(width / 2, 500 - 50))
    screen.blit(text, text_rect)
    pg.display.update()


def drawXO(row, col, width, height, screen, x_img, o_img, player):
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

    if player == 2:
        # pasting x_img over the screen
        # at a coordinate position of
        # (pos_y, posx) defined in the
        # above code
        screen.blit(o_img, (posy, posx))
    else:
        screen.blit(x_img, (posy, posx))
    pg.display.update()


def user_click(width, height, board):
    # get coordinates of mouse click
    x, y = pg.mouse.get_pos()

    # get column of mouse click (1-3)
    if x < width / 3:
        col = 1

    elif x < width / 3 * 2:
        col = 2

    elif x < width:
        col = 3

    else:
        col = None

    # get row of mouse click (1-3)
    if y < height / 3:
        row = 1

    elif y < height / 3 * 2:
        row = 2

    elif y < height:
        row = 3

    else:
        row = None

    # after getting the row and col,
    # we need to draw the images at
    # the desired positions
    if row and col and board[row - 1][col - 1] is None:
        return row - 1, col - 1

# Plays against the AI
def play():
    # Creates the first window where the user chooses the model to play against and whether to play first or second
    layout = [[sg.Text('Choose whether to play 1st or 2nd')],
              [sg.Radio('Player 1', 'RADIO1', default=True, key="-PLAYER1-")],
              [sg.Radio('Player 2', 'RADIO1', default=False)],
              [sg.Text('Choose a model to play against: ')],
              [sg.Text('Import a Model: '), sg.Input(),
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

    # creates the environment and human and model agents which will interact with it
    env = TicTacToeMultiAgentEnv()

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

    # loops the player agents using a cycle iterator
    if human_player == 1:
        players = cycle([agent_human, agent_ai])
    else:
        players = cycle([agent_ai, agent_human])

    # Resets the environment
    ts = tf_env.reset()

    # sets up a display window with pygame and renders the board state
    pg.init()

    width = 400
    height = 400

    image_dir = os.path.join('.', 'modules', 'tic_tac_toe', 'icons')

    initiating_window = pg.image.load(os.path.join(image_dir, "modified_cover.png"))
    x_img = pg.image.load(os.path.join(image_dir, "X_modified.png"))
    y_img = pg.image.load(os.path.join(image_dir, "o_modified.png"))

    screen = pg.display.set_mode((width, height + 100), 0, 32)
    pg.display.set_caption("My Tic Tac Toe")
    initiating_window = pg.transform.scale(
        initiating_window, (width, height + 100))
    x_img = pg.transform.scale(x_img, (80, 80))
    o_img = pg.transform.scale(y_img, (80, 80))

    pg.display.set_caption("My Tic Tac Toe")

    winner = None

    global draw
    draw = None

    # renders the initial board state
    game_initiating_window(screen, initiating_window, width, height, winner)

    reward = None
    player = None
    current_player = 1
    # Game loop, uses pygame to render but that actual game and all rules occur in the environment which both agents
    # act on
    while not ts.is_last():
        draw_status(None, screen, width, current_player, False)

        board_state = np.squeeze(tf_env.render().numpy())

        board_str = []

        for y in range(len(board_state)):
            row = []
            for x in range(len(board_state[0])):
                match board_state[y, x]:
                    case 0:
                        row.append(None)
                    case 1:
                        row.append('x')
                    case 2:
                        row.append('o')
            board_str.append(row)

        for y in range(len(board_str)):
            for x in range(len(board_str)):
                if board_str[y][x] == 'x':
                    drawXO(y + 1, x + 1, width, height, screen, x_img, o_img, 1)
                elif board_str[y][x] == 'o':
                    drawXO(y + 1, x + 1, width, height, screen, x_img, o_img, 2)

        if reward not in [REWARD_NOT_PASSED, REWARD_ILLEGAL_MOVE]:
            player = next(players)

        if player == agent_human:
            moved = False
            while not moved:
                for event in pg.event.get():
                    if event.type == QUIT:
                        pg.quit()
                        return True
                    if event.type == MOUSEBUTTONDOWN:
                        move = user_click(width, height, board_str)
                        if move is not None:
                            move_flat = (move[0] * 3) + move[1]
                            human_action = tf.convert_to_tensor(move_flat)
                            moved = True
            _, reward = agent_human.act(human_action)

        else:
            _, reward = player.act()

        if current_player == 1:
            current_player = 2
        else:
            current_player = 1

        ts = tf_env.current_time_step()

    if reward == REWARD_DRAW_OR_NOT_FINAL:
        draw_status(None, screen, width, current_player, True)
    else:
        if current_player == 1:
            current_player = 2
        else:
            current_player = 1
        draw_status(current_player, screen, width, current_player, False)

    board_state = np.squeeze(tf_env.render().numpy())

    board_str = []

    for y in range(len(board_state)):
        row = []
        for x in range(len(board_state[0])):
            match board_state[y, x]:
                case 0:
                    row.append(None)
                case 1:
                    row.append('x')
                case 2:
                    row.append('o')
        board_str.append(row)

    for y in range(len(board_str)):
        for x in range(len(board_str)):
            if board_str[y][x] == 'x':
                drawXO(y + 1, x + 1, width, height, screen, x_img, o_img, 1)
            elif board_str[y][x] == 'o':
                drawXO(y + 1, x + 1, width, height, screen, x_img, o_img, 2)

    while True:
        for event in pg.event.get():
            if event.type == QUIT or event.type == MOUSEBUTTONDOWN or event.type == KEYDOWN:
                pg.quit()
                return True
