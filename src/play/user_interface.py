import os

import PySimpleGUI as sg
from src.modules.tic_tac_toe.tic_tac_toe_multi import TicTacToeMultiAgentEnv
from src.modules.chess.chess_environment import ChessEnvironment
from src.modules.go.go_environment import GoEnvironment
from src.modules.draughts.draughts_environment import DraughtsEnvironment


def play_gui():
    layout = [[sg.Text("Choose A Game Environment")],
              [sg.Text("Prebuilt Environments:")],
              [sg.Button('TicTacToe'), sg.Button('Draughts'), sg.Button('Chess'), sg.Button('Go')],
              [sg.Text("Other Environments")],
              [sg.Text('Import an Environment: '), sg.Input(),
               sg.FileBrowse(initial_folder=os.path.join('..', 'modules'), file_types=(("Python Files", "*.py"),),
                             key='-IN-')],
              [sg.Button('Ok'), sg.Button('Quit')]]

    # Create the window
    window = sg.Window('Play Against AI', layout)
    # Display and interact with the Window using an Event Loop
    game = None
    while True:
        event, values = window.read()
        # See if user wants to quit or window was closed
        if event in ['TicTacToe', 'Draughts', 'Chess', 'Go']:
            game = event
            break
        if event == sg.WINDOW_CLOSED or event == 'Quit':
            break
        # Output a message to the window
        window['-OUTPUT-'].update('Hello ' + values['-INPUT-'] + "! Thanks for trying PySimpleGUI")

    # Finish up by removing from the screen
    window.close()

    match event:
        case 'TicTacToe':
            print(1)
            return False
        case 'Draughts':
            print(2)
            return False
        case 'Chess':
            print(3)
            return False
        case 'Go':
            print(4)
            return False


if __name__ == "__main__":
    play_gui()
