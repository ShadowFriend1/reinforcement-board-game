import os

import PySimpleGUI as sg

def play():
    layout = [[sg.Text("Choose a model to play against: ")],
              [sg.Text('Import an Environment: '), sg.Input(),
               sg.FileBrowse(initial_folder=os.path.join('..', 'modules'), file_types=(("Python Files", "*.py"),),
                             key='-IN-')],
              [sg.Button('Ok'), sg.Button('Quit')]]

    # Create the window
    window = sg.Window('Chess against AI', layout)

    # Display and interact with the Window using an Event Loop
    while True:
        event, values = window.read()
        # See if user wants to quit or window was closed
        if event in ['TicTacToe', 'Draughts', 'Chess', 'Go']:
            window.close()
            break
        if event == sg.WINDOW_CLOSED or event == 'Quit':
            window.close()
            return True
        # Output a message to the window
        window['-OUTPUT-'].update('Hello ' + values['-INPUT-'] + "! Thanks for trying PySimpleGUI")
