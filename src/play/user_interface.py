import os

import PySimpleGUI as sg


# Import a specified module, allows for user created modules to be accessed
def import_name(module_name, name):
    try:
        module = __import__(module_name, globals(), locals(), [name])
    except ImportError:
        return None
    return vars(module)[name]


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
    while True:
        event, values = window.read()
        # See if user wants to quit or window was closed
        if event in ['TicTacToe', 'Draughts', 'Chess', 'Go', 'Ok']:
            window.hide()
        if event == sg.WINDOW_CLOSED or event == 'Quit':
            window.close()
            break

        module_dir = "src.modules."

        match event:
            case 'TicTacToe':
                play_class = import_name(module_dir + "tic_tac_toe.play", "play")
            case 'Draughts':
                play_class = import_name(module_dir + "draughts.play", "play")
            case 'Chess':
                play_class = import_name(module_dir + "chess.play", "play")
            case 'Go':
                play_class = import_name(module_dir + "go.play", "play")
            case _:
                play_class = import_name(values['-IN-'], "play")

        play_class()

        window.un_hide()

    return True


if __name__ == "__main__":
    play_gui()
