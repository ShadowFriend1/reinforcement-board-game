import PySimpleGUI as sg

from play.user_interface import play_gui
from train.user_interface import train_gui


def app_gui():
    layout = [[sg.Text("Test or Train an AI model")],
              [sg.Button('Play against AI')],
              [sg.Button('Train Model')],
              [sg.Button('Quit')]]
    # Create the window
    window = sg.Window('Reinforcement Learning AI App', layout)
    # Display and interact with the Window using an Event Loop
    while True:
        event, values = window.read()
        # See if user wants to quit or window was closed
        if event in ['Play against AI', 'Train Model', 'Quit']:
            window.hide()
            if event == 'Play against AI':
                play_gui()
            if event == 'Train Model':
                train_gui()
            window.un_hide()
        if event == sg.WINDOW_CLOSED or event == 'Quit':
            break
    window.close()


if __name__ == "__main__":
    app_gui()
