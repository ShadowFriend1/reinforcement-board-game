import PySimpleGUI as sg

if __name__ == "__main__":
    layout = [[sg.Button('Hello World', size=(30, 4))]]
    window = sg.Window('GUI SAMPLE', layout, size=(200, 100))
    event, values = window.read()
