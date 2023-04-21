import os

import PySimpleGUI as sg
from src.play.user_interface import import_name
from src.train.train_py_env import train_model


def train_gui():
    layout = [[sg.Text("Choose A Game Environment")],
              [sg.Text("Prebuilt Environments:")],
              [sg.Button('TicTacToe'), sg.Button('Draughts'), sg.Button('Chess'), sg.Button('Go')],
              [sg.Text("Other Environments")],
              [sg.Text('Import an Environment: '), sg.Input(),
               sg.FileBrowse(initial_folder=os.path.join('..', 'modules'), file_types=(("Python Files", "*.py"),),
                             key='-IN-')],
              [sg.Text('Environment Class Name: '), sg.Input(key='-CLASS-')],
              [sg.Button('Ok'), sg.Button('Back')]]

    # Create the window
    window = sg.Window('Train a Model', layout)
    # Display and interact with the Window using an Event Loop
    while True:
        event, values = window.read()
        # See if user wants to quit or window was closed
        if event in ['TicTacToe', 'Draughts', 'Chess', 'Go', 'Ok']:
            window.hide()

            module_dir = "src.modules."

            match event:
                case 'TicTacToe':
                    env_class = import_name(module_dir + "tic_tac_toe.tic_tac_toe_multi", "TicTacToeMultiAgentEnv")
                case 'Draughts':
                    env_class = import_name(module_dir + "draughts.draughts_environment", "DraughtsEnvironment")
                case 'Chess':
                    env_class = import_name(module_dir + "chess.chess_environment", "ChessEnvironment")
                case 'Go':
                    env_class = import_name(module_dir + "go.go_environment", "GoEnvironment")
                case _:
                    env_class = import_name(values['-IN-'], values['-CLASS-'])

            new_layout = [[sg.Text('Choose Training Settings')],
                          [sg.Text('Save Directory: '), sg.Input(),
                           sg.FolderBrowse(initial_folder=os.path.join('..', 'models'), key='-SAVE_DIR-')],
                          [sg.Text('Replay Buffer Size: '), sg.Input(key='-BUFFER-')],
                          [sg.Text('Replay Buffer suggested size (Usually a bit bigger for safety):')],
                          [sg.Text('different_pieces * episodes_per_iteration * action_spec_length')],
                          [sg.Text('Number of Iterations: '), sg.Input(key='-NUM_ITER-', default_text=4000)],
                          [sg.Text('Initial Collect Episodes: '), sg.Input(key='-COLLECT-', default_text=4000)],
                          [sg.Text('Episodes Per Iteration: '), sg.Input(key='-EPI_ITER-', default_text=5)],
                          [sg.Text('Train Steps Per Iteration: '), sg.Input(key='-TRA_ITER-', default_text=1)],
                          [sg.Text('Training Batch Size: '), sg.Input(key='-TRA_BATCH-', default_text=512)],
                          [sg.Text('Number of Training Steps: '), sg.Input(key='-TRA_STEP-', default_text=2)],
                          [sg.Text('Interval Between Plots: '), sg.Input(key='-PLT_INTER-', default_text=100)],
                          [sg.Text('Learning Rate: '), sg.Input(key='-RATE-', default_text=1e-5)],
                          [sg.Checkbox('Learning rate decay: ', default=False, key='-DEC-')],
                          [sg.Text('Only matters if learning rate is decaying: ')],
                          [sg.Text('Decay Step: '), sg.Input(key='-DEC_STEP-', default_text=400)],
                          [sg.Text('Decay Rate: '), sg.Input(key='-DEC_RATE-', default_text=0.9)],
                          [sg.Text('with Decay:')],
                          [sg.Text('decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)')],
                          [sg.Button('Ok'), sg.Button('Back')]]

            new_window = sg.Window('Train a Model', new_layout)

            while True:
                new_event, new_values = new_window.read()
                if new_event == 'Ok':
                    try:
                        save_dir = new_values['-SAVE_DIR-']
                        replay_buffer_size = int(new_values['-BUFFER-'])
                        num_iterations = int(new_values['-NUM_ITER-'])
                        initial_collect_episodes = int(new_values['-COLLECT-'])
                        episodes_per_iteration = int(new_values['-EPI_ITER-'])
                        train_steps_per_iteration = int(new_values['-TRA_ITER-'])
                        training_batch_size = int(new_values['-TRA_BATCH-'])
                        training_num_steps = int(new_values['-TRA_STEP-'])
                        plot_interval = int(new_values['-PLT_INTER-'])
                        learning_rate = format(float(new_values['-RATE-']))
                        print(learning_rate)
                        learn_exp_decay = new_values['-DEC-']
                        decay_step = int(new_values['-DEC_STEP-'])
                        decay_rate = format(float(new_values['-DEC_RATE-']))

                        new_window.close()

                        train_model(env_class,
                                    save_dir,
                                    replay_buffer_size,
                                    num_iterations_par=num_iterations,
                                    initial_collect_episodes_par=initial_collect_episodes,
                                    episodes_per_iteration_par=episodes_per_iteration,
                                    train_steps_per_iteration_par=train_steps_per_iteration,
                                    training_batch_size_par=training_batch_size,
                                    training_num_steps_par=training_num_steps,
                                    plot_interval_par=plot_interval,
                                    learning_rate_par=learning_rate,
                                    learn_exp_decay=learn_exp_decay,
                                    decay_step_par=decay_step,
                                    decay_rate_par=decay_rate
                                    )
                        break
                    except ValueError:
                        print('Not Provided a Value')
                if new_event == 'Back':
                    new_window.close()
                    break

        if event == sg.WINDOW_CLOSED or event == 'Back':
            window.close()
            break

        window.un_hide()

    return True


if __name__ == "__main__":
    train_gui()
