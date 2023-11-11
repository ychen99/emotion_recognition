import os
import numpy as np
import pandas as pd
import tsfel
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl

name = ['BP Dia_mmHg.txt', 'BP_mmHg.txt', 'EDA_microsiemens.txt', 'LA Mean BP_mmHg.txt', 'LA Systolic BP_mmHg.txt',
            'Pulse Rate_BPM.txt', 'Resp_Volts.txt', 'Respiration Rate_BPM.txt']


def plot():
    mpl.use('TkAgg')
    path = r'X:\BP4D+_v0.2\Physiology\F001\T1'

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    # Flatten the array of AxesSubplot objects for easier iteration
    axs = axs.flatten()

    for idx, n in enumerate(name):
        pa = os.path.join(path, n)
        signal = pd.read_csv(pa, header=None)

        axs[idx].plot(signal.iloc[:, 0])
        axs[idx].set_title(f"File: {n}")  # Optional: Set a title for each subplot

    # Hide any unused subplots if there are less than 8 files
    for idx in range(len(name), len(axs)):
        fig.delaxes(axs[idx])

    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.show()  # Display the figure


def cal():
    file_start = r"X:\BP4D+_v0.2\Physiology"
    dir_list = os.listdir(file_start)

    x_train = []
    labels = []
    for dir_ in dir_list:
        for i in range(1, 11):
            bridge = 'T' + f'{i}'
            window = []
            signal_arr = []
            for name_ in name:
                new_path = os.path.join(file_start, dir_, bridge, name_)
                if os.path.exists(new_path):
                    signal_ = pd.read_csv(new_path, header=None)
                    signal_arr.append(signal_.values)
                else:
                    print(f'{new_path} is not exist')
            print(dir_, i)
            signal = np.concatenate(signal_arr, axis=1)
            w = tsfel.signal_window_splitter(signal=signal, window_size=2000, overlap=0.2)
            window.append(w)
            labels.append(np.repeat(i, len(w)))
        x_train.append(window)

    print(len(x_train), len(x_train[0]), len(x_train[0][0]), len(x_train[0][0][0]))


def process(signal):
    max_size = 0
    if signal.shape[0] < max_size:
        # fill
        padding = np.full((max_size - signal.shape[0], signal.shape[1]), np.nan)
        signal = np.vstack((signal.values, padding))
    else:
        signal = signal.values
    return signal