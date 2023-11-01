import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')

file_start = r"X:\BP4D+_v0.2\Physiology"
dir_list = os.listdir(file_start)
file_end = 'Pulse Rate_BPM.txt'

''''
label = pd.DataFrame({'label': [1] * 100})

for x in dir_list:
    new_path = os.path.join(path, x, file_end)
    if not os.path.exists(new_path):
        print(f'{new_path} is not exist')
        continue
    try:
        df_data = pd.read_csv(new_path, header=None)
    except FileNotFoundError:
        print(f'Error reading {new_path}')
        continue
    df_data.columns = [x]
    new_data = pd.concat([label, df_data], axis=1)
    label = new_data
'''


def create_df_data(path):
    columns = [f'{x}' for x in dir_list]
    #columns = ['label'] + columns
    new_data = pd.DataFrame(columns=columns)
    for i in range(1, 10):
        bridge = 'T' + f'{i}'
        label = pd.DataFrame({'label': [i] * 100})
        for x in dir_list:
            new_path = os.path.join(path, x, bridge, file_end)
            if not os.path.exists(new_path):
                print(f'{new_path} is not exist')
                continue
            try:
                df_data = pd.read_csv(new_path, header=None)
            except FileNotFoundError:
                print(f'Error reading {new_path}')
                continue
            n = df_data.shape[0]

            new_data_ = pd.concat([label, df_data], axis=1)
            label = new_data_
        print(new_data_.shape, new_data.shape)
        new_data = pd.concat([new_data.reset_index(drop=True),new_data_.reset_index(drop=True)],ignore_index=True)

    # new_data.columns = ['label'] + columns
    return new_data

data = create_df_data(file_start)
print(data.info)
