import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

mpl.use('TkAgg')

name = ['BP Dia_mmHg.txt', 'BP_mmHg.txt', 'EDA_microsiemens.txt', 'LA Mean BP_mmHg.txt', 'LA Systolic BP_mmHg.txt',
        'Pulse Rate_BPM.txt', 'Resp_Volts.txt', 'Respiration Rate_BPM.txt']

file_start = r"X:\BP4D+_v0.2\Physiology"
dir_list = os.listdir(file_start)

column = ['BP Dia_mmHg', 'BP_mmHg', 'EDA_microsiemens', 'LA Mean BP_mmHg', 'LA Systolic BP_mmHg','Pulse Rate_BPM', 'Resp_Volts', 'Respiration Rate_BPM']


def create_df_data(start, dir_list):
    all_data_frames = []

    for dir_ in dir_list:
        for i in range(1, 11):
            bridge = 'T' + f'{i}'
            bridge_data_frames = []
            for name_ in name:
                new_path = os.path.join(start, dir_, bridge, name_)

                if os.path.exists(new_path):
                    df_data = pd.read_csv(new_path, header=None)
                    bridge_data_frames.append(df_data)
                else:
                    print(f'{new_path} is not exist')
            if bridge_data_frames:
                # Merge all data from the same bridge
                bridge_df = pd.concat(bridge_data_frames, axis=1)

                bridge_df['label'] = i
                bridge_df['id'] = dir_
                all_data_frames.append(bridge_df)

    # Merge all data frames at once
    result_data = pd.concat(all_data_frames, ignore_index=True)

    reversed_columns = column[::-1]
    result_data.columns = reversed_columns + ['label']+['id']
    return result_data


data = create_df_data(file_start, dir_list)
print(data.inf0)
