import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

mpl.use('TkAgg')


def create_df_data(start, dir_name, end):
    columns = [f'{x}' for x in dir_name]
    new_data = pd.DataFrame()

    for i in range(1, 11):
        bridge = 'T' + f'{i}'
        dfs = pd.DataFrame()

        for x in dir_name:
            new_path = os.path.join(start, x, bridge, end)
            if os.path.exists(new_path):
                df_data = pd.read_csv(new_path, header=None)
                df_combined = pd.concat([dfs, df_data], axis=1)
                dfs = df_combined
            else:
                dfs[x] = [np.nan for _ in range(dfs.shape[0])]
                print(f'{new_path} is not exist')
                continue
        dfs.columns = columns
        label = pd.DataFrame({'label': [i] * dfs.shape[0]})
        bridge_data = pd.concat([label, dfs], axis=1)
        new_data = pd.concat([new_data, bridge_data], axis=0)
    return new_data


def generate_dict():
    file_start = r"X:\BP4D+_v0.2\Physiology"
    dir_list = os.listdir(file_start)

    name = ['BP Dia_mmHg', 'BP_mmHg', 'EDA_microsiemens', 'LA Mean BP_mmHg', 'LA Systolic BP_mmHg', 'Pulse Rate_BPM',
            'Resp_Volts', 'Respiration Rate_BPM']
    dir_train, dir_test = train_test_split(dir_list, test_size=0.3)
    data_dict = {}
    for name_ in name:
        file_end = name_ + '.txt'
        data = create_df_data(file_start, dir_train, file_end)
        data_dict[name_] = data
    return data_dict


# phy_dict = generate_dict()
# data.to_csv('X:\BP4D+_v0.2\Physiology\Pulse_Rate_BPM.csv', index=False,  encoding='utf-8')
