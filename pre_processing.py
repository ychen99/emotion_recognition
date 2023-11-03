import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')

file_start = r"X:\BP4D+_v0.2\Physiology"
dir_list = os.listdir(file_start)
file_end = 'Resp_Volts'+'.txt'


def create_df_data(path):
    columns = [f'{x}' for x in dir_list]
    new_data = pd.DataFrame()

    for i in range(1, 11):
        bridge = 'T' + f'{i}'
        dfs = pd.DataFrame()

        for x in dir_list:
            new_path = os.path.join(path, x, bridge, file_end)
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


data = create_df_data(file_start)
# data.to_csv('X:\BP4D+_v0.2\Physiology\Pulse_Rate_BPM.csv', index=False,  encoding='utf-8')
print(data.info)

name = ['BP Dia_mmHg', 'BP_mmHg', 'EDA_microsiemens','LA Mean BP_mmHg','LA Systolic BP_mmHg','Pulse Rate_BPM','Resp_Volts','Respiration Rate_BPM']