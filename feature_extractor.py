import numpy as np
import pandas as pd
import tsfel
import pre_processing as pre

X_train = pre.generate_dict('train')
X_train_sig = X_train['F041']
name = ['BP Dia_mmHg', 'BP_mmHg', 'EDA_microsiemens', 'LA Mean BP_mmHg', 'LA Systolic BP_mmHg',
        'Pulse Rate_BPM',
        'Resp_Volts', 'Respiration Rate_BPM']
cfg_file = tsfel.get_features_by_domain()
features = tsfel.time_series_features_extractor(cfg_file, X_train_sig, fs=1000, window_size=2000, header_names=name)
print(features)
