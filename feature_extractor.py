import numpy as np
import pandas as pd
import tsfel
import pre_processing as pre


X_train = pre.generate_dict()
X_train_sig = X_train['Pulse Rate_BPM']
cfg_file = tsfel.get_features_by_domain()
features = tsfel.time_series_features_extractor(cfg_file, X_train_sig, window_size=2500)
print(features)