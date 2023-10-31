import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')


file_path = r'X:\BP4D+_v0.2\Physiology\F001\T1\Pulse Rate_BPM.txt'

with open(file_path) as f:
    data = f.read()
    data_into_list = data.split("\n")


print(len(data_into_list))
plt.figure(figsize=(16, 10))
plt.plot(data_into_list)
plt.show()