import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import tsfel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from zipfile import ZipFile

mpl.use('TkAgg')


def create_df_data(start, dir_list):
    name = ['BP Dia_mmHg.txt', 'BP_mmHg.txt', 'EDA_microsiemens.txt', 'LA Mean BP_mmHg.txt', 'LA Systolic BP_mmHg.txt',
            'Pulse Rate_BPM.txt', 'Resp_Volts.txt', 'Respiration Rate_BPM.txt']

    file_start = r"X:\BP4D+_v0.2\Physiology"
    dir_list = os.listdir(file_start)

    column = ['BP Dia_mmHg', 'BP_mmHg', 'EDA_microsiemens', 'LA Mean BP_mmHg', 'LA Systolic BP_mmHg', 'Pulse Rate_BPM',
              'Resp_Volts', 'Respiration Rate_BPM']
    all_data_frames = []

    for dir_ in dir_list:
        for i in range(1, 11):
            bridge = 'T' + f'{i}'
            bridge_data_frames = []
            for name_, col_name in zip(name, column):
                new_path = os.path.join(start, dir_, bridge, name_)

                if os.path.exists(new_path):
                    df_data = pd.read_csv(new_path, header=None)
                    df_data.columns = [col_name]
                    bridge_data_frames.append(df_data)
                else:
                    print(f'{new_path} is not exist')
            if bridge_data_frames:
                # Merge all data from the same bridge
                bridge_df = pd.concat(bridge_data_frames, axis=1)

                bridge_df.insert(0, 'id', dir_)
                bridge_df.insert(1, 'label', i)
                all_data_frames.append(bridge_df)

    # Merge all data frames at once
    result_data = pd.concat(all_data_frames, ignore_index=True)

    return result_data


def spilt_in_window(window_size, overlap=0.1, data=8):
    for i in range(1, 11):
        data = 9


def select_image_files(subject, task):
    """
        Selects specific image files from a ZIP archive based on the subject, task, and frame numbers.

        Parameters:
        subject (str): Subject identifier.
        task (str): Task identifier.

        Returns:
        list: A list of filtered image file paths.
        """
    path = r'X:\BP4D+_v0.2\2D+3D'
    end = subject + '.zip'
    zip_path = os.path.join(path, end)

    try:
        with ZipFile(zip_path, 'r') as myzip:
            # First, filter for only JPG files to reduce the dataset
            jpg_files = [name for name in myzip.namelist() if name.endswith('.jpg')]

        selected_frames = frames_from_au(subject, task)[:, 0]
        selected_images = []

        for selected_ in selected_frames:
            # Substrings to check for each selected frame
            substrings = [subject, task, f'{selected_}']

            # Filter list based on substrings for each frame
            filtered_list = [s for s in jpg_files if all(sub in s for sub in substrings)]
            selected_images.extend(filtered_list)
        # print(selected_images)
        return selected_images

    except FileNotFoundError:
        print(f"File not found: {zip_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def frames_from_au(subject, task):
    """
    Extracts and merges the second column of Action Unit (AU) data from multiple files,
    corresponding to a given subject and task. Places the first column (which is the same
    across all files) as the first column in the final array.

    Parameters:
    subject (str): The subject identifier.
    task (str): The task identifier.

    Returns:
    numpy.ndarray: An array containing the merged AU data.
    """
    path = r'X:\BP4D+_v0.2\AUCoding\AU_INT'
    dir_list = os.listdir(path)
    name = subject + '_' + task + '_'
    aus = []
    column = None

    for dir_ in dir_list:
        file_name = name + dir_ + '.csv'
        file_path = os.path.join(path, dir_, file_name)

        try:
            au_data = np.array(pd.read_csv(file_path, header=None))
            aus.append(au_data[:, 1])
            column = au_data[:, 0]
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

    if not aus:
        return np.array([])

    if column is not None:
        aus.insert(0, column)

    # Combine all data into a single array
    AUs = np.array(aus).T
    return AUs


# frames_from_au(subject='F001', task='T1')

select_image_files(subject='F001', task='T1')
