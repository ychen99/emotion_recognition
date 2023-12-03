import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import tsfel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from scipy.io import loadmat
import itertools
from PIL import Image

mpl.use('Qt5Agg')


class PreProcessing:
    def __init__(self, subject, task):
        self.subject = subject
        self.task = task

    def create_df_data(self):
        name = ['BP Dia_mmHg.txt', 'BP_mmHg.txt', 'EDA_microsiemens.txt', 'LA Mean BP_mmHg.txt',
                'LA Systolic BP_mmHg.txt',
                'Pulse Rate_BPM.txt', 'Resp_Volts.txt', 'Respiration Rate_BPM.txt']

        file_start = r"X:\BP4D+_v0.2\Physiology"
        dir_list = os.listdir(file_start)

        column = ['BP Dia_mmHg', 'BP_mmHg', 'EDA_microsiemens', 'LA Mean BP_mmHg', 'LA Systolic BP_mmHg',
                  'Pulse Rate_BPM',
                  'Resp_Volts', 'Respiration Rate_BPM']
        all_data_frames = []

        for dir_ in dir_list:
            for i in range(1, 11):
                bridge = 'T' + f'{i}'
                bridge_data_frames = []
                for name_, col_name in zip(name, column):
                    new_path = os.path.join(file_start, dir_, bridge, name_)

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

    def spilt_in_windows(self, window_size, overlap=0):
        signals = self.select_physiology_signal(self.subject, self.task)
        windows = []
        for signal_ in signals:
            if signal_:
                w = tsfel.signal_window_splitter(signal=signal_, window_size=window_size, overlap=overlap)
                windows.append(w)
        windows_arr = np.array(windows)
        windows_arr = windows_arr.transpose(1, 2, 0)

        print(windows_arr.shape, f'?{self.subject} and {self.task}')

        return windows_arr

    def select_image_files(self):
        """
            Selects specific image files from a ZIP archive based on the subject, task, and frame numbers.

            Parameters:
            subject (str): Subject identifier.
            task (str): Task identifier.

            Returns:
            list: A list of filtered image file paths.
            """
        path = r'X:\BP4D+_v0.2\2D+3D'
        end = f'{self.subject}.zip'
        zip_path = os.path.join(path, end)

        try:
            with ZipFile(zip_path, 'r') as myzip:
                # First, filter for only JPG files to reduce the dataset
                jpg_files = [name for name in myzip.namelist() if (name.endswith('.jpg') and f'{self.task}/' in name)]

            head_pos = self.read_feeature_head_positions()

            # Sublsit to check for each selected image with correct head positions
            sublist = [selected_[0][0][0] for selected_ in head_pos]

            # Filter list based on sublist for each frame
            filtered_list = [s for s in jpg_files for sub in sublist if str(sub) in s]

            return filtered_list

        except FileNotFoundError:
            print(f"File not found: {zip_path}")
            return []
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def frames_from_au(self):
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
        name = f'{self.subject}_{self.task}_'
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

    def read_feeature_head_positions(self):
        """
        Extracts head position data for specific frames from a MATLAB file.
        """
        path = r'X:\BP4D+_v0.2\3DFeatures'
        bridge_path = f"{self.subject}_{self.task}.mat"
        mat_path = os.path.join(path, bridge_path)

        if not os.path.exists(mat_path):
            print(f"File does not exist: {mat_path}")
            return None

        try:
            mat_data = loadmat(mat_path)
            fit_data = mat_data['stereo'][0]
            selected_frames = self.frames_from_au()[:, 0]

            def calculate_column_averages(col_data):
                return [sum(column) / len(column) for column in zip(*col_data)]

            head_positions = [
                [fit_data[i - 1][0], fit_data[i - 1][1], fit_data[i - 1][2]]
                for i in selected_frames
                if i - 1 < len(fit_data) and len(fit_data[i - 1][2]) != 0 and
                   all(value < 22 * average for value, average in
                       zip(fit_data[i - 1][2], calculate_column_averages(fit_data[i - 1][2])))
            ]

        except FileNotFoundError:
            print(f"File not found: {mat_path}")
            return None
        except KeyError:
            print(f"'fit' data not found in the file: {mat_path}")
            return None
        except IndexError as e:
            print(f"Index error: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

        return head_positions

    def plot_head_position(self):
        head_position = self.read_feeature_head_positions()
        first_elements = []
        second_elements = []
        third_elements = []
        size = []

        # Iterate through the list and extract the elements
        for array in head_position:
            first_elements.append(array[2][0])
            second_elements.append(array[2][1])
            third_elements.append(array[2][2])
            size.append(array[0][0])

        # Convert lists to NumPy arrays if you want to perform array operations
        first_elements = np.array(first_elements)
        second_elements = np.array(second_elements)
        third_elements = np.array(third_elements)

        # Plot the lines using matplotlib
        plt.figure()

        # Plot each set of elements. The x-values are just the index of each element.
        plt.plot(size, first_elements, label='x-position')
        plt.plot(size, second_elements, label='y-position')
        plt.plot(size, third_elements, label='z-position')

        # Adding labels and legend
        plt.xlabel('Nr. of frame')
        plt.ylabel('Value')
        plt.title(f'The head position of {self.subject} for the task {self.task}')
        plt.legend()

        # Show the plot
        plt.show()

    def split_arithmetic_sequence(self, arr):
        if len(arr) < 2:  # No need to split if the array is too short
            return [arr]

        sequences = []  # List to hold the result sequences
        current_sequence = [arr[0]]  # Start the first sequence with the first element

        for i in range(1, len(arr)):
            if len(current_sequence) >= 2:
                # Check if the current element continues the arithmetic sequence
                if arr[i] - current_sequence[-1] == current_sequence[1] - current_sequence[0]:
                    current_sequence.append(arr[i])
                else:
                    # If not, add the current sequence to the list and start a new sequence
                    sequences.append(current_sequence)
                    current_sequence = [arr[i]]
            else:
                current_sequence.append(arr[i])

        # Add the last sequence to the list if not empty
        if current_sequence:
            sequences.append(current_sequence)

        return sequences

    def select_start_and_end(self, arr):
        location = []
        for sub_arr in arr:
            start = sub_arr[0]
            end = sub_arr[-1]
            location.append((start, end))
        return location

    def select_physiology_signal(self, sample=1000, fps=25):
        base_path = r'X:\BP4D+_v0.2\Physiology'
        folder_path = f'{self.subject}/{self.task}'
        signal_names = ['BP Dia_mmHg', 'BP_mmHg', 'EDA_microsiemens', 'LA Mean BP_mmHg', 'LA Systolic BP_mmHg',
                        'Pulse Rate_BPM',
                        'Resp_Volts', 'Respiration Rate_BPM']

        head_positions = self.read_head_positions()
        frames = []
        for a in head_positions:
            m = (a[0][0][0] / fps) * sample
            frames.append(m[0])

        frame_sequences = self.split_arithmetic_sequence(frames)
        sequence_locations = self.select_start_and_end(frame_sequences)
        physiology_signals = []
        for signal_name in signal_names:
            file_path = os.path.join(base_path, folder_path, f'{signal_name}.txt')
            try:
                signal_data = pd.read_csv(file_path, header=None)
            except FileNotFoundError as e:
                print(f"Error reading file {file_path}: {e}")
                continue

            signal_segments = []

            for start, end in sequence_locations:
                segment = signal_data.iloc[int(start - 1):int(end)]
                segment_values = segment.values.T
                signal_segments.append(segment_values[0])
            flatten_segments = [item for sublist in signal_segments for item in sublist]
            physiology_signals.append(flatten_segments)

        return physiology_signals


method = PreProcessing('F001', 'T8')
#method.read_feeature_head_positions()


def generate():
    tasks = ['T1', 'T6', 'T7', 'T8']
    file_start = r"X:\BP4D+_v0.2\Physiology"
    subject_list = os.listdir(file_start)
    _dataset = []
    for id in subject_list:
        task_sequence = []
        for t in tasks:
            try:
                method = PreProcessing(f'{id}', f'{t}')
                sequence = method.spilt_in_windows(window_size=3000)
                task_sequence.append(sequence)
            except Exception as e:
                print(f"No such window data {id} and {t} : {e}")
                continue

        _dataset.append(task_sequence)

    x_train = list(itertools.chain(*list(itertools.chain(*_dataset))))
    print(len(x_train), len(x_train[0]), len(x_train[0][0]))
    return x_train

# generate()
