import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data_from_directory(directory, start_label=0):
    data = []
    labels = []
    label_mapping = {}

    for offset, subdir in enumerate(sorted(os.listdir(directory))):
        subdir_path = os.path.join(directory, subdir)
        label = start_label + offset
        label_mapping[subdir] = label

        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.npy'):
                    file_path = os.path.join(subdir_path, file)
                    sequence = np.load(file_path)
                    print(f"Loaded {file_path} with shape {sequence.shape}")
                    data.append(sequence)
                    labels.append(label)

    return data, np.array(labels), label_mapping

def process_data(data, max_length=None):
    data_padded = pad_sequences(data, padding='post', dtype='float32', maxlen=max_length)
    print(f"Data after padding has shape: {data_padded.shape}")
    data_reshaped = data_padded.reshape((data_padded.shape[0], data_padded.shape[1], -1))
    return data_reshaped

# Nastavi pot do oken
data_windows, labels_windows, mapping_windows = load_data_from_directory('data_windows', start_label=0)
max_length_windows = max(len(d) for d in data_windows)
x_windows = process_data(data_windows, max_length=max_length_windows)
y_windows = labels_windows

np.save('x_windows.npy', x_windows)
np.save('y_windows.npy', y_windows)

print("WINDOWS label mapping:", mapping_windows)
print(f"x_windows shape: {x_windows.shape}, y_windows shape: {y_windows.shape}")

