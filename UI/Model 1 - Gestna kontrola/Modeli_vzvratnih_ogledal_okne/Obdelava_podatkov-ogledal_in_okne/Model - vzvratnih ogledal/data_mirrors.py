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

# Nastavi pot do ogledal
data_mirrors, labels_mirrors, mapping_mirrors = load_data_from_directory('data_mirrors', start_label=0)
max_length_mirrors = max(len(d) for d in data_mirrors)
X_mirrors = process_data(data_mirrors, max_length=max_length_mirrors)
y_mirrors = labels_mirrors

np.save('X_mirrors.npy', X_mirrors)
np.save('y_mirrors.npy', y_mirrors)

print("MIRRORS label mapping:", mapping_mirrors)
print(f"X_mirrors shape: {X_mirrors.shape}, y_mirrors shape: {y_mirrors.shape}")
