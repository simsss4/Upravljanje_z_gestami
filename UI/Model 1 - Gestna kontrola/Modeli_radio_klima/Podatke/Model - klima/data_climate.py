import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to load .npy files from a directory and return them as a list of arrays
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

# Function to process the data: Padding and reshaping
def process_data(data, max_length=None):
    data_padded = pad_sequences(data, padding='post', dtype='float32', maxlen=max_length)
    print(f"Data after padding has shape: {data_padded.shape}")
    data_reshaped = data_padded.reshape((data_padded.shape[0], data_padded.shape[1], -1))
    return data_reshaped

# Load only climate data, labels from 0
data_climate, labels_climate, mapping_climate = load_data_from_directory('data_climate', start_label=0)

# Procesiraj
max_length_climate = max(len(d) for d in data_climate)
X_climate = process_data(data_climate, max_length=max_length_climate)
y_climate = labels_climate

# Shrani
np.save('X_climate.npy', X_climate)
np.save('y_climate.npy', y_climate)

print("CLIMATE label mapping:", mapping_climate)
print(f"X_climate shape: {X_climate.shape}, y_climate shape: {y_climate.shape}")
