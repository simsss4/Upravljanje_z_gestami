import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to load .npy files from a directory and return them as a list of arrays
def load_data_from_directory(directory):
    data = []
    labels = []
    
    for label, subdir in enumerate(os.listdir(directory)):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.npy'):
                    file_path = os.path.join(subdir_path, file)
                    sequence = np.load(file_path)  # Load .npy file
                    
                    # Check the shape of the loaded data
                    print(f"Loaded {file_path} with shape {sequence.shape}")
                    
                    data.append(sequence)  # Append sequence to the data list
                    labels.append(label)  # The folder name will be the label
                    
    return data, np.array(labels)

# Function to process the data: Padding, Reshaping, and Scaling
def process_data(data, max_length=None):
    # Padding the sequences to ensure all have the same length
    data_padded = pad_sequences(data, padding='post', dtype='float32', maxlen=max_length)
    
    # Verify the shape after padding
    print(f"Data after padding has shape: {data_padded.shape}")
    
    # Reshape the data for LSTM input
    # The LSTM expects data in the shape: (samples, time_steps, features)
    data_reshaped = data_padded.reshape((data_padded.shape[0], data_padded.shape[1], -1))
    
    return data_reshaped

# Load data from the three directories
data_climate, labels_climate = load_data_from_directory('data_climate')
data_mirrors, labels_mirrors = load_data_from_directory('data_mirrors')
data_radio, labels_radio = load_data_from_directory('data_radio')
data_windows, labels_windows = load_data_from_directory('data_windows')

# Padding all data to ensure consistent sequence lengths
max_length = max(max(len(d) for d in data_mirrors),
                 max(len(d) for d in data_climate), 
                 max(len(d) for d in data_radio), 
                 max(len(d) for d in data_windows))

# Pad data from all sources
data_climate_padded = pad_sequences(data_radio, padding='post', dtype='float32', maxlen=max_length)
data_mirrors_padded = pad_sequences(data_mirrors, padding='post', dtype='float32', maxlen=max_length)
data_radio_padded = pad_sequences(data_radio, padding='post', dtype='float32', maxlen=max_length)
data_windows_padded = pad_sequences(data_windows, padding='post', dtype='float32', maxlen=max_length)

# Concatenate data from all sources after padding
X = np.concatenate((data_climate_padded, data_mirrors_padded, data_radio_padded, data_windows_padded), axis=0)
y = np.concatenate((labels_climate, labels_mirrors, labels_radio, labels_windows), axis=0)

# Process data (ensure proper padding and reshaping)
X_processed = process_data(X)

# Optionally, you can save the processed data
np.save('X.npy', X_processed)
np.save('y.npy', y)

# Print out the shape of the final data
print(f"Final shape of processed data: {X_processed.shape}")
print(f"Final shape of labels: {y.shape}")
