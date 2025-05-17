import numpy as np
import os
from scipy.ndimage import uniform_filter1d
import random

# Define gesture labels based on folder order
GESTURE_LABELS = {
    0: 'next_station',
    1: 'previous_station',
    2: 'turn_off_radio',
    3: 'turn_on_radio',
    4: 'volume_down',
    5: 'volume_up'
}

def normalize_signal(signal):
    """Normalize the signal to [0, 1] range, preserving zeros for padding."""
    signal = signal.copy()
    non_zero_mask = np.any(signal != 0, axis=1)
    if np.any(non_zero_mask):
        non_zero_signal = signal[non_zero_mask]
        min_val = non_zero_signal.min()
        max_val = non_zero_signal.max()
        if max_val > min_val:  # Avoid division by zero
            signal[non_zero_mask] = (non_zero_signal - min_val) / (max_val - min_val)
    return signal

def add_gaussian_noise(signal, noise_level=0.01):
    """Add Gaussian noise to the signal, preserving zeros."""
    noise = np.random.normal(0, noise_level, signal.shape)
    signal = signal.copy()
    non_zero_mask = np.any(signal != 0, axis=1)
    signal[non_zero_mask] += noise[non_zero_mask]
    return signal

def scale_amplitude(signal, scale_range=(0.9, 1.1)):
    """Scale the signal amplitude by a random factor, uniform across each frame."""
    signal = signal.copy()
    non_zero_mask = np.any(signal != 0, axis=1)
    for frame_idx in np.where(non_zero_mask)[0]:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        signal[frame_idx] *= scale
    return signal

def frame_shift(signal, max_shift=10):
    """Shift the signal left or right within the frame window."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    signal = signal.copy()
    if shift > 0:
        signal[:-shift] = signal[shift:]
        signal[-shift:] = 0
    elif shift < 0:
        shift = abs(shift)
        signal[shift:] = signal[:-shift]
        signal[:shift] = 0
    return signal

def motion_blur(signal, kernel_size=3):
    """Apply a moving average to simulate motion blur on landmark trajectories."""
    signal = signal.copy()
    non_zero_mask = np.any(signal != 0, axis=1)
    frames = np.where(non_zero_mask)[0]
    if len(frames) > kernel_size:
        for i in range(signal.shape[1]):  # Apply to each coordinate
            signal[frames, i] = uniform_filter1d(signal[frames, i], size=kernel_size, mode='nearest')
    return signal

def random_frame_dropping(signal, drop_rate=0.1):
    """Drop random frames and interpolate to simulate subsampling."""
    signal = signal.copy()
    num_frames = signal.shape[0]
    non_zero_mask = np.any(signal != 0, axis=1)
    frames = np.where(non_zero_mask)[0]
    if len(frames) == 0:
        return signal
    
    num_drop = int(len(frames) * drop_rate)
    drop_indices = np.random.choice(frames, size=num_drop, replace=False)
    signal[drop_indices] = 0  # Set dropped frames to zero
    
    # Interpolate to fill gaps
    for i in range(signal.shape[1]):  # For each coordinate
        non_zero_frames = np.where(np.any(signal != 0, axis=1))[0]
        if len(non_zero_frames) > 1:
            interpolated = np.interp(np.arange(num_frames), non_zero_frames, signal[non_zero_frames, i])
            signal[:, i] = interpolated
    return signal

def transform(signal, label, augmentation_prob=0.7):
    # Normalize the signal first
    signal = normalize_signal(signal)
    
    # Apply augmentations with probability
    if random.random() < augmentation_prob:
        signal = add_gaussian_noise(signal, noise_level=0.01)
    
    if random.random() < augmentation_prob:
        signal = scale_amplitude(signal, scale_range=(0.9, 1.1))
    
    if random.random() < augmentation_prob:
        signal = frame_shift(signal, max_shift=10)
    
    if random.random() < augmentation_prob:
        signal = motion_blur(signal, kernel_size=3)
    
    if random.random() < augmentation_prob:
        signal = random_frame_dropping(signal, drop_rate=0.1)
    
    # Final normalization to ensure consistency
    signal = normalize_signal(signal)
    
    return {'signal': signal, 'label': label}

def main():
    """Augment all samples from X_radio.npy and y_radio.npy and save to x_augmented.npy and y_augmented.npy."""
    # Load the full dataset
    X_radio = np.load('X_radio.npy')  # Shape: (61, 82, 63)
    y_radio = np.load('y_radio.npy')  # Shape: (61,)
    
    # Lists to store augmented data
    augmented_signals = []
    augmented_labels = []
    
    # Augment each sample
    for idx in range(X_radio.shape[0]):
        signal = X_radio[idx]  # Shape: (82, 63)
        label = int(y_radio[idx])  # Integer label (0 to 5)
        
        # Apply augmentations
        result = transform(signal, label)
        augmented_signal = result['signal']
        label = result['label']
        
        # Append to lists
        augmented_signals.append(augmented_signal)
        augmented_labels.append(label)
        
        print(f"Augmented sample {idx+1}/{X_radio.shape[0]} for gesture '{GESTURE_LABELS[label]}'")
    
    # Convert lists to NumPy arrays
    x_augmented = np.array(augmented_signals)  # Shape: (61, 82, 63)
    y_augmented = np.array(augmented_labels)  # Shape: (61,)
    
    # Save the augmented data
    np.save('X_augmented.npy', x_augmented)
    np.save('y_augmented.npy', y_augmented)
    print(f"Saved augmented data to 'x_augmented.npy' (shape: {x_augmented.shape}) and 'y_augmented.npy' (shape: {y_augmented.shape})")

if __name__ == "__main__":
    main()