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
    """Demo: Generate 10 augmented gesture signals and save them to disk."""
    X_radio = np.load('X_radio.npy')
    y_radio = np.load('y_radio.npy')
    
    augmented_signals = []
    augmented_labels = []

    indices = np.random.choice(len(X_radio), size=10, replace=False)

    for i, idx in enumerate(indices):
        signal = X_radio[idx]
        label = int(y_radio[idx])
        result = transform(signal, label)
        augmented_signals.append(result['signal'])
        augmented_labels.append(result['label'])
        print(f"[{i+1}/10] Augmented gesture: {GESTURE_LABELS[result['label']]}")

    x_aug = np.array(augmented_signals)
    y_aug = np.array(augmented_labels)

    np.save("demo_X_augmented.npy", x_aug)
    np.save("demo_y_augmented.npy", y_aug)
    print("Demo saved: 'demo_X_augmented.npy', 'demo_y_augmented.npy'")

if __name__ == "__main__":
    main()