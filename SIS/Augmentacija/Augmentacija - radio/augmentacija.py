import numpy as np
from scipy.ndimage import uniform_filter1d
import random
import matplotlib.pyplot as plt

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
    return signal  # Remove clipping here; final normalization will handle it

def scale_amplitude(signal, scale_range=(0.8, 1.2)):
    """Scale the signal amplitude by a random factor, uniform across each frame."""
    signal = signal.copy()
    non_zero_mask = np.any(signal != 0, axis=1)
    for frame_idx in np.where(non_zero_mask)[0]:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        signal[frame_idx] *= scale
    return signal  # Remove clipping here; final normalization will handle it

def frame_shift(signal, max_shift=5):
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

def random_frame_dropping(signal, drop_rate=0.05):
    signal = signal.copy()
    non_zero_mask = np.any(signal != 0, axis=1)
    frames = np.where(non_zero_mask)[0]
    if len(frames) == 0:
        return signal
    num_drop = int(len(frames) * drop_rate)
    drop_indices = np.random.choice(frames, size=num_drop, replace=False)
    signal[drop_indices] = 0  # Drop frames, no interpolation
    return signal

def rotate_landmarks(signal, max_angle=10):
    signal = signal.copy()
    non_zero_mask = np.any(signal != 0, axis=1)
    angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180  # Convert to radians
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    for frame_idx in np.where(non_zero_mask)[0]:
        landmarks = signal[frame_idx].reshape(21, 3)
        wrist = landmarks[0]
        landmarks -= wrist  # Center at wrist
        landmarks = (rotation_matrix @ landmarks.T).T  # Rotate
        landmarks += wrist  # Restore position
        signal[frame_idx] = landmarks.flatten()
    return signal

def transform(signal, label, augmentation_prob=0.5, max_augmentations=3):
    signal = normalize_signal(signal)
    augmentations = [
        lambda s: add_gaussian_noise(s, noise_level=0.01),
        lambda s: scale_amplitude(s, scale_range=(0.8, 1.2)),
        lambda s: frame_shift(s, max_shift=5),
        lambda s: motion_blur(s, kernel_size=3),
        lambda s: random_frame_dropping(s, drop_rate=0.05),
        lambda s: rotate_landmarks(s, max_angle=10)
    ]
    selected = random.sample(augmentations, k=min(max_augmentations, len(augmentations)))
    for aug in selected:
        if random.random() < augmentation_prob:
            signal = aug(signal)
    signal = normalize_signal(signal)
    return {'signal': signal, 'label': label}

def load_random_sample():
    """Load a random sample and its label from X_radio.npy and y_radio.npy."""
    X_radio = np.load('X_radio.npy')  # Shape: (61, 82, 63)
    y_radio = np.load('y_radio.npy')  # Shape: (61,)
    
    idx = random.randint(0, X_radio.shape[0] - 1)
    signal = X_radio[idx]  # Shape: (82, 63)
    label = int(y_radio[idx])  # Integer label (0 to 5)
    return signal, label

def plot_trajectory(signal, label, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(signal[:, 0], label='Wrist X')
    plt.plot(signal[:, 1], label='Wrist Y')
    plt.plot(signal[:, 2], label='Wrist Z')
    plt.title(f'Trajectory for {GESTURE_LABELS[label]}')
    plt.xlabel('Frame')
    plt.ylabel('Coordinate')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def main():
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
        plot_trajectory(result['signal'], result['label'], f'augmented_{i+1}_{GESTURE_LABELS[label]}.png')
    x_aug = np.array(augmented_signals)
    y_aug = np.array(augmented_labels)
    np.save("demo_X_augmented.npy", x_aug)
    np.save("demo_y_augmented.npy", y_aug)
    print("Demo saved: 'demo_X_augmented.npy', 'demo_y_augmented.npy'")


if __name__ == "__main__":
    main()