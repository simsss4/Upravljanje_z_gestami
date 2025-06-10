import numpy as np
import scipy.signal
import os
import random

LABELS = {
    0: 'close_rm',
    1: 'down_rm',
    2: 'left_rm',
    3: 'open_rm',
    4: 'right_rm',
    5: 'up_rm',
}

def normalize_signal(signal):
    signal = signal.copy()
    non_zero_mask = np.any(signal != 0, axis=1)
    if np.any(non_zero_mask):
        non_zero_signal = signal[non_zero_mask]
        min_val = non_zero_signal.min()
        max_val = non_zero_signal.max()
        if max_val > min_val:
            signal[non_zero_mask] = (non_zero_signal - min_val) / (max_val - min_val)
    return signal

def add_noise(signal, noise_level=0.01):
    noise = np.random.normal(0, noise_level, signal.shape)
    signal = signal.copy()
    non_zero_mask = np.any(signal != 0, axis=1)
    signal[non_zero_mask] += noise[non_zero_mask]
    return signal

def frame_shift(signal, max_shift=10):
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

def smoothing(signal, window_size=5):
    window = scipy.signal.windows.hann(window_size)
    signal = signal.copy()
    non_zero_mask = np.any(signal != 0, axis=1)
    frames = np.where(non_zero_mask)[0]
    if len(frames) > window_size:
        for i in range(signal.shape[1]):
            signal[frames, i] = np.convolve(signal[frames, i], window / window.sum(), mode='same')
    return signal

def time_warp(signal, max_stretch=0.2):
    n_frames = signal.shape[0]
    stretch_factor = 1.0 + np.random.uniform(-max_stretch, max_stretch)
    new_length = int(n_frames * stretch_factor)
    new_length = max(10, min(new_length, int(n_frames * 1.5)))

    old_indices = np.linspace(0, n_frames - 1, n_frames)
    new_indices = np.linspace(0, n_frames - 1, new_length)
    warped_signal = np.zeros((new_length, signal.shape[1]))

    for i in range(signal.shape[1]):
        warped_signal[:, i] = np.interp(new_indices, old_indices, signal[:, i])

    return warped_signal

def frame_dropping(signal, drop_rate=0.1):
    signal = signal.copy()
    num_frames = signal.shape[0]
    non_zero_mask = np.any(signal != 0, axis=1)
    frames = np.where(non_zero_mask)[0]
    if len(frames) == 0:
        return signal

    num_drop = int(len(frames) * drop_rate)
    drop_indices = np.random.choice(frames, size=num_drop, replace=False)
    signal[drop_indices] = 0

    for i in range(signal.shape[1]):
        non_zero_frames = np.where(np.any(signal != 0, axis=1))[0]
        if len(non_zero_frames) > 1:
            interpolated = np.interp(np.arange(num_frames), non_zero_frames, signal[non_zero_frames, i])
            signal[:, i] = interpolated
    return signal

def pad_or_crop(signal, target_length=152):
    current_length = signal.shape[0]
    if current_length > target_length:
        return signal[:target_length]
    elif current_length < target_length:
        return np.pad(signal, ((0, target_length - current_length), (0, 0)), mode='constant')
    return signal

def transform(signal, label, prob_noise=0.7, prob_shift=0.7, prob_smooth=0.7, prob_warp=0.7, prob_drop=0.7):
    signal = signal.copy()
    signal = normalize_signal(signal)

    if random.random() < prob_noise:
        signal = add_noise(signal, noise_level=0.01)

    if random.random() < prob_shift:
        signal = frame_shift(signal, max_shift=10)

    if random.random() < prob_smooth:
        signal = smoothing(signal, window_size=5)

    if random.random() < prob_warp:
        signal = time_warp(signal, max_stretch=0.2)

    if random.random() < prob_drop:
        signal = frame_dropping(signal, drop_rate=0.1)

    signal = pad_or_crop(signal, target_length=152)
    signal = normalize_signal(signal)

    return {'signal': (signal, label)}

def main():
    output_dir = "Augmentirane podatke"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x_windows = np.load("x_mirrors.npy")
    y_windows = np.load("y_mirrors.npy")

    unique_labels = np.unique(y_windows)
    valid_labels = set(LABELS.keys())
    if not set(unique_labels).issubset(valid_labels):
        print(f"Error: Found invalid labels in y_mirrors: {unique_labels}. Expected: {valid_labels}")
        return

    np.random.seed(42)
    indices = np.random.choice(x_windows.shape[0], size=10, replace=True)

    for i, idx in enumerate(indices):
        signal = x_windows[idx]
        label = int(y_windows[idx])
        print(f"Processing sample {i+1}/10, index: {idx}, shape: {signal.shape}, label: {LABELS[label]}")

        result = transform(signal, label)
        augmented_signal, augmented_label = result['signal']

        output_signal_file = os.path.join(output_dir, f"augmented_mirror_signal_{i}.npy")
        output_label_file = os.path.join(output_dir, f"augmented_mirror_label_{i}.npy")

        np.save(output_signal_file, augmented_signal)
        np.save(output_label_file, augmented_label)
        print(f"Saved: {output_signal_file} (shape: {augmented_signal.shape}), label: {output_label_file} (value: {LABELS[augmented_label]})")

if __name__ == "__main__":
    main()
