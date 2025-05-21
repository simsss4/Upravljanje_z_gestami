import numpy as np
from pathlib import Path

# Define labels for validation
LABELS = {
    0: 'close_back_left_window',
    1: 'close_back_right_window',
    2: 'close_front_left_window',
    3: 'close_front_right_window',
    4: 'open_back_left_window',
    5: 'open_back_right_window',
    6: 'open_front_left_window',
    7: 'open_front_right_window'
}

def compute_finger_distances(X):
    X_new = np.zeros((X.shape[0], X.shape[1], 63 + 5 + 4))  # Add 5 fingertip-to-palm + 4 fingertip-to-fingertip distances
    for i in range(X.shape[0]):
        for f in range(X.shape[1]):
            landmarks = X[i, f].reshape(21, 3)
            if np.all(landmarks == 0):
                X_new[i, f, :63] = 0
                X_new[i, f, 63:] = 0
                continue
            X_new[i, f, :63] = X[i, f]
            wrist = landmarks[0]
            fingertip_indices = [4, 8, 12, 16, 20]
            for j, idx in enumerate(fingertip_indices):
                fingertip = landmarks[idx]
                distance = np.sqrt(np.sum((fingertip - wrist) ** 2))
                X_new[i, f, 63 + j] = distance
            fingertip_pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]
            for j, (idx1, idx2) in enumerate(fingertip_pairs):
                fingertip1, fingertip2 = landmarks[idx1], landmarks[idx2]
                distance = np.sqrt(np.sum((fingertip1 - fingertip2) ** 2))
                X_new[i, f, 63 + 5 + j] = distance
    return X_new

def main():
    # Load original data
    try:
        X_windows = np.load('x_windows.npy')
        y_windows = np.load('y_windows.npy')
        print(f"Loaded x_windows shape: {X_windows.shape}")
    except FileNotFoundError as e:
        print(f"Error: Could not load x_windows.npy or y_windows.npy: {e}")
        return

    # Validate labels
    unique_labels = np.unique(y_windows)
    valid_labels = set(LABELS.keys())
    if not set(unique_labels).issubset(valid_labels):
        print(f"Error: Found invalid labels in y_windows: {unique_labels}. Expected: {valid_labels}")
        return

    # Load augmented data
    aug_dir = Path("Augmentirane podatke")
    augmented_signals = []
    augmented_labels = []
    for i in range(10):
        signal_file = aug_dir / f"augmented_window_signal_{i}.npy"
        label_file = aug_dir / f"augmented_window_label_{i}.npy"
        try:
            signal = np.load(signal_file)
            label = np.load(label_file)
            print(f"Loaded {signal_file} (shape: {signal.shape}), label: {label_file} (value: {LABELS[int(label)]})")
            augmented_signals.append(signal)
            augmented_labels.append(label)
        except FileNotFoundError as e:
            print(f"Error: Could not load {signal_file} or {label_file}: {e}")
            return

    # Combine original and augmented data
    X_augmented = np.array(augmented_signals)  # Shape: (10, 152, 63)
    print(f"Augmented data shape: {X_augmented.shape}")
    X_combined = np.concatenate([X_windows, X_augmented], axis=0)  # Shape: (n_samples + 10, n_frames, 63)
    print(f"Combined X shape: {X_combined.shape}")

    # Truncate to 80 frames
    max_frames = 80
    X_combined = X_combined[:, :max_frames, :]  # Shape: (n_samples + 10, 80, 63)
    print(f"X_combined after truncation shape: {X_combined.shape}")

    # Compute finger distances
    X_combined = compute_finger_distances(X_combined)  # Shape: (n_samples + 10, 80, 72)
    print(f"X_combined after finger distances shape: {X_combined.shape}")

    # Compute mean and standard deviation
    MEAN = X_combined.mean()
    STD = X_combined.std()
    print(f"Windows Mean: {MEAN}")
    print(f"Windows Std: {STD}")

    # Save normalization parameters
    with open('windows_normalization_params.txt', 'w') as f:
        f.write(f"Mean: {MEAN}\n")
        f.write(f"Std: {STD}\n")
    print("Saved normalization parameters to windows_normalization_params.txt")

if __name__ == "__main__":
    main()
