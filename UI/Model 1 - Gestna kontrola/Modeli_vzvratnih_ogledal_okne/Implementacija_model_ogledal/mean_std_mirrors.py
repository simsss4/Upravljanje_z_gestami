import numpy as np
from pathlib import Path

# Define labels for validation
LABELS = {
    0: 'close_rm',
    1: 'down_rm',
    2: 'left_rm',
    3: 'open_rm',
    4: 'right_rm',
    5: 'up_rm'
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
        X_mirrors = np.load('X_mirrors.npy')
        print(f"Loaded X_mirrors shape: {X_mirrors.shape}")
    except FileNotFoundError as e:
        print(f"Error: Could not load X_mirrors.npy: {e}")
        return

    # Load augmented data
    aug_dir = Path("Augmentirane podatke")
    augmented_signals = []
    augmented_labels = []
    for i in range(10):
        signal_file = aug_dir / f"augmented_mirror_signal_{i}.npy"
        label_file = aug_dir / f"augmented_mirror_label_{i}.npy"
        try:
            signal = np.load(signal_file)
            label = np.load(label_file)
            print(f"Loaded {signal_file} (shape: {signal.shape}), label: {label_file} (value: {LABELS[int(label)]})")
            augmented_signals.append(signal)
            augmented_labels.append(label)
        except FileNotFoundError as e:
            print(f"Error: Could not load {signal_file} or {label_file}: {e}")
            return

    # Validate labels
    unique_labels = np.unique(augmented_labels)
    valid_labels = set(LABELS.keys())
    if not set(unique_labels).issubset(valid_labels):
        print(f"Error: Found invalid labels in augmented data: {unique_labels}. Expected: {valid_labels}")
        return

    # Truncate to 80 frames before combining
    max_frames = 80
    X_mirrors = X_mirrors[:, :max_frames, :]  # Shape: (180, 80, 63)
    print(f"X_mirrors after truncation shape: {X_mirrors.shape}")
    X_augmented = np.array(augmented_signals)  # Shape: (10, 152, 63)
    X_augmented = X_augmented[:, :max_frames, :]  # Shape: (10, 80, 63)
    print(f"Augmented data shape after truncation: {X_augmented.shape}")

    # Combine original and augmented data
    X_combined = np.concatenate([X_mirrors, X_augmented], axis=0)  # Shape: (190, 80, 63)
    print(f"Combined X shape: {X_combined.shape}")

    # Compute finger distances
    X_combined = compute_finger_distances(X_combined)  # Shape: (190, 80, 72)
    print(f"X_combined after finger distances shape: {X_combined.shape}")

    # Compute mean and standard deviation
    MEAN = X_combined.mean()
    STD = X_combined.std()
    print(f"Mirrors Mean: {MEAN}")
    print(f"Mirrors Std: {STD}")

    # Save normalization parameters
    with open('mirrors_normalization_params.txt', 'w') as f:
        f.write(f"Mean: {MEAN}\n")
        f.write(f"Std: {STD}\n")
    print("Saved normalization parameters to mirrors_normalization_params.txt")

if __name__ == "__main__":
    main()