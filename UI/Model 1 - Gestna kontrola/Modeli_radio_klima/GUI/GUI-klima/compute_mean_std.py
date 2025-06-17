import numpy as np

# Load data
X_climate = np.load('X_climate.npy')  # Shape: (samples, 82, 63)
X_climate_augmented = np.load('X_climate_augmented.npy')  # Shape: (samples, 82, 63)

# Combine original and augmented data
X_combined = np.concatenate([X_climate, X_climate_augmented], axis=0)
print(f"Combined X shape: {X_combined.shape}")

# Truncate sequences
max_frames = 80
X_combined = X_combined[:, :max_frames, :]  # Shape: (n_samples, 80, 63)

# Preprocess to include finger distances
def compute_finger_distances(X):
    # X shape: (samples, frames, 63), where 63 = 21 landmarks × 3 (x, y, z)
    # MediaPipe landmark indices:
    # 0: wrist, 4: thumb tip, 8: index tip, 12: middle tip, 16: ring tip, 20: pinky tip
    X_new = np.zeros((X.shape[0], X.shape[1], 63 + 5 + 4))  # Add 5 fingertip-to-palm + 4 fingertip-to-fingertip distances
    for i in range(X.shape[0]):  # For each sample
        for f in range(X.shape[1]):  # For each frame
            # Extract landmarks for this frame
            landmarks = X[i, f].reshape(21, 3)  # Shape: (21, 3)
            if np.all(landmarks == 0):  # Skip padded frames
                X_new[i, f, :63] = 0
                X_new[i, f, 63:] = 0
                continue

            # Copy raw landmarks
            X_new[i, f, :63] = X[i, f]

            # Compute fingertip-to-palm distances (wrist as palm proxy)
            wrist = landmarks[0]  # Wrist landmark (x, y, z)
            fingertip_indices = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
            for j, idx in enumerate(fingertip_indices):
                fingertip = landmarks[idx]
                distance = np.sqrt(np.sum((fingertip - wrist) ** 2))
                X_new[i, f, 63 + j] = distance

            # Compute fingertip-to-fingertip distances (adjacent fingers)
            fingertip_pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]  # Thumb-index, index-middle, middle-ring, ring-pinky
            for j, (idx1, idx2) in enumerate(fingertip_pairs):
                fingertip1, fingertip2 = landmarks[idx1], landmarks[idx2]
                distance = np.sqrt(np.sum((fingertip1 - fingertip2) ** 2))
                X_new[i, f, 63 + 5 + j] = distance

    return X_new

# Compute new features
X_combined = compute_finger_distances(X_combined)  # Shape: (n_samples, 80, 72)

# Compute mean and standard deviation
MEAN = X_combined.mean()
STD = X_combined.std()

# Print results
print(f"Climate Mean: {MEAN}")
print(f"Climate Std: {STD}")

# Save results to a file for reference
with open('climate_normalization_params.txt', 'w') as f:
    f.write(f"Mean: {MEAN}\n")
    f.write(f"Std: {STD}\n")