import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split


# Load data
X_radio = np.load('X_radio.npy')  # Shape: (61, 82, 63)
y_radio = np.load('y_radio.npy')  # Shape: (61,)
X_augmented = np.load('X_augmented.npy')  # Shape: (samples, 82, 63)
y_augmented = np.load('y_augmented.npy')  # Shape: (samples,)

# Combine original and new augmented data
X_radio = np.concatenate([X_radio, X_augmented], axis=0)
y_radio = np.concatenate([y_radio, y_augmented], axis=0)
print(f"Combined X_radio shape: {X_radio.shape}")
print(f"Combined y_radio shape: {y_radio.shape}")

# Analyze sequence length
non_zero_frames = np.sum(np.any(X_radio != 0, axis=2), axis=1)
print("Average non-zero frames:", non_zero_frames.mean())
print("Max non-zero frames:", non_zero_frames.max())

# Truncate sequences
max_frames = 80
X_radio = X_radio[:, :max_frames, :]  # Shape: (n_samples, 80, 63)

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
X_radio = compute_finger_distances(X_radio)  # Shape: (n_samples, 80, 72)

# Data augmentation (quadruple the dataset)
def augment_landmarks(X, noise_level=0.01, scale_factor=0.1, shift=0.05):
    noise = np.random.normal(0, noise_level, X.shape)
    scale = np.random.uniform(1 - scale_factor, 1 + scale_factor, X.shape)
    shift_val = np.random.uniform(-shift, shift, X.shape)
    return X * scale + noise + shift_val


# Normalize using mean and std
mean = X_radio.mean()
std = X_radio.std()
X_radio = (X_radio - mean) / std

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_radio), y=y_radio)
class_weights = dict(enumerate(class_weights))

# Visualize gesture trajectories (using raw wrist coordinates)
def plot_gestures(X, y):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    for i in range(6):
        idx = np.where(y == i)[0][0]
        ax1.plot(X[idx, :, 0], label=f'Class {i}: {["next_station", "previous_station", "turn_off_radio", "turn_on_radio", "volume_down", "volume_up"][i]}')
        ax2.plot(X[idx, :, 1], label=f'Class {i}')
        ax3.plot(X[idx, :, 2], label=f'Class {i}')
    ax1.set_title('Wrist X Coordinate')
    ax2.set_title('Wrist Y Coordinate')
    ax3.set_title('Wrist Z Coordinate')
    ax3.set_xlabel('Frame')
    for ax in [ax1, ax2, ax3]:
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('gesture_trajectories.png')

plot_gestures(X_radio, y_radio)

# Visualize fingertip-to-palm distance for thumb (feature 63)
def plot_finger_distances(X, y):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(6):
        idx = np.where(y == i)[0][0]
        ax.plot(X[idx, :, 63], label=f'Class {i}: {["next_station", "previous_station", "turn_off_radio", "turn_on_radio", "volume_down", "volume_up"][i]}')
    ax.set_title('Thumb Tip to Wrist Distance')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Distance')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('finger_distances.png')

plot_finger_distances(X_radio, y_radio)

# Improved model architecture
def create_radio_model():
    model = models.Sequential([
        layers.Input(shape=(max_frames, 72)),  # Input with added features
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(16)),
        layers.Dropout(0.3),  # Lower dropout
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(6, activation='softmax')  # 6 gesture classes
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


fold = 1
test_accuracies = []
confusion_matrices = []


# K-fold cross-validation
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X_radio, y_radio):  # <- note: y_radio is added here
    print(f"\nFold {fold}")
    X_train, X_val = X_radio[train_idx], X_radio[val_idx]
    y_train, y_val = y_radio[train_idx], y_radio[val_idx]

    model = create_radio_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=8,
                       class_weight=class_weights, callbacks=[early_stop, reduce_lr], verbose=2)

    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)
    print(f"Fold {fold} accuracy: {test_acc * 100:.2f}%")
    test_accuracies.append(test_acc)

    y_pred = np.argmax(model.predict(X_val), axis=-1)
    print(f"Fold {fold} Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2, 3, 4, 5])
    print(cm)
    confusion_matrices.append(cm)
    print(f"Fold {fold} Classification Report:")
    print(classification_report(y_val, y_pred, labels=[0, 1, 2, 3, 4, 5],
                               target_names=['next_station', 'previous_station', 'turn_off_radio', 'turn_on_radio', 'volume_down', 'volume_up'],
                               zero_division=0))
    fold += 1

print("\nCross-Validation Results:")
print(f"Average test accuracy: {np.mean(test_accuracies) * 100:.2f}% ± {np.std(test_accuracies) * 100:.2f}%")
print("Average confusion matrix:")
print(np.mean(confusion_matrices, axis=0))

# Split for final training (80% train, 20% validation)
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_radio, y_radio, test_size=0.2, stratify=y_radio, random_state=42
)

# Train final model
model = create_radio_model()
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = model.fit(X_train_final, y_train_final, validation_data=(X_val_final, y_val_final),
                   epochs=50, batch_size=32, class_weight=class_weights,
                   callbacks=[early_stop, reduce_lr], verbose=2)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss med učenjem')
plt.xlabel('Epoka')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy med učenjem')
plt.xlabel('Epoka')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Evaluate final model
val_loss, val_acc = model.evaluate(X_val_final, y_val_final, verbose=2)
print(f"Final validation accuracy: {val_acc * 100:.2f}%")

y_pred_final = np.argmax(model.predict(X_val_final), axis=-1)
print("Final Confusion Matrix:")
print(confusion_matrix(y_val_final, y_pred_final, labels=[0, 1, 2, 3, 4, 5]))
print("Final Classification Report:")
print(classification_report(y_val_final, y_pred_final, labels=[0, 1, 2, 3, 4, 5],
                           target_names=['next_station', 'previous_station', 'turn_off_radio', 'turn_on_radio', 'volume_down', 'volume_up'],
                           zero_division=0))

cm_final = confusion_matrix(y_val_final, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues',
            xticklabels=['next', 'prev', 'off', 'on', 'vol-', 'vol+'],
            yticklabels=['next', 'prev', 'off', 'on', 'vol-', 'vol+'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Final Confusion Matrix (Radio Model)')
plt.tight_layout()
plt.show()

# Save final model
model.save('radio_model_augmented.keras')