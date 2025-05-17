import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split


# Load fully augmented dataset
X_climate = np.load('X_augmented.npy')  # Already includes augmentation
y_climate = np.load('y_augmented.npy')

print(f"Loaded X_climate shape: {X_climate.shape}")
print(f"Loaded y_climate shape: {y_climate.shape}")

# Analyze sequence length
non_zero_frames = np.sum(np.any(X_climate != 0, axis=2), axis=1)
print("Average non-zero frames:", non_zero_frames.mean())
print("Max non-zero frames:", non_zero_frames.max())

# Truncate sequences
max_frames = 80
X_climate = X_climate[:, :max_frames, :]  # Shape: (n_samples, 80, 63)

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
X_climate = compute_finger_distances(X_climate)  # Shape: (n_samples, 80, 72)

# Normalize using mean and std
mean = X_climate.mean(axis=(0,1), keepdims=True)
std = X_climate.std(axis=(0,1), keepdims=True)
X_climate = (X_climate - mean) / std

# Compute class weights for the entire dataset BEFORE K-fold
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_climate),
    y=y_climate
)
class_weights = dict(enumerate(class_weights_array))

# Visualize gesture trajectories (using raw wrist coordinates)
def plot_gestures(X, y):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    for i in range(4):
        idxs = np.where(y == i)[0]
        if len(idxs) == 0:
            continue  # Skip missing class
        idx = idxs[0]
        labels = ['climate_colder', 'climate_warmer', 'fan_stronger', 'fan_weaker']
        label = labels[i]
        ax1.plot(X[idx, :, 0], label=f'Class {i}: {label}')
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

plot_gestures(X_climate, y_climate)

# Visualize fingertip-to-palm distance for thumb (feature 63)
def plot_finger_distances(X, y):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(4):
        idx = np.where(y == i)[0][0]
        labels = ['climate_colder', 'climate_warmer', 'fan_stronger', 'fan_weaker']
        label = labels[i]
        ax.plot(X[idx, :, 63], label=f'Class {i}: {label}')  # feature 63 is thumb tip to wrist distance
    ax.set_title('Thumb Tip to Wrist Distance')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Distance')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('finger_distances.png')

plot_finger_distances(X_climate, y_climate)

# Improved model architecture
def create_climate_model():
    model = models.Sequential([
        layers.Input(shape=(max_frames, 72)),  # Input with added features
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(16)),
        layers.Dropout(0.3),  # Lower dropout
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(4, activation='softmax')  # 4 gesture classes
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

for train_idx, val_idx in skf.split(X_climate, y_climate):  
    print(f"\nFold {fold}")
    X_train, X_val = X_climate[train_idx], X_climate[val_idx]
    y_train, y_val = y_climate[train_idx], y_climate[val_idx]

    model = create_climate_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Use class_weights computed on the entire dataset here
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=8,
                       class_weight=class_weights, callbacks=[early_stop, reduce_lr], verbose=2)

    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)
    print(f"Fold {fold} accuracy: {test_acc * 100:.2f}%")
    test_accuracies.append(test_acc)

    y_pred = np.argmax(model.predict(X_val), axis=-1)
    print(f"Fold {fold} Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2, 3])
    print(cm)
    confusion_matrices.append(cm)
    print(f"Fold {fold} Classification Report:")
    print(classification_report(y_val, y_pred, labels=[0, 1, 2, 3],
                               target_names=['climate_colder', 'climate_warmer', 'fan_stronger', 'fan_weaker'],
                               zero_division=0))
    fold += 1

print("\nCross-Validation Results:")
print(f"Average test accuracy: {np.mean(test_accuracies) * 100:.2f}% ± {np.std(test_accuracies) * 100:.2f}%")
print("Average confusion matrix:")
print(np.mean(confusion_matrices, axis=0))

# Split for final training (80% train, 20% validation)
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_climate, y_climate, test_size=0.2, stratify=y_climate, random_state=42
)

# Compute class weights using FINAL training set for final training
class_weights_array_final = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_final),
    y=y_train_final
)
class_weights_final = dict(enumerate(class_weights_array_final))

# Train final model
model = create_climate_model()
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = model.fit(X_train_final, y_train_final, validation_data=(X_val_final, y_val_final),
                   epochs=50, batch_size=32, class_weight=class_weights_final,
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
print(confusion_matrix(y_val_final, y_pred_final, labels=[0, 1, 2, 3]))
print("Final Classification Report:")
print(classification_report(y_val_final, y_pred_final, labels=[0, 1, 2, 3],
                           target_names=['climate_colder', 'climate_warmer', 'fan_stronger', 'fan_weaker'],
                           zero_division=0))

cm_final = confusion_matrix(y_val_final, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues',
            xticklabels=['cold', 'warm', 'strong', 'weak'],
            yticklabels=['cold', 'warm', 'strong', 'weak'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Final Confusion Matrix (Climate Model)')
plt.tight_layout()
plt.show()

# Save final model
model.save('climate_model_augmented.keras')
