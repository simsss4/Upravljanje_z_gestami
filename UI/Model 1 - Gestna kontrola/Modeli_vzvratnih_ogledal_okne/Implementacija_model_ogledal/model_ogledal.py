import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt

# Load data
x_mirrors = np.load('x_mirrors.npy')  # Shape: (40, 698, 63)
y_mirrors = np.load('y_mirrors.npy')  # Shape: (40,)

# Analyze sequence length
non_zero_frames = np.sum(np.any(x_mirrors != 0, axis=2), axis=1)
print("Average non-zero frames:", non_zero_frames.mean())
print("Max non-zero frames:", non_zero_frames.max())

# Truncate sequences
max_frames = 80
x_mirrors = x_mirrors[:, :max_frames, :]

# Data augmentation
def augment_landmarks(x, noise_level=0.01, scale_factor=0.1, shift=0.05):
    noise = np.random.normal(0, noise_level, x.shape)
    scale = np.random.uniform(1 - scale_factor, 1 + scale_factor, x.shape)
    shift_val = np.random.uniform(-shift, shift, x.shape)
    return x * scale + noise + shift_val

# Augment data (triple the dataset)
x_mirrors_aug1 = augment_landmarks(x_mirrors)
x_mirrors_aug2 = augment_landmarks(x_mirrors)
x_mirrors = np.concatenate([x_mirrors, x_mirrors_aug1, x_mirrors_aug2], axis=0)
y_mirrors = np.concatenate([y_mirrors, y_mirrors, y_mirrors], axis=0)
print("Augmented x_mirrors shape:", x_mirrors.shape)  # Expected: (120, 80, 63)

# Normalize using mean and std
mean = x_mirrors.mean()
std = x_mirrors.std()
x_mirrors = (x_mirrors - mean) / std

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_mirrors), y=y_mirrors)
class_weights = dict(enumerate(class_weights))

# Visualize gesture trajectories (multi-dimensional)
def plot_gestures(X, y):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    for i in range(6):
        idx = np.where(y == i)[0][0]
        ax1.plot(X[idx, :, 0], label=f'Class {i}: {["close_rm", "down_rm", "left_rm", "open_rm", "right_rm", "up_rm"][i]}')
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
    plt.show()

plot_gestures(x_mirrors, y_mirrors)

# Model architecture (simplified LSTM)
def create_mirrors_model():
    model = models.Sequential([
        layers.InputLayer(shape=(max_frames, 63)),
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(layers.LSTM(12)),
        layers.Dropout(0.4),
        layers.Dense(8, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(6, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
test_accuracies = []
confusion_matrices = []

for train_idx, val_idx in kf.split(x_mirrors):
    print(f"\nFold {fold}")
    x_train, x_val = x_mirrors[train_idx], x_mirrors[val_idx]
    y_train, y_val = y_mirrors[train_idx], y_mirrors[val_idx]

    model = create_mirrors_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=8,
                       class_weight=class_weights, callbacks=[early_stop, reduce_lr], verbose=2)

    test_loss, test_acc = model.evaluate(x_val, y_val, verbose=2)
    print(f"Fold {fold} accuracy: {test_acc * 100:.2f}%")
    test_accuracies.append(test_acc)

    y_pred = np.argmax(model.predict(x_val), axis=-1)
    print(f"Fold {fold} Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2, 3])
    print(cm)
    confusion_matrices.append(cm)  # Fixed: Append cm to list
    print(f"Fold {fold} Classification Report:")
    print(classification_report(y_val, y_pred, labels=[0, 1, 2, 3, 4, 5],
                               target_names=['close_rm', 'down_rm', 'left_rm', 'open_rm', 'right_rm', 'up_rm'],
                               zero_division=0))
    fold += 1

print("\nCross-Validation Results:")
print(f"Average test accuracy: {np.mean(test_accuracies) * 100:.2f}% ± {np.std(test_accuracies) * 100:.2f}%")
print("Average confusion matrix:")
print(np.mean(confusion_matrices, axis=0))

# Split for final training (80% train, 20% validation)
x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(
    x_mirrors, y_mirrors, test_size=0.2, stratify=y_mirrors, random_state=42
)

# Train final model with validation
model = create_mirrors_model()
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = model.fit(x_train_final, y_train_final, validation_data=(x_val_final, y_val_final),
                   epochs=50, batch_size=8, class_weight=class_weights,
                   callbacks=[early_stop, reduce_lr], verbose=2)

# Evaluate final model
val_loss, val_acc = model.evaluate(x_val_final, y_val_final, verbose=2)
print(f"Final validation accuracy: {val_acc * 100:.2f}%")

y_pred_final = np.argmax(model.predict(x_val_final), axis=-1)
print("Final Confusion Matrix:")
print(confusion_matrix(y_val_final, y_pred_final, labels=[0, 1, 2, 3, 4, 5]))
print("Final Classification Report:")
print(classification_report(y_val_final, y_pred_final, labels=[0, 1, 2, 3, 4, 5],
                           target_names=['close_rm', 'down_rm', 'left_rm', 'open_rm', 'right_rm', 'up_rm'],
                           zero_division=0))

# Save final model
model.save('mirrors_model.keras')
