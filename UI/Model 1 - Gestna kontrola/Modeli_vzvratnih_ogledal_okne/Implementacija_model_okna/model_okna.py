import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Nalaganje podatkov
X_windows = np.load('X_windows.npy')  # Oblika: (n, frames, 63)
y_windows = np.load('y_windows.npy')  # Oblika: (n,)

# Trunciranje zaporedij
max_frames = 80
X_windows = X_windows[:, :max_frames, :]

# Augmentacija
def augment_landmarks(X, noise_level=0.02, scale_factor=0.1, shift=0.1):
    noise = np.random.normal(0, noise_level, X.shape)
    scale = np.random.uniform(1 - scale_factor, 1 + scale_factor, X.shape)
    shift_val = np.random.uniform(-shift, shift, X.shape)
    return X * scale + noise + shift_val

X_aug1 = augment_landmarks(X_windows)
X_aug2 = augment_landmarks(X_windows)
X_windows = np.concatenate([X_windows, X_aug1, X_aug2], axis=0)
y_windows = np.concatenate([y_windows, y_windows, y_windows], axis=0)

# Normalizacija
mean = X_windows.mean()
std = X_windows.std()
X_windows = (X_windows - mean) / std

# Class weights with emphasis on open and close gestures
class_weights = compute_class_weight('balanced', classes=np.unique(y_windows), y=y_windows)
class_weights = dict(enumerate(class_weights))
for i in range(8):
    class_weights[i] *= 1.2  # Emphasize all classes slightly

# Imena razredov
class_names = [
    'close_back_left_window', 'close_back_right_window', 'close_front_left_window', 'close_front_right_window',
    'open_back_left_window', 'open_back_right_window', 'open_front_left_window', 'open_front_right_window'
]

# Model
def create_window_model():
    model = models.Sequential([
        layers.InputLayer(shape=(max_frames, 63)),
        layers.TimeDistributed(layers.Dense(32, activation='relu')),
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(16, return_sequences=True)),  # Fixed: Added return_sequences=True
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(8, activation='softmax')  # 8 razredov za okna
    ])
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(0.001, decay_steps=50 * 18)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Funkcija za vizualizacijo
def plot_results(history, y_true, y_pred, class_names, fold=None):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix {"(Fold " + str(fold) + ")" if fold else ""}')
    plt.tight_layout()
    #plt.savefig(f'fold_{fold}_results.png' if fold else 'final_results.png')
    plt.close()

# K-fold validacija
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
test_accuracies = []
confusion_matrices = []

for train_idx, val_idx in kf.split(X_windows):
    print(f"\nFold {fold}")
    X_train, X_val = X_windows[train_idx], X_windows[val_idx]
    y_train, y_val = y_windows[train_idx], y_windows[val_idx]

    model = create_window_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16,
                        class_weight=class_weights, callbacks=[early_stop], verbose=2)

    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Fold {fold} accuracy: {test_acc * 100:.2f}%")
    test_accuracies.append(test_acc)

    y_pred = np.argmax(model.predict(X_val), axis=-1)
    cm = confusion_matrix(y_val, y_pred, labels=list(range(8)))
    confusion_matrices.append(cm)

    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))
    plot_results(history, y_val, y_pred, class_names, fold)

    fold += 1

# Povzetek rezultatov
print("\nPovprečna natančnost: {:.2f}% ± {:.2f}%".format(np.mean(test_accuracies) * 100, np.std(test_accuracies) * 100))
print("Povprečna konfuzijska matrika:")
print(np.mean(confusion_matrices, axis=0).astype(int))

# Končno učenje na vseh podatkih
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_windows, y_windows, test_size=0.2, stratify=y_windows, random_state=42
)

model = create_window_model()
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history = model.fit(X_train_final, y_train_final, validation_data=(X_val_final, y_val_final),
                    epochs=50, batch_size=16, class_weight=class_weights,
                    callbacks=[early_stop], verbose=2)

# Evalvacija
val_loss, val_acc = model.evaluate(X_val_final, y_val_final, verbose=2)
print(f"Končna validacijska natančnost: {val_acc * 100:.2f}%")
y_pred_final = np.argmax(model.predict(X_val_final), axis=-1)
print("Konfuzijska matrika:")
print(confusion_matrix(y_val_final, y_pred_final))
print("Poročilo klasifikacije:")
print(classification_report(y_val_final, y_pred_final, target_names=class_names, zero_division=0))
plot_results(history, y_val_final, y_pred_final, class_names)

# Shrani model
model.save('window_control_model.keras')