import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt

# Nalaganje podatkov
X_mirror = np.load('X_mirrors.npy')  # Oblika: (n, frames, 63)
y_mirror = np.load('y_mirrors.npy')  # Oblika: (n,)

# Trunciranje zaporedij (če je potrebno)
max_frames = 80
X_mirror = X_mirror[:, :max_frames, :]

# Augmentacija
def augment_landmarks(X, noise_level=0.01, scale_factor=0.1, shift=0.05):
    noise = np.random.normal(0, noise_level, X.shape)
    scale = np.random.uniform(1 - scale_factor, 1 + scale_factor, X.shape)
    shift_val = np.random.uniform(-shift, shift, X.shape)
    return X * scale + noise + shift_val

X_aug1 = augment_landmarks(X_mirror)
X_aug2 = augment_landmarks(X_mirror)
X_mirror = np.concatenate([X_mirror, X_aug1, X_aug2], axis=0)
y_mirror = np.concatenate([y_mirror, y_mirror, y_mirror], axis=0)

# Normalizacija
mean = X_mirror.mean()
std = X_mirror.std()
X_mirror = (X_mirror - mean) / std

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_mirror), y=y_mirror)
class_weights = dict(enumerate(class_weights))

# Imena razredov
class_names = ['close_rm', 'down_rm', 'left_rm', 'open_rm', 'right_rm', 'up_rm']

# Model
def create_mirror_model():
    model = models.Sequential([
        layers.InputLayer(shape=(max_frames, 63)),
        layers.Masking(mask_value=0.0),
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(16)),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(6, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# K-fold validacija
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
test_accuracies = []
confusion_matrices = []

for train_idx, val_idx in kf.split(X_mirror):
    print(f"\nFold {fold}")
    X_train, X_val = X_mirror[train_idx], X_mirror[val_idx]
    y_train, y_val = y_mirror[train_idx], y_mirror[val_idx]

    model = create_mirror_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=8,
                        class_weight=class_weights, callbacks=[early_stop, reduce_lr], verbose=2)

    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Fold {fold} accuracy: {test_acc * 100:.2f}%")
    test_accuracies.append(test_acc)

    y_pred = np.argmax(model.predict(X_val), axis=-1)
    cm = confusion_matrix(y_val, y_pred, labels=list(range(6)))
    confusion_matrices.append(cm)

    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))

    fold += 1

# Povzetek rezultatov
print("\nPovprečna natančnost: {:.2f}% ± {:.2f}%".format(np.mean(test_accuracies) * 100, np.std(test_accuracies) * 100))
print("Povprečna konfuzijska matrika:")
print(np.mean(confusion_matrices, axis=0).astype(int))

# Končno učenje na vseh podatkih
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_mirror, y_mirror, test_size=0.2, stratify=y_mirror, random_state=42
)

model = create_mirror_model()
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

model.fit(X_train_final, y_train_final, validation_data=(X_val_final, y_val_final),
          epochs=50, batch_size=8, class_weight=class_weights,
          callbacks=[early_stop, reduce_lr], verbose=2)

# Evalvacija
val_loss, val_acc = model.evaluate(X_val_final, y_val_final, verbose=2)
print(f"Končna validacijska natančnost: {val_acc * 100:.2f}%")
y_pred_final = np.argmax(model.predict(X_val_final), axis=-1)
print("Konfuzijska matrika:")
print(confusion_matrix(y_val_final, y_pred_final))
print("Poročilo klasifikacije:")
print(classification_report(y_val_final, y_pred_final, target_names=class_names, zero_division=0))

# Shrani model
model.save('mirror_control_model.keras')

