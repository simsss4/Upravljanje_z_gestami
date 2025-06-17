import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Paths
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data')
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# 1) Load data
X_train = np.load(os.path.join(train_dir, 'X_train.npy'))
y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
X_val = np.load(os.path.join(val_dir, 'X_val.npy'))
y_val = np.load(os.path.join(val_dir, 'y_val.npy'))
X_test = np.load(os.path.join(test_dir, 'X_test.npy'))
y_test = np.load(os.path.join(test_dir, 'y_test.npy'))

# 2) Preprocess: normalize and add channel dimension
X_train = X_train[..., np.newaxis] / 255.0
X_val = X_val[..., np.newaxis] / 255.0
X_test = X_test[..., np.newaxis] / 255.0

# 3) Build simple CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# 4) Train
checkpoint = ModelCheckpoint('model_drowsy_detector.h5', save_best_only=True)
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint]
)

# 5) Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.3f}')

# 6) Plot training history
plt.figure(figsize=(8,3))
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
