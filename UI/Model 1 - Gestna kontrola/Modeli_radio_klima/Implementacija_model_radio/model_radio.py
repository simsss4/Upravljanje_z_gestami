import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# ====================
# LOAD & NORMALIZE DATA
# ====================
X_train = np.load('X_radio_train.npy')
y_train = np.load('y_radio_train.npy')
X_val = np.load('X_radio_val.npy')
y_val = np.load('y_radio_val.npy')
X_test = np.load('X_radio_test.npy')
y_test = np.load('y_radio_test.npy')

# Normalize (mean and std from training set)
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# ====================
# MODEL ARCHITECTURE
# ====================
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

# Two LSTM layers (first returns sequences)
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.3))

# Fully connected layer
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(len(np.unique(y_train)), activation='softmax'))

# Compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ====================
# CALLBACKS
# ====================
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# ====================
# CLASS WEIGHTING
# ====================
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# ====================
# TRAINING
# ====================
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=8,
                    validation_data=(X_val, y_val),
                    class_weight=class_weights,
                    callbacks=[early_stop, reduce_lr])

# ====================
# EVALUATION
# ====================
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Confusion matrix and detailed performance
y_pred = np.argmax(model.predict(X_test), axis=1)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
model.save('radio_model.keras')
