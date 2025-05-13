import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Load the data (X_climate.npy, y_climate.npy)
X_climate = np.load('X_climate.npy')
y_climate = np.load('y_climate.npy')

# Normalize the data (if needed)
X_climate = X_climate.astype('float32') / 255.0

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_climate, y_climate, test_size=0.2, random_state=42)

# Model architecture
def create_climate_model():
    model = models.Sequential([
        layers.InputLayer(input_shape=X_train.shape[1:]),
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='softmax')  # Adjust output neurons for 4 classes
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
model = create_climate_model()

# Adjust the batch size based on the available resources
batch_size = 16  # Feel free to adjust this based on your system's capabilities

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=batch_size, verbose=2)

# Test the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Predictions and classification report
y_pred = np.argmax(model.predict(X_test), axis=-1)
print(classification_report(y_test, y_pred))
