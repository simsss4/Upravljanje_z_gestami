import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Naloži podatke
X = np.load('X_climate.npy')  # vhodni podatki
y = np.load('y_climate.npy')  # oznake

# Preveri oblike podatkov
print("X shape:", X.shape)
print("y shape:", y.shape)

# Prikaži število primerkov na razred
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))


# Razdeli podatke na učne, validacijske in testne množice
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) 

# Preveri oblike deljenih podatkov
print("Train shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_val.shape, y_val.shape)
print("Test shape:", X_test.shape, y_test.shape)

# Ustvari LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(np.unique(y)), activation='softmax'))

# Kompilacija modela
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Usposabljanje modela
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Oceni model na testnih podatkih
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Shrani model
model.save('gesture_recognition_model.keras')


# Load your model
model = load_model('gesture_recognition_model.keras')

# Print summary of the architecture
model.summary()




