import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Naloži podatke
X = np.load('X_radio.npy')  # vhodni podatki
y = np.load('y_radio.npy')  # oznake

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

np.save('X_radio_train.npy', X_train)
np.save('y_radio_train.npy', y_train)
np.save('X_radio_val.npy', X_val)
np.save('y_radio_val.npy', y_val)
np.save('X_radio_test.npy', X_test)
np.save('y_radio_test.npy', y_test)





