import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Naloži podatke
x = np.load('x_mirrors.npy')  # vhodni podatki
y = np.load('y_mirrors.npy')  # oznake

# Preveri oblike podatkov
print("x shape:", x.shape)
print("y shape:", y.shape)

# Prikaži število primerkov na razred
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))


# Razdeli podatke na učne, validacijske in testne množice
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42) 

# Preveri oblike deljenih podatkov
print("Train shape:", x_train.shape, y_train.shape)
print("Validation shape:", x_val.shape, y_val.shape)
print("Test shape:", x_test.shape, y_test.shape)

np.save('x_mirrors_train.npy', x_train)
np.save('y_mirrors_train.npy', y_train)
np.save('x_mirrors_val.npy', x_val)
np.save('y_mirrors_val.npy', y_val)
np.save('x_mirrors_test.npy', x_test)
np.save('y_mirrors_test.npy', y_test)





