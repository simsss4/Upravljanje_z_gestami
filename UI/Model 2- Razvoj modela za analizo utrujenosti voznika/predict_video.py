import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time

# --- Nastavitve ---
VIDEO_FILE = "test3.mov"  # <- ime datoteke videa (mora biti v isti mapi)
MODEL_FILE = "model_drowsy_detector.h5"
IMG_SIZE = 128
SECS_THRESHOLD = 3 # po koliko sekundah utrujenosti prikaže opozorilo
FPS = 30  # privzeta hitrost snemanja

# --- Poti ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model", MODEL_FILE)
VIDEO_PATH = os.path.join(SCRIPT_DIR, VIDEO_FILE)

# --- Naloži model ---
model = load_model(MODEL_PATH)

# --- Odpri video ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Napaka pri odpiranju videa.")
    exit()

# --- Pomnilnik zadnjih napovedi ---
history = deque(maxlen=SECS_THRESHOLD * FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pretvori v grayscale in pripravi dimenzije
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    input_img = resized.astype('float32') / 255.0
    input_img = np.expand_dims(input_img, axis=(0, -1))  # (1, 128, 128, 1)

    # Napoved
    prediction = model.predict(input_img)[0][0]
    is_drowsy = prediction > 0.9
    history.append(is_drowsy)

    # Prikaz slike
    label = "Utrujen" if is_drowsy else "Budnost OK"
    color = (0, 0, 255) if is_drowsy else (0, 255, 0)
    cv2.putText(frame, f"{label} ({prediction:.2f})", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Če je bil voznik "utrujen" vsaj 2 sekundi zapored
    if sum(history) >= 0.8 * len(history):
        cv2.putText(frame, "OPOZORILO: VOZNIK ZASPI!", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Prepoznavanje utrujenosti", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
