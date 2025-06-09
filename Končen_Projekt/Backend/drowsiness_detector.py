import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import sys
import tkinter as tk
from tkinter import filedialog

sys.stdout.reconfigure(encoding='utf-8')

# Nastavitve
MODEL_FILE = "model_drowsy_detector.h5"
IMG_SIZE = 128
SECS_THRESHOLD = 3
FPS = 30
DROWSY_THRESHOLD = 0.9
HISTORY_FRACTION = 0.8
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Poti
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "Models", MODEL_FILE)
DATA_DRIVER_DIR = SCRIPT_DIR

def select_video():
    """Odpre File Explorer za izbiro videa."""
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Izberi video",
        initialdir=DATA_DRIVER_DIR,
        filetypes=[("Video files", "*.mov *.mp4 *.avi")]
    )
    root.destroy()
    return video_path

def analyze_drowsiness(video_path):
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video ne obstaja: {video_path}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model ne obstaja: {MODEL_FILE}")

        # Naloži model
        model = load_model(MODEL_PATH)

        # Validacija oblike vnosa
        expected_shape = model.input_shape[1:3]
        if (IMG_SIZE, IMG_SIZE) != expected_shape:
            raise ValueError(f"Neskladje oblike vnosa: pričakovano {expected_shape}, dobljeno ({IMG_SIZE}, {IMG_SIZE})")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Ne morem odpreti videa: {video_path}")

        history = deque(maxlen=SECS_THRESHOLD * FPS)
        predictions = []
        frame_count = 0
        current_state = None
        last_state = None

        cv2.namedWindow("Prepoznavanje utrujenosti", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Prepoznavanje utrujenosti", WINDOW_WIDTH, WINDOW_HEIGHT)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            input_img = resized.astype('float32') / 255.0
            input_img = np.expand_dims(input_img, axis=(0, -1))

            prediction = model.predict(input_img, verbose=0)[0][0]
            is_drowsy = prediction > DROWSY_THRESHOLD
            history.append(is_drowsy)
            predictions.append(prediction)

            # Določi trenutno stanje
            if len(history) == history.maxlen and sum(history) >= HISTORY_FRACTION * len(history):
                current_state = "Utrujen"
                label_color = (0, 0, 255)  # Rdeča
            else:
                current_state = "Buden"
                label_color = (0, 255, 0)  # Zelena

            # Izpiši spremembo stanja v terminal
            if current_state != last_state:
                if current_state == "Utrujen":
                    print("Opozorilo: Voznik zaspi!", flush=True)
                else:
                    print("Voznik je buden", flush=True)
                last_state = current_state

            # Prikaz informacij na sličici
            cv2.putText(frame, f"Stanje: {current_state}", (30, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
            
            # Prikaz verjetnosti utrujenosti
            bar_width = 200
            bar_height = 20
            bar_x, bar_y = 30, 80
            drowsiness_level = min(prediction, 1.0)
            filled_width = int(bar_width * drowsiness_level)
            
            # Barva glede na stanje
            bar_color = label_color
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (200, 200, 200), -1)  # Ozadje
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                         bar_color, -1)  # Napolnjen del
            cv2.putText(frame, f"Verj. utrujenosti: {prediction:.2f}", 
                       (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Prepoznavanje utrujenosti", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        error_message = f"Napaka pri analizi utrujenosti: {str(e)}"
        print(error_message, flush=True)

if __name__ == "__main__":
    video_path = select_video()
    if video_path:
        analyze_drowsiness(video_path)
    else:
        print("Video ni izbran. Prekinjam.", flush=True)