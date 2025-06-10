import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import queue

# Inicializacija
mp_hands = mp.solutions.hands # MediaPipe Hands za zaznavanje in sledenje rok
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) # Objekt Hands - zazna največ eno roko z minimalno zaupnostjo 0.7
mp_drawing = mp.solutions.drawing_utils # Orodja za risanje MediaPipe za vizualizacijo točk in povezavo rok

# Nalaganje predhodno treniranih modelov
mirrors_model = tf.keras.models.load_model('mirrors_model.keras')
windows_model = tf.keras.models.load_model('windows_model.keras')
radio_model = tf.keras.models.load_model('radio_model.keras')
climate_model = tf.keras.models.load_model('climate_model.keras')

# Povprečje in standardni odklon za normalizacijo podatkov za modele (iz skripte: mean_std_mirrors.py/mean_std_windows.py)
MIRRORS_MEAN = 0.32715722213633686  
MIRRORS_STD = 0.3050974859034138    
WINDOWS_MEAN = 0.3218953795343205  
WINDOWS_STD = 0.2855003991679068  
RADIO_MEAN = 0.1824056036195469
RADIO_STD = 0.2894906092020188
CLIMATE_MEAN = 0.20537318328833137
CLIMATE_STD = 0.2955649804045277

# Seznam imen razredov (z labele - izhodne vrednosti)
MIRRORS_CLASSES = ['close_rm', 'down_rm', 'left_rm', 'open_rm', 'right_rm', 'up_rm']
WINDOWS_CLASSES = ['close_back_left_window', 'close_back_right_window', 'close_front_left_window', 'close_front_right_window', 'open_back_left_window', 'open_back_right_window', 'open_front_left_window', 'open_front_right_window']
RADIO_CLASSES = ['next_station', 'previous_station', 'turn_off_radio', 'turn_on_radio', 'volume_down', 'volume_up']
CLIMATE_CLASSES = [ 'climate_colder', 'climate_warmer',  'fan_stronger', 'fan_weaker']
MAX_FRAMES = 80

def compute_finger_distances(X):
    # X shape: (frames, 63), where 63 = 21 landmarks × 3 (x, y, z)
    X_new = np.zeros((X.shape[0], 63 + 5 + 4))  # Add 5 fingertip-to-palm + 4 fingertip-to-fingertip distances
    for f in range(X.shape[0]):  # For each frame
        landmarks = X[f].reshape(21, 3)  # Shape: (21, 3)
        if np.all(landmarks == 0):  # Skip empty frames
            X_new[f, :63] = 0
            X_new[f, 63:] = 0
            continue

        # Copy raw landmarks
        X_new[f, :63] = X[f]

        # Compute fingertip-to-palm distances (wrist as palm proxy)
        wrist = landmarks[0]
        fingertip_indices = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
        for j, idx in enumerate(fingertip_indices):
            fingertip = landmarks[idx]
            distance = np.sqrt(np.sum((fingertip - wrist) ** 2))
            X_new[f, 63 + j] = distance

        # Compute fingertip-to-fingertip distances (adjacent fingers)
        fingertip_pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]  # Thumb-index, index-middle, middle-ring, ring-pinky
        for j, (idx1, idx2) in enumerate(fingertip_pairs):
            fingertip1, fingertip2 = landmarks[idx1], landmarks[idx2]
            distance = np.sqrt(np.sum((fingertip1 - fingertip2) ** 2))
            X_new[f, 63 + 5 + j] = distance

    return X_new

class GestureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition")
        self.cap = cv2.VideoCapture(0)
        self.recording = False
        self.frames = [] # Prazen seznam za shranjevanje koordinat točk rok med snemanjem
        self.handedness_labels = []  # Prazen seznam za shranjevanje oznak leve/desne roke za vsak okvir
        #self.frame_queue = queue.Queue()
        self.current_model = 'Mirrors'  # Default model

        # Konfiguracija modelov
        self.models = {
            'Mirrors': {
                'model': mirrors_model,
                'mean': MIRRORS_MEAN,
                'std': MIRRORS_STD,
                'classes': MIRRORS_CLASSES
            },
            'Windows': {
                'model': windows_model,
                'mean': WINDOWS_MEAN,
                'std': WINDOWS_STD,
                'classes': WINDOWS_CLASSES
            },
            'Radio': {
                'model': radio_model,
                'mean': RADIO_MEAN,
                'std': RADIO_STD,
                'classes': RADIO_CLASSES
            },
            'Climate': {
                'model': climate_model,
                'mean': CLIMATE_MEAN,
                'std': CLIMATE_STD,
                'classes': CLIMATE_CLASSES
            }
        }

        # GUI elemente
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.model_var = tk.StringVar(value='Mirrors')
        self.model_menu = tk.OptionMenu(root, self.model_var, 'Mirrors', 'Windows', 'Radio', 'Climate', command=self.switch_model)
        self.model_menu.pack(pady=5)

        self.label = tk.Label(root, text="Predicted Gesture: None", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn_capture = tk.Button(root, text="Capture Gesture", command=self.start_recording)
        self.btn_capture.pack(pady=5)

        self.btn_quit = tk.Button(root, text="Quit", command=self.quit)
        self.btn_quit.pack(pady=5)

        self.update_video()

    # Metoda za posodobitev trenutnega modela ob izbiri iz menija
    def switch_model(self, model_name):
        self.current_model = model_name
        self.label.config(text=f"Predicted Gesture: None (Using {model_name} model)")

    # Metoda za vklop/izklop snemanja gest ob kliku na gumb
    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.frames = []
            self.handedness_labels = [] 
            self.btn_capture.config(text="Stop Recording")
            self.label.config(text="Recording...")
        else:
            self.recording = False
            self.btn_capture.config(text="Capture Gesture")
            self.process_gesture()

    # Metoda za obdelavo okvirjev, normalizacijo in napoved geste
    def process_gesture(self):
        if len(self.frames) == 0:
            self.label.config(text="No frames captured")
            return

        X = np.zeros((MAX_FRAMES, 63))
        for i in range(min(len(self.frames), MAX_FRAMES)):
            X[i] = self.frames[i]
        if len(self.frames) < MAX_FRAMES:
            X[len(self.frames):] = 0

        # === Apply compute_finger_distances only for Radio and Climate ===
        if self.current_model in ['Radio', 'Climate']:
            X = compute_finger_distances(X)

        mean = self.models[self.current_model]['mean']
        std = self.models[self.current_model]['std']
        X = (X - mean) / std

        X = X[np.newaxis, ...]

        model = self.models[self.current_model]['model']
        pred = model.predict(X, verbose=0)
        pred_class = np.argmax(pred, axis=1)[0]
        pred_conf = pred[0, pred_class]
        gesture = self.models[self.current_model]['classes'][pred_class]

        if self.handedness_labels:
            left_count = self.handedness_labels.count('Left')
            right_count = self.handedness_labels.count('Right')
            hand = 'Left' if left_count >= right_count else 'Right'
        else:
            hand = 'Unknown'

        if self.current_model == 'Mirrors':
            display_text = f"{hand} mirror: {gesture} ({pred_conf:.2%})"
        elif self.current_model == 'Windows':
            display_text = f"Window: {gesture} ({pred_conf:.2%})"
        elif self.current_model == 'Radio':
            display_text = f"Radio: {gesture} ({pred_conf:.2%})"
        elif self.current_model == 'Climate':
            display_text = f"Climate: {gesture} ({pred_conf:.2%})"
        else:
            display_text = f"{self.current_model}: {gesture} ({pred_conf:.2%})"

        self.label.config(text=f"Predicted Gesture: {display_text}")

        # Save wrist trajectory plot (frame[0] is wrist x, y, z)
        plt.figure(figsize=(10, 5))
        plt.plot(X[0, :, 0], label='Wrist X')
        plt.plot(X[0, :, 1], label='Wrist Y')
        plt.plot(X[0, :, 2], label='Wrist Z')
        plt.title(f'Wrist Trajectory for {gesture} ({self.current_model}, {hand} hand)')
        plt.xlabel('Frame')
        plt.ylabel('Coordinate')
        plt.legend()
        plt.savefig(f'captured_gesture_trajectory_{self.current_model.lower()}_{hand.lower()}.png')
        plt.close()


    # Metoda za stalno posodabljanje platna z okvirji kamere in obdelavo rok
    def update_video(self):
        try:
            ret, frame = self.cap.read() # Prebere okvir iz spletne kamere; ret - uspeh, frame - podatke slike
            if ret: # Preveri, ali je bil okvir uspešno prebran
                frame = cv2.flip(frame, 1) # Zrcali okvir vodoravno
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Pretvori okvir iz BRG (OpenCV) v RGB za MediaPipe prikaz
                results = hands.process(frame_rgb) # Obdela RGN okvir za zaznavanje toč rok in leve/desne roke

                if results.multi_hand_landmarks: # Preveri, ali so bile v okvirju zaznane točke rok
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks): # Iterira čez zaznano roko, idx - indeks, hand_landmarks - podatke o točkah
                        mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Vizualizacija točke rok in povezave

                        if self.recording: # Preveri ali je snemanje aktivno
                            landmarks = np.zeros(63) # Ustvari tabelo z 63 ničlami za shranjevanje x, y, z koordinat 21 točk rok
                            for i, lm in enumerate(hand_landmarks.landmark): # Iterira čez 21 točk rok, i - indeks, lm - x, y, z koordinate
                                landmarks[i*3:i*3+3] = [lm.x, lm.y, lm.z] # Shrani koordinate trenutne točke v tabelo landmarks
                            self.frames.append(landmarks) # Doda podatke o točkah (63 koordinat) v seznam okvirjev za trenutni okvir

                            handedness = results.multi_handedness[idx].classification[0].label # Pridobi oznako leve/desne roke
                            self.handedness_labels.append(handedness) # Doda oznako leve/desne roke v seznam handedness_labels za trenutni okvir

                img = Image.fromarray(frame_rgb)
                img = ImageTk.PhotoImage(img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                self.canvas.image = img
        except Exception as e:
            print(f"[ERROR in update_video]: {e}")

        self.root.after(10, self.update_video) # Načrtuje ponoven klic update_video po 10 milisekundah za neprekinjen video cikel

    def quit(self):
        self.cap.release()
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = GestureGUI(root)
        root.mainloop()
    except Exception as e:
        import traceback
        traceback.print_exc()