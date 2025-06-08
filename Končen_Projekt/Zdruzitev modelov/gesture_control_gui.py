import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import sys

# Prisilimo utf-8 za standardni izhod
sys.stdout.reconfigure(encoding='utf-8')

# Inicializacija
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Nalaganje modelov
mirrors_model = tf.keras.models.load_model('Models/mirrors_model.keras')
windows_model = tf.keras.models.load_model('Models/windows_model.keras')
radio_model = tf.keras.models.load_model('Models/radio_model.keras')
climate_model = tf.keras.models.load_model('Models/climate_model.keras')

# Povprečje in standardni odklon
MIRRORS_MEAN = 0.32715722213633686
MIRRORS_STD = 0.3050974859034138
WINDOWS_MEAN = 0.3218953795343205
WINDOWS_STD = 0.2855003991679068
RADIO_MEAN = 0.1824056036195469
RADIO_STD = 0.2894906092020188
CLIMATE_MEAN = 0.20537318328833137
CLIMATE_STD = 0.2955649804045277

# Seznam razredov
MIRRORS_CLASSES = ['close_rm', 'down_rm', 'left_rm', 'open_rm', 'right_rm', 'up_rm']
WINDOWS_CLASSES = ['close_back_left_window', 'close_back_right_window', 'close_front_left_window', 'close_front_right_window', 'open_back_left_window', 'open_back_right_window', 'open_front_left_window', 'open_front_right_window']
RADIO_CLASSES = ['next_station', 'previous_station', 'turn_off_radio', 'turn_on_radio', 'volume_down', 'volume_up']
CLIMATE_CLASSES = ['climate_colder', 'climate_warmer', 'fan_stronger', 'fan_weaker']
MAX_FRAMES = 80

def compute_finger_distances(X):
    X_new = np.zeros((X.shape[0], 63 + 5 + 4))
    for f in range(X.shape[0]):
        landmarks = X[f].reshape(21, 3)
        if np.all(landmarks == 0):
            X_new[f, :63] = 0
            X_new[f, 63:] = 0
            continue

        X_new[f, :63] = X[f]
        wrist = landmarks[0]
        fingertip_indices = [4, 8, 12, 16, 20]
        for j, idx in enumerate(fingertip_indices):
            fingertip = landmarks[idx]
            distance = np.sqrt(np.sum((fingertip - wrist) ** 2))
            X_new[f, 63 + j] = distance

        fingertip_pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]
        for j, (idx1, idx2) in enumerate(fingertip_pairs):
            fingertip1, fingertip2 = landmarks[idx1], landmarks[idx2]
            distance = np.sqrt(np.sum((fingertip1 - fingertip2) ** 2))
            X_new[f, 63 + 5 + j] = distance

    return X_new

class GestureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Prepoznava Gest")
        self.root.geometry("450x700")  # Povečano za vertikalno postavitev
        self.root.configure(bg="#B1CACF")
        self.root.resizable(False, False)

        self.cap = cv2.VideoCapture(0)
        self.recording = False
        self.frames = []
        self.handedness_labels = []
        self.current_model = 'Vzvratna ogledala'

        self.models = {
            'Vzvratna ogledala': {'model': mirrors_model, 'mean': MIRRORS_MEAN, 'std': MIRRORS_STD, 'classes': MIRRORS_CLASSES},
            'Okna': {'model': windows_model, 'mean': WINDOWS_MEAN, 'std': WINDOWS_STD, 'classes': WINDOWS_CLASSES},
            'Radio': {'model': radio_model, 'mean': RADIO_MEAN, 'std': RADIO_STD, 'classes': RADIO_CLASSES},
            'Klimatska naprava': {'model': climate_model, 'mean': CLIMATE_MEAN, 'std': CLIMATE_STD, 'classes': CLIMATE_CLASSES}
        }

        # Uporaba teme
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TButton", font=("Segoe UI", 12, "bold"), padding=10, background="#005B8D", foreground="#FFFFFF")
        self.style.map("TButton", background=[("active", "#0073B1")], foreground=[("active", "#FFFFFF")])
        self.style.configure("Small.TButton", font=("Segoe UI", 10, "bold"), padding=5)
        self.style.map("Small.TButton", background=[("active", "#0073B1")], foreground=[("active", "#FFFFFF")])
        self.style.configure("TLabel", background="#B1CACF", foreground="#00378E", font=("Segoe UI", 10))
        self.style.configure("TFrame", background="#B1CACF")

        # Glavni okvir
        self.main_frame = ttk.Frame(self.root, padding=15, style="TFrame")
        self.main_frame.pack(fill="both", expand=True)

        # Naslov
        ttk.Label(self.main_frame, text="Prepoznava Gest", font=("Segoe UI", 16, "bold")).pack(pady=10)

        # Video okno
        self.video_canvas = tk.Canvas(self.main_frame, width=400, height=300, highlightthickness=1, bg="#FFFFFF")
        self.video_canvas.pack(pady=10)

        # Gumbi za izbiro modelov
        self.model_buttons = {}
        model_frame = ttk.Frame(self.main_frame, style="TFrame")
        model_frame.pack(pady=5)
        models = ['Vzvratna ogledala', 'Okna', 'Radio', 'Klimatska naprava']
        for i, model in enumerate(models):
            btn = ttk.Button(model_frame, text=model, command=lambda m=model: self.switch_model(m), width=16, style="Small.TButton")
            btn.grid(row=i, column=0, padx=5, pady=5)
            self.model_buttons[model] = btn

        # Gumbi za zajem in izhod
        button_frame = ttk.Frame(self.main_frame, style="TFrame")
        button_frame.pack(pady=5)
        self.btn_capture = ttk.Button(button_frame, text="Zajem Gest", command=self.start_recording, width=15)
        self.btn_capture.grid(row=0, column=0, padx=5)
        self.btn_quit = ttk.Button(button_frame, text="Izhod", command=self.quit, width=15)
        self.btn_quit.grid(row=0, column=1, padx=5)

        # Statusna vrstica
        self.status_var = tk.StringVar()
        self.status_var.set("Izberi model in začni zajem")
        self.status_label = ttk.Label(self.main_frame, textvariable=self.status_var, anchor="center", style="TLabel", wraplength=400)
        self.status_label.pack(pady=10, fill="x")

        self.update_video()

    def switch_model(self, model_name):
        self.current_model = model_name
        self.status_var.set(f"Izbran model: {model_name}")
        for model, btn in self.model_buttons.items():
            btn.config(state="normal" if model != model_name else "disabled")

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.frames = []
            self.handedness_labels = []
            self.btn_capture.config(text="Ustavi Zajem", state="normal")
            self.status_var.set("Zajemanje gest...")
            for btn in self.model_buttons.values():
                btn.config(state="disabled")
            self.btn_quit.config(state="disabled")
        else:
            self.recording = False
            self.btn_capture.config(text="Zajem Gest", state="normal")
            if self.frames:  # Obdelaj zadnje okvirje, če obstajajo
                self.process_gesture()
            for btn in self.model_buttons.values():
                btn.config(state="normal" if btn.cget("text") != self.current_model else "disabled")
            self.btn_quit.config(state="normal")

    def process_gesture(self):
        if len(self.frames) == 0:
            self.status_var.set("Ni zajetih okvirjev")
            return

        X = np.zeros((MAX_FRAMES, 63))
        for i in range(min(len(self.frames), MAX_FRAMES)):
            X[i] = self.frames[i]
        if len(self.frames) < MAX_FRAMES:
            X[len(self.frames):] = 0

        if self.current_model in ['Radio', 'Klimatska naprava']:
            X = compute_finger_distances(X)

        mean = self.models[self.current_model]['mean']
        std = self.models[self.current_model]['std']
        X = (X - mean) / std

        X = X[np.newaxis, ...]

        model = self.models[self.current_model]['model']
        pred = model.predict(X, verbose=0)
        pred_class = np.argmax(pred, axis=1)[0]
        gesture = self.models[self.current_model]['classes'][pred_class]

        display_text = f"{self.current_model}: {gesture}"
        self.status_var.set(f"Rezultat: {display_text}")
        print(f"Gesta: {display_text}")

        plt.figure(figsize=(10, 5))
        plt.plot(X[0, :, 0], label='Wrist X')
        plt.plot(X[0, :, 1], label='Wrist Y')
        plt.plot(X[0, :, 2], label='Wrist Z')
        plt.title(f'Trajectory for {gesture} ({self.current_model})')
        plt.xlabel('Frame')
        plt.ylabel('Coordinate')
        plt.legend()
        plt.savefig(f'captured_gesture_trajectory_{self.current_model.lower().replace(" ", "_")}.png')
        plt.close()

        # Vrnemo rezultat za merging_gui.py
        sys.stdout.write(display_text + "\n")
        sys.stdout.flush()

    def update_video(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        if self.recording:
                            landmarks = np.zeros(63)
                            for i, lm in enumerate(hand_landmarks.landmark):
                                landmarks[i*3:i*3+3] = [lm.x, lm.y, lm.z]
                            self.frames.append(landmarks)
                            handedness = results.multi_handedness[idx].classification[0].label
                            self.handedness_labels.append(handedness)

                            # Obdelaj gesto, če je zbranih MAX_FRAMES
                            if len(self.frames) >= MAX_FRAMES:
                                self.process_gesture()
                                self.frames = []  # Ponastavi za naslednjo gesto
                                self.handedness_labels = []

                img = Image.fromarray(frame_rgb)
                img = img.resize((400, 300), Image.Resampling.LANCZOS)
                img = ImageTk.PhotoImage(img)
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img)
                self.video_canvas.image = img
        except Exception as e:
            print(f"[ERROR in update_video]: {e}")

        self.root.after(10, self.update_video)

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