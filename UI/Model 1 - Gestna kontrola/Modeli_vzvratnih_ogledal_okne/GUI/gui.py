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

# Povprečje in standardni odklon za normalizacijo podatkov za modele (iz skripte: mean_std_mirrors.py/mean_std_windows.py)
MIRRORS_MEAN = 0.32715722213633686  
MIRRORS_STD = 0.3050974859034138    
WINDOWS_MEAN = 0.3218953795343205  
WINDOWS_STD = 0.2855003991679068  

# Seznam imen razredov (z labele - izhodne vrednosti)
MIRRORS_CLASSES = ['close_rm', 'down_rm', 'left_rm', 'open_rm', 'right_rm', 'up_rm']
WINDOWS_CLASSES = ['close_back_left_window', 'close_back_right_window', 'close_front_left_window', 'close_front_right_window', 'open_back_left_window', 'open_back_right_window', 'open_front_left_window', 'open_front_right_window']
MAX_FRAMES = 80

class GestureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition (Windows & Mirrors)")
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
            }
        }

        # GUI elemente
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.model_var = tk.StringVar(value='Mirrors')
        self.model_menu = tk.OptionMenu(root, self.model_var, 'Mirrors', 'Windows', command=self.switch_model)
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
        if len(self.frames) == 0: # Preveri, ali ni bilo zajetih nobenih okvirjev med snemanjem
            self.label.config(text="No frames captured")
            return

        X = np.zeros((MAX_FRAMES, 63)) # Ustvari tabelo oblike (80,63) z ničlami za shranjevanje podatkov o gestah
        for i in range(min(len(self.frames), MAX_FRAMES)): # Iterira čez zajete okvirje, do največ MAX_FRAMES (80), da prepreči prekoračitev
            X[i] = self.frames[i] # Kopira koordinate točk (63) iz zajetega okvirja v ustrezno vrstico tabele X
        if len(self.frames) < MAX_FRAMES: # Preveri, ali je bilo zajetih manj kot 80 okvirjev
            X[len(self.frames):] = 0 # Napolni preostale vrstice v X z ničlami, da zagotovi 80 okvirjev (padding)

        #Normalizacija
        mean = self.models[self.current_model]['mean'] # Pridobi povprečje za normalizacijo iz nastavitev trenutnega modela
        std = self.models[self.current_model]['std'] # Pridobi standardni odklon za normalizacijo iz nastavitev trenutnega modela
        X = (X - mean) / std # Normalizira vhodne podatke z odštevanjem povprečja in deljenjem s standardnim odklonom

        X = X[np.newaxis, ...] # Preoblikovanje X iz (80, 63) v (1, 80, 63) za vhod modela

        # Predikcije
        model = self.models[self.current_model]['model'] # Pridobi objekt modela za trenutni model
        pred = model.predict(X, verbose=0) # Izvede napoved modela na normaliziranih podatkih X, vrne verjetnosti razreda
        pred_class = np.argmax(pred, axis=1)[0] # Poišče indeks razreda z najvišjo verjetnostjo za izbor napovedane geste
        pred_conf = pred[0, pred_class] # Pridobi verjetnost za napovedani razred
        gesture = self.models[self.current_model]['classes'][pred_class] # Preslika indeks napovedanega iazreda v labelo

        # Preveri, ali so bile zbrane oznake leve/desne roke med snemanjem
        if self.handedness_labels:
            left_count = self.handedness_labels.count('Left') # Prešteje, koliko okvirjev je označenih kot leva roka
            right_count = self.handedness_labels.count('Right') # Prešteje, koliko okvirjev je označenih kot desna roka
            hand = 'Left' if left_count >= right_count else 'Right' # Določi uporabljeno roko
        else:
            hand = 'Unknown'

        # Preveri, ali je trenutni model za ogledala, da prilagodi izpis
        if self.current_model == 'Mirrors':
            display_text = f"{hand} mirror: {gesture} ({pred_conf:.2%})"
        else:
            display_text = f"Window: {gesture} ({pred_conf:.2%})"

        # Ustvari formatiran niz za geste oken
        self.label.config(text=f"Predicted Gesture: {display_text}")

        # Ustvari sliko Matplotlib za risanje poti zapestja
        plt.figure(figsize=(10, 5))
        plt.plot(X[0, :, 0], label='Wrist X') # Nariše x-koordinate zapestja (točka 0) čez vse okvirje, označeno kot "Wrist X"
        plt.plot(X[0, :, 1], label='Wrist Y') # Nariše y-koordinate zapestja čez vse okvirje, označeno kot "Wrist Y"
        plt.plot(X[0, :, 2], label='Wrist Z') # Nariše z-koordinate zapestja čez vse okvirje, označeno kot "Wrist Z"
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