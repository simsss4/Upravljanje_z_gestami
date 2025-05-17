import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from threading import Thread
import queue

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
radio_model = tf.keras.models.load_model('radio_model.keras')
climate_model = tf.keras.models.load_model('climate_model.keras')

# Load or define mean and std for normalization (from training data)
# Replace these with actual values from your training process
RADIO_MEAN = 0.1824056036195469
RADIO_STD = 0.2894906092020188
CLIMATE_MEAN = 0.2143674777389519
CLIMATE_STD = 0.30413323740495335

# Class names
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
        self.root.title("Gesture Recognition (Radio & Climate)")
        self.cap = cv2.VideoCapture(0)
        self.recording = False
        self.frames = []
        self.frame_queue = queue.Queue()
        self.current_model = 'Radio'  # Default model

        # Model configurations
        self.models = {
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

        # GUI elements
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.model_var = tk.StringVar(value='Radio')
        self.model_menu = tk.OptionMenu(root, self.model_var, 'Radio', 'Climate', command=self.switch_model)
        self.model_menu.pack(pady=5)

        self.label = tk.Label(root, text="Predicted Gesture: None", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn_capture = tk.Button(root, text="Capture Gesture", command=self.start_recording)
        self.btn_capture.pack(pady=5)

        self.btn_quit = tk.Button(root, text="Quit", command=self.quit)
        self.btn_quit.pack(pady=5)

        # Start video feed
        self.update_video()

    def switch_model(self, model_name):
        self.current_model = model_name
        self.label.config(text=f"Predicted Gesture: None (Using {model_name} model)")

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.frames = []
            self.btn_capture.config(text="Stop Recording")
            self.label.config(text="Recording...")
        else:
            self.recording = False
            self.btn_capture.config(text="Capture Gesture")
            self.process_gesture()

    def process_gesture(self):
        if len(self.frames) == 0:
            self.label.config(text="No frames captured")
            return

        # Convert frames to model input
        X = np.zeros((MAX_FRAMES, 63))  # 63 = 21 landmarks × 3 (x, y, z)
        for i in range(min(len(self.frames), MAX_FRAMES)):
            X[i] = self.frames[i]
        if len(self.frames) < MAX_FRAMES:
            X[len(self.frames):] = 0  # Pad with zeros

        # Compute finger distances
        X = compute_finger_distances(X)  # Shape: (80, 72)

        # Normalize using model-specific parameters
        mean = self.models[self.current_model]['mean']
        std = self.models[self.current_model]['std']
        X = (X - mean) / std

        # Reshape for model input
        X = X[np.newaxis, ...]  # Shape: (1, 80, 72)

        # Predict
        model = self.models[self.current_model]['model']
        pred = model.predict(X, verbose=0)
        pred_class = np.argmax(pred, axis=1)[0]
        pred_conf = pred[0, pred_class]
        gesture = self.models[self.current_model]['classes'][pred_class]

        # Display prediction
        self.label.config(text=f"Predicted Gesture: {gesture} ({pred_conf:.2%})")

        # Plot wrist trajectory
        plt.figure(figsize=(10, 5))
        plt.plot(X[0, :, 0], label='Wrist X')
        plt.plot(X[0, :, 1], label='Wrist Y')
        plt.plot(X[0, :, 2], label='Wrist Z')
        plt.title(f'Wrist Trajectory for {gesture} ({self.current_model})')
        plt.xlabel('Frame')
        plt.ylabel('Coordinate')
        plt.legend()
        plt.savefig(f'captured_gesture_trajectory_{self.current_model.lower()}.png')
        plt.close()

    def update_video(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        if self.recording:
                            landmarks = np.zeros(63)
                            for i, lm in enumerate(hand_landmarks.landmark):
                                landmarks[i*3:i*3+3] = [lm.x, lm.y, lm.z]
                            self.frames.append(landmarks)

                img = Image.fromarray(frame_rgb)
                img = ImageTk.PhotoImage(img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                self.canvas.image = img
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
