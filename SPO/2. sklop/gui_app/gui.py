import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN
os.environ['CUDA_VISIBLE_DEVICES'] = ''    # Disable GPU
import sys
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import queue
from prometheus_client import start_http_server, Counter, Gauge, Summary
import paho.mqtt.client as mqtt
import psutil
import time
from tkinter import ttk

# Start metrics server
start_http_server(8000, addr='0.0.0.0')

# Metrics
cpu_usage_mqtt_docker = Gauge('cpu_usage_mqtt_docker_percent', 'CPU usage percent for MQTT/Docker application')
memory_usage_mqtt_docker = Gauge('memory_usage_mqtt_docker_bytes', 'Memory usage bytes for MQTT/Docker application')
mqtt_messages_per_sec = Counter('mqtt_messages_per_second', 'MQTT messages sent or received per second')
mqtt_logs_per_sec = Counter('mqtt_logs_per_second', 'MQTT logs per second')
processed_frames_per_sec = Gauge('processed_frames_per_second', 'Processed frames per second in gesture recognition algorithm')
cpu_usage_gesture_algorithm = Gauge('cpu_usage_gesture_algorithm_percent', 'CPU usage percent of gesture recognition algorithm')
memory_usage_gesture_algorithm = Gauge('memory_usage_gesture_algorithm_bytes', 'Memory usage bytes of gesture recognition algorithm')
processed_mqtt_messages_total = Counter('processed_mqtt_messages_total', 'Total number of processed MQTT messages')
recognized_gestures = Gauge('recognized_gestures_per_frame', 'Number of recognized gestures per frame')
model_predictions_total = Counter('gesture_model_predictions_total', 'Total number of predictions per model', ['model_name'])
predicted_gestures_total = Counter('predicted_gestures_total', 'Total number of times a gesture was predicted', ['model_name', 'gesture'])
gesture_confidence_summary = Summary('gesture_confidence_score', 'Confidence scores of predicted gestures', ['model_name'])

proc = psutil.Process()
last_mqtt_message_count = 0
last_mqtt_log_count = 0
last_time = time.time()

def update_system_metrics():
    global last_mqtt_message_count, last_mqtt_log_count, last_time
    try:
        now = time.time()
        elapsed = now - last_time
        if elapsed == 0:
            return
        cpu_gesture = proc.cpu_percent(interval=None)
        mem_gesture = proc.memory_info().rss
        cpu_usage_gesture_algorithm.set(cpu_gesture)
        memory_usage_gesture_algorithm.set(mem_gesture)
        cpu_usage_mqtt_docker.set(cpu_gesture)
        memory_usage_mqtt_docker.set(mem_gesture)
        current_message_count = processed_mqtt_messages_total._value.get()
        delta_messages = current_message_count - last_mqtt_message_count
        mqtt_messages_per_sec.inc(delta_messages)
        last_mqtt_message_count = current_message_count
        last_time = now
    except Exception as e:
        print(f"Error in update_system_metrics: {e}")
        import traceback
        traceback.print_exc()

def on_connect(client, userdata, flags, rc):
    print(f"MQTT povezan z rezultatom: {rc}")

def on_log(client, userdata, level, buf):
    print("MQTT log:", buf)
    mqtt_logs_per_sec.inc()

def on_message(client, userdata, msg):
    print(f"Received message: {msg.topic} -> {msg.payload.decode()}")
    processed_mqtt_messages_total.inc()

def on_disconnect(client, userdata, rc):
    print("Disconnected from MQTT broker")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load models
try:
    mirrors_model = tf.keras.models.load_model('Models/mirrors_model.keras')
    windows_model = tf.keras.models.load_model('Models/windows_model.keras')
    radio_model = tf.keras.models.load_model('Models/radio_model.keras')
    climate_model = tf.keras.models.load_model('Models/climate_model.keras')
    print("Models loaded successfully")
    print("Mirrors model input shape:", mirrors_model.input_shape)
    print("Radio model input shape:", radio_model.input_shape)
except Exception as e:
    print(f"Error loading models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Model parameters
MIRRORS_MEAN = 0.32715722213633686
MIRRORS_STD = 0.3050974859034138
WINDOWS_MEAN = 0.3218953795343205
WINDOWS_STD = 0.2855003991679068
RADIO_MEAN = 0.1824056036195469
RADIO_STD = 0.2894906092020188
CLIMATE_MEAN = 0.20537318328833137
CLIMATE_STD = 0.2955649804045277
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


def setup_mqtt(self):
    try:
        self.client = mqtt.Client()
        self.client.on_connect = on_connect
        self.client.on_log = on_log
        self.client.on_message = on_message
        self.client.on_disconnect = on_disconnect
        mqtt_broker = os.getenv('MQTT_BROKER', 'mosquitto')  # Use service name
        self.client.connect(mqtt_broker, 1883, 60)
        self.client.loop_start()
    except Exception as e:
        print(f"Error in MQTT setup: {e}")
        import traceback
        traceback.print_exc()
        self.client = None



class GestureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Prepoznava Gest")
        self.root.geometry("450x700")
        self.root.configure(bg="#B1CACF")
        self.root.resizable(False, False)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            self.status_var = tk.StringVar()
            self.status_var.set("Napaka: Ni mogoče odpreti kamere")
            tk.Label(self.root, textvariable=self.status_var).pack()
            sys.exit(1)

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

        # GUI setup
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TButton", font=("Segoe UI", 12, "bold"), padding=10, background="#005B8D", foreground="#FFFFFF")
        self.style.map("TButton", background=[("active", "#0073B1")], foreground=[("active", "#FFFFFF")])
        self.style.configure("Small.TButton", font=("Segoe UI", 10, "bold"), padding=5)
        self.style.map("Small.TButton", background=[("active", "#0073B1")], foreground=[("active", "#FFFFFF")])
        self.style.configure("TLabel", background="#B1CACF", foreground="#00378E", font=("Segoe UI", 10))
        self.style.configure("TFrame", background="#B1CACF")

        self.main_frame = ttk.Frame(self.root, padding=15, style="TFrame")
        self.main_frame.pack(fill="both", expand=True)
        ttk.Label(self.main_frame, text="Prepoznava Gest", font=("Segoe UI", 16, "bold")).pack(pady=10)
        self.video_canvas = tk.Canvas(self.main_frame, width=400, height=300, highlightthickness=1, bg="#FFFFFF")
        self.video_canvas.pack(pady=10)

        self.model_buttons = {}
        model_frame = ttk.Frame(self.main_frame, style="TFrame")
        model_frame.pack(pady=5)
        models = ['Vzvratna ogledala', 'Okna', 'Radio', 'Klimatska naprava']
        for i, model in enumerate(models):
            btn = ttk.Button(model_frame, text=model, command=lambda m=model: self.switch_model(m), width=16, style="Small.TButton")
            btn.grid(row=i, column=0, padx=5, pady=5)
            self.model_buttons[model] = btn

        button_frame = ttk.Frame(self.main_frame, style="TFrame")
        button_frame.pack(pady=5)
        self.btn_capture = ttk.Button(button_frame, text="Zajem Gest", command=self.start_recording, width=15)
        self.btn_capture.grid(row=0, column=0, padx=5)
        self.btn_quit = ttk.Button(button_frame, text="Izhod", command=self.quit, width=15)
        self.btn_quit.grid(row=0, column=1, padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("Izberi model in začni zajem")
        self.status_label = ttk.Label(self.main_frame, textvariable=self.status_var, anchor="center", style="TLabel", wraplength=400)
        self.status_label.pack(pady=10, fill="x")

        self.setup_mqtt()
        self.update_metrics_periodically()
        self.update_video()

    def setup_mqtt(self):
        try:
            self.client = mqtt.Client()
            self.client.on_connect = on_connect
            self.client.on_log = on_log
            self.client.on_message = on_message
            self.client.on_disconnect = on_disconnect
            self.client.connect("172.25.70.243", 1883, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"Error in MQTT setup: {e}")
            import traceback
            traceback.print_exc()
            self.client = None

    def update_metrics_periodically(self):
        try:
            update_system_metrics()
        except Exception as e:
            print(f"Error in update_metrics_periodically: {e}")
            import traceback
            traceback.print_exc()
        self.root.after(1000, self.update_metrics_periodically)

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
            if self.frames:
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
        try:
            pred = model.predict(X, verbose=0)
            pred_class = np.argmax(pred, axis=1)[0]
            pred_conf = pred[0][pred_class]
            gesture = self.models[self.current_model]['classes'][pred_class]

            model_predictions_total.labels(model_name=self.current_model).inc()
            predicted_gestures_total.labels(model_name=self.current_model, gesture=gesture).inc()
            gesture_confidence_summary.labels(model_name=self.current_model).observe(pred_conf)

            display_text = f"{self.current_model}: {gesture} ({pred_conf:.2%})"
            self.status_var.set(f"Rezultat: {display_text}")
            print(f"Gesta: {display_text}")

            if self.client:
                topic = f"gestures/{self.current_model.lower().replace(' ', '_')}"
                message = f"{gesture} ({pred_conf:.2%})"
                self.client.publish(topic, message)
                print(f"Published MQTT message: topic={topic}, message={message}")
        except Exception as e:
            print(f"Error during model prediction: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Napaka pri obdelavi geste: {e}")
            return

        try:
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
        except Exception as e:
            print(f"Error plotting trajectory: {e}")
            import traceback
            traceback.print_exc()

        sys.stdout.write(display_text + "\n")
        sys.stdout.flush()

    def update_video(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame from webcam")
                self.status_var.set("Napaka: Ni mogoče zajeti slike iz kamere")
                return
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

                        if len(self.frames) >= MAX_FRAMES:
                            self.process_gesture()
                            self.frames = []
                            self.handedness_labels = []

            img = Image.fromarray(frame_rgb)
            img = img.resize((400, 300), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(img)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.video_canvas.image = img
        except Exception as e:
            print(f"Error in update_video: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Napaka pri posodabljanju videa: {e}")
        self.root.after(30, self.update_video)

    def quit(self):
        try:
            self.cap.release()
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            print(f"Error in quit: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = GestureGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Main application error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)