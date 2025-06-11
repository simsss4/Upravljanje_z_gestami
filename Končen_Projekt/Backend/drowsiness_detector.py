import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import sys
import tkinter as tk
from tkinter import filedialog
from prometheus_client import start_http_server, Counter, Gauge, Summary
import psutil
import time
import paho.mqtt.client as mqtt
import traceback

# Start metrics server on a different port
start_http_server(8002, addr='0.0.0.0')

# Metrics
cpu_usage_drowsiness = Gauge('cpu_usage_drowsiness_detection_percent', 'CPU usage percent for drowsiness detection')
memory_usage_drowsiness = Gauge('memory_usage_drowsiness_detection_bytes', 'Memory usage bytes for drowsiness detection')
drowsiness_prediction_total = Counter('drowsiness_prediction_total', 'Total number of drowsiness predictions')
drowsiness_confidence_summary = Summary('drowsiness_confidence_score', 'Confidence scores of drowsiness predictions')
frame_processing_time = Summary('frame_processing_time_seconds', 'Time to process each video frame')

proc = psutil.Process()

# Define missing MQTT callbacks
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with result code {rc}")

def on_message(client, userdata, msg):
    print(f"Received message from {msg.topic}: {msg.payload.decode()}")

def on_disconnect(client, userdata, rc):
    print("Disconnected from MQTT Broker")

def on_log(client, userdata, level, buf):
    print(f"MQTT Log: {buf}")


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

#172.25.70.243
zeroTierHostIP = "10.147.20.65";

# MQTT Setup
def setup_mqtt():
    try:
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        client.on_log = on_log

        # Hardcoded MQTT Broker IP
        mqtt_broker = zeroTierHostIP

        client.connect(mqtt_broker, 1883, 60)
        client.loop_start()
        return client
    except Exception as e:
        print(f"Error in MQTT setup: {e}")
        traceback.print_exc()
        return None

# Initialize MQTT Client
mqtt_client = setup_mqtt()

def analyze_drowsiness(video_path):
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video ne obstaja: {video_path}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model ne obstaja: {MODEL_FILE}")

        # Load model
        model = load_model(MODEL_PATH)

        # Validate input shape
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
            start_time = time.time()  # Fix: Define start_time here for each frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            input_img = resized.astype('float32') / 255.0
            input_img = np.expand_dims(input_img, axis=(0, -1))

            prediction = model.predict(input_img, verbose=0)[0][0]
            is_drowsy = prediction > DROWSY_THRESHOLD
            history.append(is_drowsy)

            # Determine driver state
            current_state = "Utrujen" if is_drowsy else "Buden"

            # Publish MQTT updates
            if mqtt_client:
                mqtt_client.publish("drowsiness/status", current_state)
                mqtt_client.publish("drowsiness/metrics", f"CPU: {proc.cpu_percent()}%, Memory: {proc.memory_info().rss} bytes")

                if is_drowsy:
                    mqtt_client.publish("drowsiness/alert", "⚠️ Driver is drowsy! Take a break!")

            print(f"🚗 MQTT Update: {current_state}")

            # Update metrics
            drowsiness_prediction_total.inc()
            drowsiness_confidence_summary.observe(prediction)
            cpu_usage_drowsiness.set(proc.cpu_percent(interval=None))
            memory_usage_drowsiness.set(proc.memory_info().rss)
            frame_processing_time.observe(time.time() - start_time)

            # Check for state change
            if len(history) == history.maxlen and sum(history) >= HISTORY_FRACTION * len(history):
                current_state = "Utrujen"
                label_color = (0, 0, 255)  # Red
            else:
                current_state = "Buden"
                label_color = (0, 255, 0)  # Green

            if current_state != last_state:
                if current_state == "Utrujen":
                    print("Opozorilo: Voznik zaspi!", flush=True)
                else:
                    print("Voznik je buden", flush=True)
                last_state = current_state

            # Display state on frame
            cv2.putText(frame, f"Stanje: {current_state}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
            cv2.imshow("Prepoznavanje utrujenosti", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Properly close MQTT connection at script exit
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()

    except Exception as e:
        error_message = f"Napaka pri analizi utrujenosti: {str(e)}"
        print(error_message, flush=True)

if __name__ == "__main__":
    video_path = select_video()
    if video_path:
        analyze_drowsiness(video_path)
    else:
        print("Video ni izbran. Prekinjam.", flush=True)