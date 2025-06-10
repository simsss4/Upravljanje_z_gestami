import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
import tkinter as tk
from tkinter import filedialog
import torch.nn as nn
import torch.nn.functional as F
import os
from prometheus_client import start_http_server, Counter, Gauge, Summary
import psutil
import time
import paho.mqtt.client as mqtt
import os
import traceback

# Start metrics server on a different port
start_http_server(8001, addr='0.0.0.0')

# Metrics
cpu_usage_environment = Gauge('cpu_usage_environment_analysis_percent', 'CPU usage percent for environment analysis')
memory_usage_environment = Gauge('memory_usage_environment_analysis_bytes', 'Memory usage bytes for environment analysis')
environment_prediction_total = Counter('environment_prediction_total', 'Total number of environment predictions', ['model_name'])
environment_confidence_summary = Summary('environment_confidence_score', 'Confidence scores of environment predictions', ['model_name'])
image_processing_time = Summary('image_processing_time_seconds', 'Time to process each image')

proc = psutil.Process()


def on_connect(client, userdata, flags, rc):
    print(f"MQTT povezan z rezultatom: {rc}")


def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with result code {rc}")

def on_message(client, userdata, msg):
    print(f"Received message from {msg.topic}: {msg.payload.decode()}")

client = mqtt.Client()
client.on_connect = on_connect
client.connect("172.25.70.243", 1883, 60)
client.loop_start()

# Model classes
class WeatherCNN(nn.Module):
    def __init__(self):
        super(WeatherCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TimeOfDayCNN(nn.Module):
    def __init__(self):
        super(TimeOfDayCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load models
daynight_model = TimeOfDayCNN()
weather_model = WeatherCNN()

daynight_weights = torch.load("Models/model_daynight.pth", map_location='cpu', weights_only=True)
weather_weights = torch.load("Models/model_weather.pth", map_location='cpu', weights_only=True)

daynight_model.load_state_dict(daynight_weights)
weather_model.load_state_dict(weather_weights)

daynight_model.eval()
weather_model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Labels
timeofday_labels = ['day', 'night']
weather_labels = ['clear', 'foggy', 'rainy']

# Simulate car behavior
def simulate_car_behavior(time_label, weather_label):
    actions = []
    if time_label == 'night':
        actions.append("Vklop - Vklop zasenčenih (ali dolgih) žarometov.")
        actions.append("Vklop - Ambiente svetlobe.")
        actions.append("Zatemnitev - Armaturne plošče in zaslona")
        actions.append("Opozorilo - Če je slaba vidljivost zmanjšaj hitrost!")
    else:
        actions.append("Vklop - Dnevnih luči")

    if weather_label == 'rainy':
        actions.append("Vklop - Brisalcev vetrobranskega stekla")
        actions.append("Vklop - Gretje stekel in ogledal")
        actions.append("Opozorilo - Zmanjšaj hitrost!")
    elif weather_label == 'foggy':
        actions.append("Vklop - Meglenk (sprednjih in zadnjih)")
        actions.append("Opozorilo - Zmanjšaj hitrost!")
    else:
        actions.append("Jasno vreme: Priporočena normalna vožnja")

    return actions

# File explorer for image selection
root = tk.Tk()
root.withdraw()  # Hide the main window
print("Please select an image file...")
img_path = filedialog.askopenfilename(
    title="Select Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if not img_path:
    print("No image selected. Exiting...")
    exit()



# MQTT Setup
def setup_mqtt():
    try:
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_log = on_log
        client.on_message = on_message
        client.on_disconnect = on_disconnect

        # Hardcoded MQTT Broker IP
        mqtt_broker = "172.25.70.243"

        client.connect(mqtt_broker, 1883, 60)
        client.loop_start()
        return client
    except Exception as e:
        print(f"Error in MQTT setup: {e}")
        traceback.print_exc()
        return None

# Initialize MQTT Client
mqtt_client = setup_mqtt()


# Update image processing
try:
    image = Image.open(img_path).convert('RGB')
    start_time = time.time()
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        # Day/night prediction
        pred_dn = daynight_model(input_tensor)
        pred_dn_probs = torch.softmax(pred_dn, dim=1)[0]
        pred_dn_idx = torch.argmax(pred_dn_probs).item()
        pred_dn_conf = pred_dn_probs[pred_dn_idx].item()
        environment_prediction_total.labels(model_name='daynight').inc()
        environment_confidence_summary.labels(model_name='daynight').observe(pred_dn_conf)

        # Weather prediction
        pred_wt = weather_model(input_tensor)
        pred_wt_probs = torch.softmax(pred_wt, dim=1)[0]
        pred_wt_idx = torch.argmax(pred_wt_probs).item()
        pred_wt_conf = pred_wt_probs[pred_wt_idx].item()
        environment_prediction_total.labels(model_name='weather').inc()
        environment_confidence_summary.labels(model_name='weather').observe(pred_wt_conf)


    # Update system metrics
    cpu_usage_environment.set(proc.cpu_percent(interval=None))
    memory_usage_environment.set(proc.memory_info().rss)
    image_processing_time.observe(time.time() - start_time)

    print(f"\nStanje okolja: {weather_labels[pred_wt_idx]} {timeofday_labels[pred_dn_idx]}")
    actions = simulate_car_behavior(timeofday_labels[pred_dn_idx], weather_labels[pred_wt_idx])
    for action in actions:
        print("🚗", action)

        # Publish MQTT messages after environment analysis
    if mqtt_client:
        mqtt_client.publish("environment/status", f"Weather: {weather_labels[pred_wt_idx]}, Time of Day: {timeofday_labels[pred_dn_idx]}")
        mqtt_client.publish("environment/metrics", f"CPU: {proc.cpu_percent()}%, Memory: {proc.memory_info().rss} bytes")

        # Send alerts for low visibility conditions
        if weather_labels[pred_wt_idx] in ["foggy", "rainy"] or timeofday_labels[pred_dn_idx] == "night":
            mqtt_client.publish("environment/alerts", "⚠️ Low visibility detected! Adjust driving speed.")

except Exception as e:
    print(f"Error processing image: {str(e)}")

if mqtt_client:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()