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

# Model classes
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

# Process selected image
try:
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    # Run through models
    with torch.no_grad():
        pred_dn = torch.argmax(daynight_model(input_tensor), dim=1).item()
        pred_wt = torch.argmax(weather_model(input_tensor), dim=1).item()

    # Print results
    print(f"\nStanje okolja: {weather_labels[pred_wt]} {timeofday_labels[pred_dn]}")
    actions = simulate_car_behavior(timeofday_labels[pred_dn], weather_labels[pred_wt])
    for action in actions:
        print("🚗", action)

except Exception as e:
    print(f"Error processing image: {str(e)}")