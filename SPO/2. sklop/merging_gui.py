import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import threading
from prometheus_client import start_http_server, Counter, Gauge
import psutil
import time
import traceback
import paho.mqtt.client as mqtt
from prometheus_client import Counter, Gauge, start_http_server
import psutil
import time

start_http_server(8003, addr='0.0.0.0')
cpu_usage_percent = Gauge('cpu_usage_percent', 'CPU usage percentage')
memory_usage_bytes = Gauge('memory_usage_bytes', 'Memory usage in bytes')
mqtt_messages_total = Counter('mqtt_messages_total', 'Total MQTT messages')
processed_messages_total = Counter('processed_messages_total', 'Total processed messages')
processed_frames_per_second = Gauge('processed_frames_per_second', 'Processed frames per second')


# Metrics
button_clicks_total = Counter('button_clicks_total', 'Total button clicks', ['button'])
script_execution_total = Counter('script_execution_total', 'Total successful script executions', ['script'])
script_execution_errors_total = Counter('script_execution_errors_total', 'Total script execution errors', ['script'])
gui_uptime_seconds = Gauge('gui_uptime_seconds', 'GUI uptime in seconds')
processed_merging_messages_total = Counter('processed_merging_messages_total', 'Total merging messages processed')

# Track process and start time
proc = psutil.Process()
start_time = time.time()

# Update GUI uptime
def update_uptime():
    gui_uptime_seconds.set(time.time() - start_time)
    root.after(1000, update_uptime)  # Update every second

# MQTT Callbacks
def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"Connected to MQTT Broker with result code {reason_code}")
    client.subscribe("environment/#")
    client.subscribe("gestures/#")
    client.subscribe("drowsiness/#")

def on_message(client, userdata, msg):
    print(f"Received: {msg.topic} -> {msg.payload.decode()}")
    processed_merging_messages_total.inc()  # Increment the counter
    mqtt_messages_total.inc()  # Track total MQTT messages received

def on_log(client, userdata, level, buf):
    print(f"MQTT log: {buf}")

def on_disconnect(client, userdata, *args, **kwargs):
    print(f"Disconnected with code {args[0] if args else 'unknown'}")

def update_metrics():
    cpu_usage_percent.set(psutil.cpu_percent())  # Update CPU usage
    memory_usage_bytes.set(psutil.virtual_memory().used)  # Update memory usage
    root.after(5000, update_metrics)  # Schedule updates every 5 seconds



# Initialize MQTT client
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_connect = on_connect
mqtt_client.on_log = on_log
mqtt_client.on_message = on_message
mqtt_client.on_disconnect = on_disconnect
mqtt_client.connect("172.25.70.243", 1883, 60)
mqtt_client.loop_start()

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("UI projekt - backend")
        self.root.geometry("500x400")
        self.root.configure(bg="#B1CACF")
        self.root.resizable(False, False)
        
        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Font settings
        self.title_font = ("Helvetica", 18, "bold")
        self.button_font = ("Helvetica", 12, "bold")
        self.text_font = ("Helvetica", 10)
        
        # Configure colors
        self.style.configure("TFrame", background="#B1CACF")
        self.style.configure("TLabel", background="#B1CACF", foreground="#00378E", font=self.text_font)
        self.style.configure("Title.TLabel", font=self.title_font)
        self.style.configure("TButton", 
                            font=self.button_font,
                            padding=15,
                            background="#005B8D",
                            foreground="#FFFFFF",
                            borderwidth=1)
        self.style.map("TButton",
                      background=[("active", "#0073B1")],
                      foreground=[("active", "#FFFFFF")])
        
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding=20, style="TFrame")
        self.main_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(self.main_frame, 
                               text="Izberi model", 
                               style="Title.TLabel")
        title_label.pack(pady=(0, 30))
        
        # Buttons frame
        buttons_frame = ttk.Frame(self.main_frame, style="TFrame")
        buttons_frame.pack(pady=10)
        
        # Environment Analysis Button
        env_button = ttk.Button(buttons_frame, 
                              text="Analiza okolja", 
                              command=self.run_environment_analysis,
                              width=20)
        env_button.pack(pady=10)
        
        # Gesture Control Button
        gesture_button = ttk.Button(buttons_frame, 
                                 text="Gestna kontrola", 
                                 command=self.run_gesture_control,
                                 width=20)
        gesture_button.pack(pady=10)
        
        # Driver Analysis Button
        driver_button = ttk.Button(buttons_frame, 
                                text="Analiza voznika", 
                                command=self.run_driver_analysis,
                                width=20)
        driver_button.pack(pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Pripravljen")
        status_label = ttk.Label(self.main_frame, 
                               textvariable=self.status_var, 
                               style="TLabel")
        status_label.pack(side="bottom", pady=(20, 0))
        
        # Start uptime updates
        update_uptime()
        # Call it once to start periodic updates
        update_metrics()


    def run_script_in_thread(self, script_name, status_message, button_label, script_label):
        """Helper function to run a script in a separate thread"""
        button_clicks_total.labels(button=button_label).inc()
        def thread_target():
            self.status_var.set(status_message)
            self.root.update()
            try:
                subprocess.run([sys.executable, script_name], check=True)
                script_execution_total.labels(script=script_label).inc()
                self.status_var.set(f"{status_message.split('...')[0]} zaključena")
            except Exception as e:
                script_execution_errors_total.labels(script=script_label).inc()
                self.status_var.set(f"Napaka pri {status_message.split('...')[0].lower()}: {str(e)}")
            self.root.update()
        
        thread = threading.Thread(target=thread_target, daemon=True)
        thread.start()
    
    def run_environment_analysis(self):
        self.run_script_in_thread("detect_env_data.py", "Zaganjam analizo okolja...", 
                                "environment_analysis", "detect_env_data")
    
    def run_gesture_control(self):
        self.run_script_in_thread("gesture_control_gui.py", "Zaganjam gestno kontrolo...", 
                                "gesture_control", "gesture_control_gui")
    
    def run_driver_analysis(self):
        self.run_script_in_thread("drowsiness_detector.py", "Zaganjam analizo voznika...", 
                                "driver_analysis", "drowsiness_detector")

    def metric_updater():
        while True:
            cpu_usage_percent.set(psutil.cpu_percent())
            memory_usage_bytes.set(psutil.virtual_memory().used)
            time.sleep(5)  # Adjust the interval as needed

    # Start the metric updater thread
    threading.Thread(target=metric_updater, daemon=True).start()

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        root = tk.Tk()
        app = MainApplication(root)
        root.mainloop()
    except Exception as e:
        script_execution_errors_total.labels(script='merging_gui').inc()
        traceback.print_exc()