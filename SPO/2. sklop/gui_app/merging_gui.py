import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import os
import threading
import torch

sys.stdout.reconfigure(encoding='utf-8')

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
        
    def run_script_in_thread(self, script_name, status_message):
        """Helper function to run a script in a separate thread"""
        def thread_target():
            self.status_var.set(status_message)
            self.root.update()
            try:
                subprocess.run([sys.executable, script_name], check=True)
                self.status_var.set(f"{status_message.split('...')[0]} zaključena")
            except Exception as e:
                self.status_var.set(f"Napaka pri {status_message.split('...')[0].lower()}: {str(e)}")
            self.root.update()
        
        thread = threading.Thread(target=thread_target, daemon=True)
        thread.start()
    
    def run_environment_analysis(self):
        self.run_script_in_thread("detect_env_data.py", "Zaganjam analizo okolja...")
    
    def run_gesture_control(self):
        self.run_script_in_thread("gesture_control_gui.py", "Zaganjam gestno kontrolo...")
    
    def run_driver_analysis(self):
        self.run_script_in_thread("drowsiness_detector.py", "Zaganjam analizo voznika...")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = MainApplication(root)
        root.mainloop()
    except Exception as e:
        import traceback
        traceback.print_exc()