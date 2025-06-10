import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

# Settings
DATA_DIR = 'data_mirrors'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

recording = False
sequence = []

print("[INFO] Press 'r' to start/stop recording, 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[ERROR] Failed to capture image")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # If recording, save landmarks
    if recording and results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        sequence.append(landmarks)

    # Show status
    status_text = f"Recording: {'Yes' if recording else 'No'}"
    cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0) if recording else (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Gesture Recorder', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        if not recording:
            recording = True
            sequence = []
            print("[INFO] Started recording...")
        else:
            recording = False
            if sequence:  # if something was recorded
                print("[INFO] Recording stopped. Please enter label:")
                label = input(">> ").strip()

                label_dir = os.path.join(DATA_DIR, label)
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir)

                sequence_array = np.array(sequence)
                seq_filename = os.path.join(label_dir, f"{label}_{len(os.listdir(label_dir))}.npy")
                np.save(seq_filename, sequence_array)

                print(f"[INFO] Saved sequence: {seq_filename}")
                sequence = []

    elif key == ord('q'):
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
