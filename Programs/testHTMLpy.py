import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from gtts import gTTS
import os
import time
from flask import Flask
from flask_socketio import SocketIO
from collections import deque

# Initialize Flask WebSocket Server
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.85)
mp_drawing = mp.solutions.drawing_utils

# Load ASL Models
static_model = tf.keras.models.load_model(r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\Models\Updated.h5")
  # Static letters A-Y
dynamic_j_model = tf.keras.models.load_model(r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\DynamicJZModel.h5")  # Motion-based J/Z model

# Define Labels
static_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
dynamic_labels = ["J", "Z"]

# Motion Buffer (Stores Last 20 Frames)
SEQUENCE_LENGTH = 20
motion_buffer = deque(maxlen=SEQUENCE_LENGTH)

# Function to extract hand landmarks
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return landmarks  # Return first detected hand

    return None  # No landmarks detected

# Function to detect motion
def detect_motion():
    if len(motion_buffer) < SEQUENCE_LENGTH:
        return False
    avg_movement = np.mean(np.abs(np.diff(motion_buffer, axis=0)))
    return avg_movement > 0.02  # Adjust threshold if needed

# Function to predict static letters (A-Y)
def predict_static(landmarks):
    prediction = static_model.predict(np.array(landmarks).reshape(1, -1))
    max_index = np.argmax(prediction)
    return static_labels[max_index], prediction[0][max_index]

# Function to predict dynamic J/Z letters
def predict_dynamic_j():
    sequence = np.array(motion_buffer).reshape(1, SEQUENCE_LENGTH, 63)
    prediction = dynamic_j_model.predict(sequence)
    max_index = np.argmax(prediction)
    return dynamic_labels[max_index], prediction[0][max_index]

# Function to announce detected letters
def announce_letter(letter):
    tts = gTTS(text=letter, lang='en', slow=False)
    tts.save("letter.mp3")
    os.system("afplay letter.mp3")  # macOS (Use 'mpg123 letter.mp3' on Windows/Linux)

# WebSocket event to send detected letters
@socketio.on("connect")
def handle_connect():
    print("Client connected!")

# Function to run hand-tracking and send real-time updates
def run_hand_tracking():
    cam = cv2.VideoCapture(0)
    last_logged_letter = None

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = extract_landmarks(frame)
                if landmarks:
                    motion_buffer.append(landmarks)

                    # Detect motion and classify letter
                    if detect_motion():
                        recognized_letter, confidence = predict_dynamic_j()
                    else:
                        recognized_letter, confidence = predict_static(landmarks)

                    # Emit detected letter to clients via WebSocket
                    if recognized_letter != last_logged_letter and confidence > 0.75:
                        print(f"Sending to WebSocket: {recognized_letter}")
                        socketio.emit("asl_translation", {"letter": recognized_letter})
                        announce_letter(recognized_letter)
                        last_logged_letter = recognized_letter

                    # Overlay recognized letter
                    cv2.putText(frame, f"Recognized: {recognized_letter} ({confidence:.2f})",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

     #  cv2.imshow('ASL Translator', frame)
        pass  # Do nothing
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# Start tracking in a background thread
@socketio.on("start_tracking")
def start_tracking():
    socketio.start_background_task(run_hand_tracking)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
