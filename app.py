from flask import Flask, render_template
from flask_socketio import SocketIO
import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import base64
import time
from collections import deque
from gtts import gTTS
import pygame

app = Flask(__name__, template_folder=r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\templates")
socketio = SocketIO(app, cors_allowed_origins="*")

# MediaPipe Hands + Models
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

STATIC_MODEL_PATH = r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\Models\Updated.h5"
DYNAMIC_J_MODEL_PATH = r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\DynamicJZModel.h5"

if not os.path.exists(STATIC_MODEL_PATH) or not os.path.exists(DYNAMIC_J_MODEL_PATH):
    raise FileNotFoundError("Model files not found.")

static_model = tf.keras.models.load_model(STATIC_MODEL_PATH)
dynamic_j_model = tf.keras.models.load_model(DYNAMIC_J_MODEL_PATH)

static_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
dynamic_labels = ["J", "Z"]

SEQUENCE_LENGTH = 20
motion_buffer = deque(maxlen=SEQUENCE_LENGTH)
stable_letter_buffer = deque(maxlen=5)
captured_letters = []
last_logged_letter = None
last_capture_time = time.time()
capture_cooldown = 1.0

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return landmarks
    return None

def detect_motion():
    if len(motion_buffer) < SEQUENCE_LENGTH:
        return False
    avg_movement = np.mean(np.abs(np.diff(motion_buffer, axis=0)))
    return avg_movement > 0.02

def predict_static(landmarks):
    prediction = static_model.predict(np.array(landmarks).reshape(1, -1))
    max_index = np.argmax(prediction)
    return static_labels[max_index], float(prediction[0][max_index])

def predict_dynamic_j():
    sequence = np.array(motion_buffer).reshape(1, SEQUENCE_LENGTH, 63)
    prediction = dynamic_j_model.predict(sequence)
    max_index = np.argmax(prediction)
    return dynamic_labels[max_index], float(prediction[0][max_index])

@socketio.on("video_frame")
def handle_frame(data):
    global last_logged_letter, last_capture_time

    img_data = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return

    landmarks = extract_landmarks(frame)

    # Only run prediction if a hand is detected
    if landmarks:
        motion_buffer.append(landmarks)

        # Default to static prediction
        if detect_motion():
            letter, confidence = predict_dynamic_j()
        else:
            letter, confidence = predict_static(landmarks)

        # Stability check before confirming
        if confidence > 0.70:
            stable_letter_buffer.append(letter)
            if len(stable_letter_buffer) == stable_letter_buffer.maxlen and all(
                l == stable_letter_buffer[0] for l in stable_letter_buffer):
                current_time = time.time()
                if letter != last_logged_letter and (current_time - last_capture_time > capture_cooldown):
                    captured_letters.append(letter)
                    last_logged_letter = letter
                    last_capture_time = current_time

        # ✨ Send result to UI
        socketio.emit("handshape_result", {
            "letter": letter,
            "confidence": float(confidence)
        })

        socketio.emit("captured_sequence", {
            "sequence": ''.join(captured_letters[-20:])
        })

@socketio.on("speak_sequence")
def speak_sequence():
    if captured_letters:
        text = ''.join(captured_letters)
        print(f"Speaking: {text}")
        tts = gTTS(text=text, lang='en', slow=False)

        filename = f"spoken_{int(time.time())}.mp3"
        try:
            tts.save(filename)
            pygame.mixer.init()
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            os.remove(filename)
        except Exception as e:
            print(f"Audio playback failed: {e}")

@socketio.on("clear_sequence")
def clear_sequence():
    captured_letters.clear()
    print("🧹 Captured sequence cleared.")
    socketio.emit("captured_sequence", {
        "sequence": ''
    })

@socketio.on("remove_last_letter")
def remove_last_letter():
    if captured_letters:
        removed = captured_letters.pop()
        print(f"Removed last captured letter: {removed}")
        socketio.emit("captured_sequence", {
            "sequence": ''.join(captured_letters[-20:])
        })
    else:
        print("No letters to remove.")

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
