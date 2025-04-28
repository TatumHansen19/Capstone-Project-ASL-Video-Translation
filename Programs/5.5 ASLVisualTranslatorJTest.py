import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from gtts import gTTS
import os
import time
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.85)
mp_drawing = mp.solutions.drawing_utils

# Load Static and Dynamic Models
static_model = tf.keras.models.load_model(r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\Models\Updated.h5")
dynamic_j_model = tf.keras.models.load_model(r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\DynamicJZModel.h5")

# Define Labels
static_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
dynamic_labels = ["J", "Z"]

# Buffers
SEQUENCE_LENGTH = 20
motion_buffer = deque(maxlen=SEQUENCE_LENGTH)
stable_letter_buffer = deque(maxlen=5)
captured_letters = []  # Store captured letters

# Logging
def log_recognized_letter(letter, confidence):
    with open("recognized_letters.txt", "a") as file:
        file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Detected Letter: {letter} (Confidence: {confidence:.2f})\n")

def announce_text(text):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save("spoken.mp3")
    os.system("afplay spoken.mp3")  # Use appropriate player for your OS

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return landmarks, hand_landmarks
    return None, None

def detect_motion():
    if len(motion_buffer) < SEQUENCE_LENGTH:
        return False
    avg_movement = np.mean(np.abs(np.diff(motion_buffer, axis=0)))
    return avg_movement > 0.02

def predict_static(landmarks):
    prediction = static_model.predict(np.array(landmarks).reshape(1, -1))
    max_index = np.argmax(prediction)
    return static_labels[max_index], prediction[0][max_index]

def predict_dynamic_j():
    sequence = np.array(motion_buffer).reshape(1, SEQUENCE_LENGTH, 63)
    prediction = dynamic_j_model.predict(sequence)
    max_index = np.argmax(prediction)
    return dynamic_labels[max_index], prediction[0][max_index]

def draw_capture_animation(frame, bbox):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
    cv2.putText(frame, "Captured!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def run_hand_tracking():
    cam = cv2.VideoCapture(0)
    last_logged_letter = None
    last_announced_letter = None
    last_capture_time = time.time()
    capture_cooldown = 1.5  # seconds between valid captures
    capture_timer = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        capture_area = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
                capture_area = (x_min, y_min, x_max - x_min, y_max - y_min)

                landmarks, _ = extract_landmarks(frame)
                if landmarks:
                    motion_buffer.append(landmarks)

                    if detect_motion():
                        letter, confidence = predict_dynamic_j()
                    else:
                        letter, confidence = predict_static(landmarks)

                    # Confidence threshold
                    if confidence > 0.8:
                        stable_letter_buffer.append(letter)

                        if len(stable_letter_buffer) == stable_letter_buffer.maxlen and all(
                            l == stable_letter_buffer[0] for l in stable_letter_buffer):
                            
                            current_time = time.time()
                            if stable_letter_buffer[0] != last_logged_letter and (current_time - last_capture_time > capture_cooldown):
                                captured_letters.append(stable_letter_buffer[0])
                                log_recognized_letter(stable_letter_buffer[0], confidence)
                                last_logged_letter = stable_letter_buffer[0]
                                last_capture_time = current_time
                                capture_timer = 10

                    # Display current letter and confidence
                    cv2.putText(
                        frame,
                        f"{letter} ({confidence:.2f})",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )

        # Draw capture animation
        if capture_timer > 0 and capture_area:
            draw_capture_animation(frame, capture_area)
            capture_timer -= 1

        # Show captured letter sequence
        captured_text = ''.join(captured_letters[-20:])
        cv2.putText(frame, f"Captured: {captured_text}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("ASL Translator", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s') and captured_letters:
            announce_text(''.join(captured_letters))

    cam.release()
    cv2.destroyAllWindows()

run_hand_tracking()
