import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from gtts import gTTS
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.85)
mp_drawing = mp.solutions.drawing_utils

# Load the trained TensorFlow model
model = tf.keras.models.load_model('trained_model.h5')

# Label Encoder (ensure this matches the classes used during training)
labels = ["A", "B", "C", "D", "E"]  # Update if more labels are added during training

# Function to log recognized letters to a text file
def log_recognized_letter(letter):
    with open("recognized_letters.txt", "a") as file:
        file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Detected Letter: {letter}\n")

# Function to announce the recognized letter using gTTS
def announce_letter(letter):
    tts = gTTS(text=letter, lang='en', slow=False)
    tts.save("letter.mp3")
    os.system("afplay letter.mp3")  # Use afplay for macOS

# Function to run the hand-tracking camera feed
def run_hand_tracking():
    cam = cv2.VideoCapture(0)
    last_logged_letter = None
    last_announced_letter = None

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Collect landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                # Convert the landmarks into a NumPy array and reshape for the model
                landmarks = np.array(landmarks).reshape(1, -1)

                # Make prediction with the trained model
                prediction = model.predict(landmarks)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]

                # Debugging: Print prediction details
                print(f"Prediction: {prediction}, Class: {predicted_class}, Confidence: {confidence:.2f}")

                # Get the recognized letter
                recognized_letter = labels[predicted_class]

                # If confidence is above 70% and the letter is different from the last logged letter
                if confidence >= 0.7 and recognized_letter != last_logged_letter:
                    log_recognized_letter(recognized_letter)
                    last_logged_letter = recognized_letter

                # If confidence is above 70% and the letter is different from the last announced letter
                if confidence >= 0.7 and recognized_letter != last_announced_letter:
                    announce_letter(recognized_letter)  # Announce the recognized letter
                    last_announced_letter = recognized_letter

                # Overlay recognized letter on the frame
                cv2.putText(
                    frame,
                    f"Recognized Letter: {recognized_letter} (Confidence: {confidence:.2f})",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# Run the hand tracking
run_hand_tracking()