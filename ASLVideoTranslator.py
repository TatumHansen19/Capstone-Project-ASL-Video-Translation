import cv2
import mediapipe as mp
import tkinter as tk
from threading import Thread

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to update the tkinter text window
def update_text_window(text_widget, text_data):
    text_widget.config(state=tk.NORMAL)  # Enable editing the text widget
    text_widget.delete(1.0, tk.END)  # Clear previous text
    text_widget.insert(tk.END, text_data)  # Insert the new text
    text_widget.config(state=tk.DISABLED)  # Disable editing again

# Function to run the hand-tracking camera feed
def run_hand_tracking(text_widget):
    cam = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cam.read()

        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB as MediaPipe requires
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find hand landmarks and handedness
        result = hands.process(rgb_frame)

        text_data = ""  # Initialize text data for the tkinter window

        if result.multi_hand_landmarks and result.multi_handedness:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Draw hand landmarks on the original BGR frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the classification (handedness) of the current hand
                handedness = result.multi_handedness[idx].classification[0].label

                # Define the text to display ('Left' or 'Right')
                if handedness == 'Left':
                    hand_label = 'Left'
                else:
                    hand_label = 'Right'

                # Calculate the position to display the label near the wrist
                wrist_landmark = hand_landmarks.landmark[0]  # Wrist landmark
                wrist_x = int(wrist_landmark.x * frame.shape[1])
                wrist_y = int(wrist_landmark.y * frame.shape[0])

                # Display the hand label on the frame
                cv2.putText(frame, hand_label, (wrist_x - 20, wrist_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # List of landmarks corresponding to each finger tip
                finger_tips = [4, 8, 12, 16, 20]

                # Get landmarks for the hand
                landmarks = hand_landmarks.landmark

                # Count how many fingers are up
                finger_count = 0

                # Thumb (Tip is [4], joint [3])
                if landmarks[4].x < landmarks[3].x:  # Check thumb for right hand
                    finger_count += 1

                # Other fingers (Tips are [8, 12, 16, 20], joints are [6, 10, 14, 18])
                for tip in finger_tips[1:]:
                    # Check if the finger is up by comparing tip y-coordinate with the PIP joint
                    if landmarks[tip].y < landmarks[tip - 2].y:
                        finger_count += 1

                # Display the finger count on the frame
                cv2.putText(frame, f'Fingers: {finger_count}', (wrist_x - 20, wrist_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Add the hand label and finger count to the text data
                text_data += f"Hand: {hand_label}, Fingers: {finger_count}\n"

        # Update the tkinter text widget with the new data
        update_text_window(text_widget, text_data)

        # Display the frame
        cv2.imshow('Hand Tracking', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and close windows
    cam.release()
    cv2.destroyAllWindows()

# Function to create and run the tkinter window
def create_text_window():
    # Create the tkinter window
    root = tk.Tk()
    root.title("Hand Tracking Info")

    # Create a text widget to display hand tracking data
    text_widget = tk.Text(root, height=10, width=40)
    text_widget.pack()

    # Start the hand tracking in a separate thread so that the tkinter window remains responsive
    tracking_thread = Thread(target=run_hand_tracking, args=(text_widget,))
    tracking_thread.start()

    # Start the tkinter event loop
    root.mainloop()

# Run the tkinter text window
create_text_window()
