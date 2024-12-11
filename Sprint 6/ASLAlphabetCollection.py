import cv2
import mediapipe as mp
import pandas as pd
import os

# Base path to the directory containing subdirectories 'a', 'b', 'c', 'd', 'e'
base_directory = r"C:\Users\tatum\OneDrive\Fall 2024\STG 451\Sprint 5\Images\preprocessed_images"

# Subdirectories to process
subdirectories = ['a', 'b', 'c', 'd', 'e']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3)  # Reduced confidence threshold
mp_drawing = mp.solutions.drawing_utils

# Create an empty dataframe to store the landmarks and labels
columns = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
data = pd.DataFrame(columns=columns)

# Function to process images in a directory
def process_images(label, directory):
    global data
    file_count = 0  # Track the number of files processed for the current label
    added_rows = 0  # Track the number of rows added for the current label

    for image_name in os.listdir(directory):
        image_path = os.path.join(directory, image_name)

        # Skip non-image files
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {image_name}")
            continue

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image {image_name}, skipping.")
            continue

        # Resize image to standard dimensions
        image = cv2.resize(image, (640, 480))

        # Convert the image to RGB as required by MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to find hand landmarks
        result = hands.process(rgb_image)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks on the image (for visualization purposes, optional)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Collect the landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)

                # Add the label and append to the DataFrame
                landmarks.append(label)
                data.loc[len(data)] = landmarks
                added_rows += 1

                print(f"Assigned data point to '{label}' from file: {image_name}. Total rows for '{label}': {added_rows}")
        else:
            print(f"No landmarks detected in image {image_name}, skipping.")

        file_count += 1

    print(f"Finished processing {file_count} files for label '{label}'. Total rows added: {added_rows}")

    cv2.destroyAllWindows()

# Process each subdirectory
for subdir in subdirectories:
    directory_path = os.path.join(base_directory, subdir)
    if os.path.exists(directory_path):
        print(f"Processing directory: {directory_path}")
        process_images(label=subdir, directory=directory_path)
    else:
        print(f"Directory {directory_path} does not exist, skipping.")

# Save the collected data to a CSV file
data.to_csv('gesture_data_from_images3.csv', index=False)
print("Data processing completed and saved to 'gesture_data_from_images3.csv'.")
