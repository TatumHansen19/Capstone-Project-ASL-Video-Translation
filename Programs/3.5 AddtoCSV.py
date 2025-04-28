import cv2
import mediapipe as mp
import pandas as pd
import os

# Base path to the directory containing subdirectories 'a', 'b', 'c', 'd', 'e'
base_directory = r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\preprocessed_images"

# Subdirectories to process
subdirectories = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l','j', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v' ,'w', 'x', 'y']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# Columns for the CSV file
columns = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']

# Function to process images in a directory and append to CSV file
def process_images(label, directory, csv_file):
    file_count = 0  # Track the number of files processed for the current label
    added_rows = 0  # Track the number of rows added for the current label

    # Load existing data if the file exists
    if os.path.exists(csv_file):
        data = pd.read_csv(csv_file)
    else:
        data = pd.DataFrame(columns=columns)

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

    # Sort data alphabetically by the 'label' column
    if not data.empty:
        data = data.sort_values(by='label')
        data.to_csv(csv_file, index=False)
        print(f"Finished processing {file_count} files for label '{label}'. Total rows added: {added_rows}. Data sorted and saved to {csv_file}")
    else:
        print(f"No data to save for label '{label}'.")

    cv2.destroyAllWindows()

# Function to remove rows by label from the CSV file
def remove_label_from_csv(label, csv_file):
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} does not exist.")
        return

    data = pd.read_csv(csv_file)
    initial_count = len(data)

    # Filter out rows with the specified label
    data = data[data['label'] != label]
    final_count = len(data)

    # Save the updated CSV file
    data.to_csv(csv_file, index=False)
    print(f"Removed {initial_count - final_count} rows with label '{label}' from {csv_file}.")

# Function to display the letters in the CSV file and their counts
def display_csv_summary(csv_file):
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} does not exist.")
        return

    data = pd.read_csv(csv_file)
    summary = data['label'].value_counts().sort_index()

    print("\nSummary of letters in the CSV file:")
    for label, count in summary.items():
        print(f"Letter '{label}': {count} rows")

if __name__ == "__main__":
    while True:
        action = input("Would you like to add, remove, view summary, or exit? (add/remove/summary/exit): ").strip().lower()

        if action == "add":
            label = input("Which letter would you like to add?: ").strip().lower()
            if label in subdirectories:
                directory_path = os.path.join(base_directory, label)
                csv_file = r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\CSV Files\AMC.csv"

                if os.path.exists(directory_path):
                    print(f"Processing directory: {directory_path}")
                    process_images(label=label, directory=directory_path, csv_file=csv_file)
                else:
                    print(f"Directory {directory_path} does not exist, skipping.")
            else:
                print(f"Letter '{label}' is not in the predefined subdirectories list.")

        elif action == "remove":
            label = input("Which letter would you like to remove?: ").strip().lower()
            csv_file = r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\CSV Files\AMC.csv"
            remove_label_from_csv(label=label, csv_file=csv_file)

        elif action == "summary":
            csv_file = r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\CSV Files\AMC.csv"
            display_csv_summary(csv_file=csv_file)

        elif action == "exit":
            print("Exiting the program.")
            break

        else:
            print("Invalid action. Please type 'add', 'remove', 'summary', or 'exit'.")

        print("Task completed.") #
