import os
import cv2
import mediapipe as mp


class MediaPipeProcessor:
    def __init__(self, input_folder, output_folder):
        """
        Initialize the MediaPipeProcessor with input and output folders.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.create_output_folders()

    def create_output_folders(self):
        """
        Ensure that output subfolders for each ASL letter (a, b, c, d, e) exist.
        """
        for folder_name in ['a', 'b', 'c', 'd', 'e']:
            folder_path = os.path.join(self.output_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)

    def process_image(self, image_path, output_path):
        """
        Process a single image using MediaPipe Hands and save the result.
        """
        hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_path}")
            return

        # Convert to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Draw landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Save the processed image
        cv2.imwrite(output_path, image)
        print(f"Processed and saved: {output_path}")

        hands.close()

    def process_images(self):
        """
        Process all images in the input folder and save to the output folder.
        """
        for folder_name in ['a', 'b', 'c', 'd', 'e']:
            input_folder_path = os.path.join(self.input_folder, folder_name)
            output_folder_path = os.path.join(self.output_folder, folder_name)

            if not os.path.exists(input_folder_path):
                print(f"Folder '{folder_name}' does not exist in input folder. Skipping...")
                continue

            print(f"Processing images in folder: {folder_name}")
            for file_name in os.listdir(input_folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    input_file_path = os.path.join(input_folder_path, file_name)
                    output_file_path = os.path.join(output_folder_path, file_name)
                    self.process_image(input_file_path, output_file_path)


# Usage
if __name__ == "__main__":
    input_folder = r"C:\Users\tatum\OneDrive\Fall 2024\STG 451\Sprint 5\Images\abcde"
    output_folder = r"C:\Users\tatum\OneDrive\Fall 2024\STG 451\Sprint 5\Images\preprocessed_images"
    processor = MediaPipeProcessor(input_folder, output_folder)
    processor.process_images()
