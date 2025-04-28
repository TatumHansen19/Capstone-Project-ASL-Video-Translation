import os
import time
import cv2

class ASLImageCollector:
    def __init__(self, base_folder):
        """
        Initialize the ASLImageCollector with the base folder for storing images.
        """
        self.base_folder = base_folder
        self.create_folders()

    def create_folders(self):
        """
        Ensure that subfolders for each ASL letter (a, b, c, d, e) exist.
        """
        for folder_name in ['a', 'b', 'c', 'd', 'e', 'f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']:
            folder_path = os.path.join(self.base_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)

    def save_image(self, image, label):
        """
        Save an image to the specified folder based on the label (a, b, c, d, e).
        """
        folder_path = os.path.join(self.base_folder, label)
        if not os.path.exists(folder_path):
            print(f"Error: Folder for label '{label}' does not exist.")
            return
        
        # Create a unique filename
        file_name = f"{label}_{len(os.listdir(folder_path)) + 1}.jpg"
        file_path = os.path.join(folder_path, file_name)

        # Save the image
        cv2.imwrite(file_path, image)
        print(f"Image saved to {file_path}")

    def collect_images_from_camera(self):
        """
        Open the webcam to collect images and direct them to the appropriate folder.
        """
        cap = cv2.VideoCapture(0)
        print("Press 'a', 'b', 'c', 'd', or 'e' to save an image for that letter.")
        print("Press 'q' to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image. Exiting...")
                break

            # Display the live feed
            cv2.imshow("ASL Image Collector", frame)

            # Wait for a key press
            key = cv2.waitKey(1) & 0xFF

            # Check if the key corresponds to a valid label
            if chr(key) in ['a', 'b', 'c', 'd', 'e', 'f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']:
                self.save_image(frame, chr(key))
            
            
            if chr(key) in ['j', 'z']:
                sequence_folder = os.path.join(self.base_folder, f"{chr(key)}_sequence")
                os.makedirs(sequence_folder, exist_ok=True)
                print(f"Capturing motion for letter {chr(key)}. Please move your hand slowly.")

                for i in range(20):
                    ret, frame = cap.read()  # Capture a new frame in each loop iteration
                    if not ret:
                        print("Failed to capture frame. Exiting sequence capture.")
                        break
                    file_name = f"{chr(key)}_{len(os.listdir(sequence_folder)) + 1}.jpg"
                    file_path = os.path.join(sequence_folder, file_name)
                    cv2.imwrite(file_path, frame)
                    print(f"Saved motion frame {i+1} for letter {chr(key)}")

                    cv2.imshow("Capturing Motion", frame)
                    if cv2.waitKey(100) & 0xFF == ord('2'):
                        break
                #cap.release()
                print(f"Motion capture for {chr(key)} complete.")

            elif key == ord('1'):
                print("Exiting...")
                break

        cap.release()
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    base_folder = r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\alphabet"
    collector = ASLImageCollector(base_folder)
    collector.collect_images_from_camera()
