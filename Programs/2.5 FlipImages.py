import cv2
import os

# Folder containing the images
input_folder = r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\preprocessed_images\c" 
output_folder = r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\preprocessed_images\c"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    # Check if the file is an image (you can add more extensions if needed)
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        # Construct the full file path
        input_image_path = os.path.join(input_folder, filename)
        
        # Read the image
        image = cv2.imread(input_image_path)
        
        # Flip the image horizontally
        flipped_image = cv2.flip(image, 1)  # 1 indicates horizontal flip
        
        # Save the flipped image to the output folder
        output_image_path = os.path.join(output_folder, f'flipped_{filename}')
        cv2.imwrite(output_image_path, flipped_image)
        
        # Optionally, print a message for each processed image
        print(f'Flipped {filename} and saved to {output_image_path}')

