import cv2
import numpy as np
import os

source_dir = 'python/TechVariable/data'

output_dir = 'python/TechVariable/pprset'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def preprocess_images(letter_dir, output_letter_dir):
    for image_file in os.listdir(letter_dir):
        image_path = os.path.join(letter_dir, image_file)
        image = cv2.imread(image_path)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        resized_image = cv2.resize(gray_image, (64, 64))

        output_path = os.path.join(output_letter_dir, image_file)  # Update output_path
        cv2.imwrite(output_path, resized_image)

for letter in range(ord('A'), ord('Z')+1):
    letter_folder = chr(letter)
    letter_dir = os.path.join(source_dir, letter_folder)
    
    if os.path.exists(letter_dir):
        output_letter_dir = os.path.join(output_dir, letter_folder)
        if not os.path.exists(output_letter_dir):
            os.makedirs(output_letter_dir)
        
        print(f"Processing images for letter {letter_folder}")
        preprocess_images(letter_dir, output_letter_dir)  # Pass output_letter_dir to the function
    else:
        print(f"Letter {letter_folder} not found in the source directory.")

print("Preprocessing completed.")
