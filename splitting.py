import os
import random
from pathlib import Path

source_dir = 'python/TechVariable/data'

output_dir = 'python/TechVariable/main_dataset'
training_dir = Path(output_dir, 'training')
testing_dir = Path(output_dir, 'testing')

if not Path(output_dir).exists():
    Path(output_dir).mkdir(parents=True)

def preprocess_images(letter_dir, output_letter_dir, split_ratio=0.8):
    images = os.listdir(letter_dir)
    random.shuffle(images)
    num_images = len(images)
    split_index = int(split_ratio * num_images)

    training_images = images[:split_index]
    testing_images = images[split_index:]

    for image_file in training_images:
        image_path = Path(letter_dir, image_file)
        output_path = Path(output_letter_dir, 'training', image_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_path.read_bytes())

    for image_file in testing_images:
        image_path = Path(letter_dir, image_file)
        output_path = Path(output_letter_dir, 'testing', image_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_path.read_bytes())

for letter in range(ord('A'), ord('Z')+1):
    letter_folder = chr(letter)
    letter_dir = Path(source_dir, letter_folder)
    
    if letter_dir.exists():
        output_letter_dir = Path(output_dir, letter_folder)
        output_letter_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing images for letter {letter_folder}")
        preprocess_images(letter_dir, output_letter_dir)
    else:
        print(f"Letter {letter_folder} not found in the source directory.")

print("Splitting into training and testing sets completed.")
