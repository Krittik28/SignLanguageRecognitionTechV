from PIL import Image
import os

directory_path = "python/handsignPOC/poc2"

gesture_images = {}
for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    image_path = os.path.join(directory_path, f'{letter}.jpg')
    if os.path.isfile(image_path):
        gesture_images[letter] = Image.open(image_path)
    else:
        print(f"Image not found for letter {letter}")

input_letter = input("Enter a letter: ").upper()
if input_letter in gesture_images:
    gesture_image = gesture_images[input_letter]
    gesture_image.show()
else:
    print("Invalid letter.")
