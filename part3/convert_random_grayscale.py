import os
import random
from PIL import Image

directory = 'data_augmentation/black/female/'

all_files = os.listdir(directory)

image_files = [file for file in all_files if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))]

num_to_convert = int(0.8 * len(image_files))

images_to_convert = random.sample(image_files, num_to_convert)

for image_file in images_to_convert:
    image_path = os.path.join(directory, image_file)
    with Image.open(image_path) as img:
        grayscale_img = img.convert('L')
        grayscale_img.save(image_path)

print(f"Converted {num_to_convert} images to grayscale in the directory: {directory}")
