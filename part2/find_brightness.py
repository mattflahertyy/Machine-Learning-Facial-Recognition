from PIL import Image
import os
import numpy as np


# this file calculates tje brightness of all images and lists them with their values in descending order
def calculate_brightness(image):
    grayscale_image = image.convert('L')
    np_image = np.array(grayscale_image)
    brightness = np.mean(np_image)
    return brightness


def process_images(directory):
    brightness_data = []

    # Calculate brightness for each image and store the results
    for filename in os.listdir(directory):
        if filename.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff')):
            file_path = os.path.join(directory, filename)
            with Image.open(file_path) as img:
                brightness = calculate_brightness(img)
                brightness_data.append((filename, brightness))

    # Sort images by brightness in descending order (brightest to darkest)
    brightness_data.sort(key=lambda x: x[1], reverse=True)

    # Print the sorted brightness data
    for filename, brightness in brightness_data:
        print(f"Brightness of {filename}: {brightness:.2f}")


if __name__ == "__main__":
    directory = "data_new/happy"
    process_images(directory)
