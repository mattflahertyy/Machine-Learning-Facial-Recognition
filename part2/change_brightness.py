from PIL import Image, ImageEnhance
import os
import numpy as np

def calculate_brightness(image):
    """
    Calculate the RMS (root mean square) brightness of an image.
    """
    grayscale_image = image.convert('L')
    np_image = np.array(grayscale_image)
    brightness = np.sqrt(np.mean(np_image ** 2))
    return brightness

def increase_brightness(image, factor=1.2):
    """
    Increase the brightness of an image by a given factor.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def process_images(directory, brightness_threshold=100, brightness_increase_factor=1.2):
    """
    Process all images in a directory: increase brightness if below a certain threshold.
    """
    for filename in os.listdir(directory):
        if filename.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff')):
            file_path = os.path.join(directory, filename)
            with Image.open(file_path) as img:
                brightness = calculate_brightness(img)
                print(f"Brightness of {filename}: {brightness:.2f}")
                if brightness < brightness_threshold:
                    print(f"Increasing brightness of {filename} (brightness: {brightness:.2f})")
                    brightened_image = increase_brightness(img, brightness_increase_factor)
                    brightened_image.save(file_path)
                else:
                    print(f"{filename} is sufficiently bright (brightness: {brightness:.2f})")

if __name__ == "__main__":
    directory = "data2/happy"
    process_images(directory)
