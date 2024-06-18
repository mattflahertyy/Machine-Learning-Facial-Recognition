from PIL import Image, ImageEnhance
import os
import numpy as np

# given a threshold and brightness decrease factor, this file goes through all images in a directory and decreases
# the brightness if it is above the threshold
class ImageProcessor:
    def __init__(self, directory, brightness_threshold=180, brightness_decrease_factor=0.9):
        self.directory = directory
        self.brightness_threshold = brightness_threshold
        self.brightness_decrease_factor = brightness_decrease_factor

    def calculate_brightness(self, image):
        grayscale_image = image.convert('L')
        np_image = np.array(grayscale_image)
        brightness = np.mean(np_image)
        return brightness

    def decrease_brightness(self, image, factor=0.9):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def process_images(self):
        updated_count = 0
        for filename in os.listdir(self.directory):
            if filename.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff')):
                file_path = os.path.join(self.directory, filename)
                try:
                    with Image.open(file_path) as img:
                        brightness = self.calculate_brightness(img)
                        print(f"Brightness of {filename}: {brightness:.2f}")
                        if brightness > self.brightness_threshold:
                            print(f"Decreasing brightness of {filename} (brightness: {brightness:.2f})")
                            darkened_image = self.decrease_brightness(img, self.brightness_decrease_factor)
                            darkened_image.save(file_path)
                            updated_count += 1
                        else:
                            print(f"{filename} is sufficiently dark (brightness: {brightness:.2f})")
                except Exception as e:
                    print(f"Could not process {filename}: {e}")

        print(f"Total images updated: {updated_count}")


if __name__ == "__main__":
    directory = "data_new/angry"
    processor = ImageProcessor(directory)
    processor.process_images()
