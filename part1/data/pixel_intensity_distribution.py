import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path definition
data_path = os.path.dirname(os.path.abspath(__file__))

# Function for images and pixel intensities
def load_and_aggregate_pixel_intensities(data_path):
    class_pixel_intensities = {}
    for class_dir in ['neutral', 'happy', 'focused', 'angry']:
        class_path = os.path.join(data_path, class_dir)
        pixel_intensities = []
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                if os.path.isfile(file_path):
                    # Grayscale image reader 
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        pixel_intensities.extend(img.flatten())
                    else:
                        print(f"Warning: Unable to read {file_path}")
        class_pixel_intensities[class_dir] = pixel_intensities
    return class_pixel_intensities

# Load and aggregate pixel intensities
class_pixel_intensities = load_and_aggregate_pixel_intensities(data_path)

# Plot the pixel intensity distributions
plt.figure(figsize=(12, 8))
for class_label, intensities in class_pixel_intensities.items():
    if intensities:
        # Plot in chunks to avoid memory issues
        chunk_size = 1000000
        for i in range(0, len(intensities), chunk_size):
            plt.hist(intensities[i:i + chunk_size], bins=256, range=(0, 256), alpha=0.5, label=class_label if i == 0 else "")

plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Pixel Intensity Distribution per Class')
plt.legend(loc='upper right')
plt.show()