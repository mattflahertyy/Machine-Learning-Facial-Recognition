import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Ensure you have the correct path to the files
data_path = os.path.dirname(os.path.abspath(__file__))

# Load and aggregate pixel intensities
def load_and_aggregate_pixel_intensities(data_path, class_dir):
    class_path = os.path.join(data_path, class_dir)
    pixel_intensities = {'red': [], 'green': [], 'blue': [], 'gray': []}
    if os.path.isdir(class_path):
        file_names = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        for file_path in file_names:
            img = cv2.imread(file_path)
            if img is not None:
                if len(img.shape) == 2:  # Grayscale image
                    pixel_intensities['gray'].extend(img.flatten())
                else:  # Color image
                    b, g, r = cv2.split(img)
                    pixel_intensities['red'].extend(r.flatten())
                    pixel_intensities['green'].extend(g.flatten())
                    pixel_intensities['blue'].extend(b.flatten())
            else:
                print(f"Warning: Unable to read {file_path}")
    return pixel_intensities

# Display pixel intensity distributions for the selected class
def display_class_intensity_distribution(class_label):
    if class_label in ['neutral', 'happy', 'focused', 'angry']:
        print(f"Loading and aggregating pixel intensities for class '{class_label}'...")
        intensities = load_and_aggregate_pixel_intensities(data_path, class_label)
        print(f"Plotting pixel intensity distribution for class '{class_label}'...")
        plt.figure(figsize=(12, 8))
        if intensities['gray']:
            plt.hist(intensities['gray'], bins=256, range=(0, 256), color='gray', alpha=0.5, label='Gray')
        if intensities['red'] or intensities['green'] or intensities['blue']:
            plt.hist(intensities['red'], bins=256, range=(0, 256), color='red', alpha=0.5, label='Red')
            plt.hist(intensities['green'], bins=256, range=(0, 256), color='green', alpha=0.5, label='Green')
            plt.hist(intensities['blue'], bins=256, range=(0, 256), color='blue', alpha=0.5, label='Blue')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title(f'Pixel Intensity Distribution for {class_label}')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Class '{class_label}' not found. Please choose from 'neutral', 'happy', 'focused', 'angry'.")

# Change the below function to either 'neutral', 'happy', 'focused', or 'angry'
display_class_intensity_distribution('neutral')