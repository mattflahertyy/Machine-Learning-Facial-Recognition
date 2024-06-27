import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure you have the correct path to the files
data_path = 'part3/data_new/'

# Load and aggregate pixel intensities
def load_and_aggregate_pixel_intensities(data_path, class_dir):
    class_path = os.path.join(data_path, class_dir)
    pixel_intensities = {'red': [], 'green': [], 'blue': [], 'gray': []}
    
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
    
    return pixel_intensities

# Display pixel intensity distributions for the selected class
def display_class_intensity_distribution(class_label):
    print(f"Loading and aggregating pixel intensities for class '{class_label}'...")
    intensities = load_and_aggregate_pixel_intensities(data_path, class_label)
    
    print(f"Red intensities: {len(intensities['red'])}")
    print(f"Green intensities: {len(intensities['green'])}")
    print(f"Blue intensities: {len(intensities['blue'])}")
    print(f"Gray intensities: {len(intensities['gray'])}")
    
    print(f"Plotting pixel intensity distribution for class '{class_label}'...")
    plt.figure(figsize=(12, 8))
    
    if intensities['gray']:
        plt.hist(intensities['gray'], bins=256, range=(0, 256), color='gray', alpha=0.5, label='Gray')
    if intensities['red']:
        plt.hist(intensities['red'], bins=256, range=(0, 256), color='red', alpha=0.5, label='Red')
    if intensities['green']:
        plt.hist(intensities['green'], bins=256, range=(0, 256), color='green', alpha=0.5, label='Green')
    if intensities['blue']:
        plt.hist(intensities['blue'], bins=256, range=(0, 256), color='blue', alpha=0.5, label='Blue')
    
    if any([intensities['gray'], intensities['red'], intensities['green'], intensities['blue']]):
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title(f'Pixel Intensity Distribution for {class_label}')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    else:
        print(f"No pixel intensities found for class '{class_label}'. Nothing to plot.")

# Change the below function to either 'neutral', 'happy', 'focused', or 'angry'
display_class_intensity_distribution('neutral')