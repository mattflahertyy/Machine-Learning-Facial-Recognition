import os
import cv2
import random
import matplotlib.pyplot as plt

# Ensure you have the correct path to the files
data_path = os.path.dirname(os.path.abspath(__file__))

# Sample image loader function
def load_sample_images(data_path, num_samples=15):
    class_samples = {}
    for class_dir in ['neutral', 'happy', 'focused', 'angry']:
        class_path = os.path.join(data_path, class_dir)
        samples = []
        if os.path.isdir(class_path):
            file_names = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            if len(file_names) > num_samples:
                sample_file_names = random.sample(file_names, num_samples)
            else:
                sample_file_names = file_names
            for file_name in sample_file_names:
                file_path = os.path.join(class_path, file_name)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    samples.append(img)
                else:
                    print(f"Warning: Unable to read {file_path}")
        class_samples[class_dir] = samples
    return class_samples

# Load sample images
num_samples = 15 # 15 samples per class
class_samples = load_sample_images(data_path, num_samples)

rows, cols = 5, 6  # 5 rows and 6 columns (the histograms will be next to the sample images)

# Display sample images and the histograms for each class
for class_idx, (class_label, images) in enumerate(class_samples.items()):
    plt.figure(figsize=(20, 20))
    for img_idx, img in enumerate(images):
        img_pos = (img_idx // (cols // 2)) * cols + (img_idx % (cols // 2)) * 2 + 1
        plt.subplot(rows, cols, img_pos)
        plt.imshow(img, cmap='gray')
        plt.title(f"{class_label} {img_idx + 1}")
        plt.axis('off')
        
        hist_pos = img_pos + 1 # Calculate subplot position for histogram
        plt.subplot(rows, cols, hist_pos)
        plt.hist(img.flatten(), bins=256, range=(0, 256), color='gray', alpha=0.75)
        plt.title(f"Histogram of {class_label} {img_idx + 1}")
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.suptitle(f"Class: {class_label}", fontsize=16)
    plt.show()