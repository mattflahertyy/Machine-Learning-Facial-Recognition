import os
import cv2
import random
import matplotlib.pyplot as plt

data_path = os.path.dirname(os.path.abspath(__file__))

# Load random sample function
def load_sample_images(data_path, num_samples=5):
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
num_samples = 5  # Number of samples to display per class
class_samples = load_sample_images(data_path, num_samples)

# Display the sample images in a grid
plt.figure(figsize=(15, 20))

for class_idx, (class_label, images) in enumerate(class_samples.items()):
    for img_idx, img in enumerate(images):
        # Plot the sample image
        plt.subplot(len(class_samples) * 2, num_samples, class_idx * num_samples * 2 + img_idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"{class_label} {img_idx + 1}")
        plt.axis('off')
        
        # Plot the histogram of pixel intensities
        plt.subplot(len(class_samples) * 2, num_samples, class_idx * num_samples * 2 + num_samples + img_idx + 1)
        plt.hist(img.flatten(), bins=256, range=(0, 256), color='gray', alpha=0.75)
        plt.title(f"Histogram of {class_label} {img_idx + 1}")
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

plt.tight_layout()
plt.show()