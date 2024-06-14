import os
import json


# Function to label images based on folder names
def label_images(data_dir):
    classes = sorted(os.listdir(data_dir))  # Get the class folders in sorted order
    labels = {}  # Dictionary to hold class labels
    images = []  # List to store image paths
    image_labels = []  # List to store image labels

    for idx, cls in enumerate(classes):
        if cls == ".DS_Store":  # Skip the ".DS_Store" file
            continue
        labels[cls] = idx
        class_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(class_dir):  # Skip non-directory files
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            images.append(img_path)
            image_labels.append(idx)  # Assign numerical label based on class index

    return images, image_labels, labels


if __name__ == "__main__":
    data_dir = "../data2"  # Directory containing class folders

    images, labels, class_labels = label_images(data_dir)

    print(f"Total images: {len(images)}")
    print(f"Total labels: {len(labels)}")

    # Store labeling information in a JSON file
    labeling_info = {'images': images, 'labels': labels, 'class_labels': class_labels}
    with open('labeling_info2.json', 'w') as json_file:
        json.dump(labeling_info, json_file)
