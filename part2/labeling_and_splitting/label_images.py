import os
import json


# function to label images based on folder names
def label_images(data_dir):
    classes = sorted(os.listdir(data_dir))
    labels = {}
    images = []
    image_labels = []

    for idx, cls in enumerate(classes):
        if cls == ".DS_Store":
            continue
        labels[cls] = idx
        class_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            images.append(img_path)
            image_labels.append(idx)

    return images, image_labels, labels


if __name__ == "__main__":
    data_dir = "../data_new"

    images, labels, class_labels = label_images(data_dir)

    print(f"Total images: {len(images)}")
    print(f"Total labels: {len(labels)}")

    # store labeling information in a json file
    labeling_info = {'images': images, 'labels': labels, 'class_labels': class_labels}
    with open('labeling_info_final.json', 'w') as json_file:
        json.dump(labeling_info, json_file)
