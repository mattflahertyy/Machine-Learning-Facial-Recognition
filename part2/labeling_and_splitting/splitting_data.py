import os
import json
from sklearn.model_selection import train_test_split


def load_labeling_info(json_file):
    with open(json_file, 'r') as file:
        labeling_info = json.load(file)
    return labeling_info['images'], labeling_info['labels']


# save data to json files
def save_to_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)

if __name__ == "__main__":
    json_file = 'labeling_info_final.json'
    images, labels = load_labeling_info(json_file)

    # split the dataset into training, validation, and test sets
    train_images, temp_images, train_labels, temp_labels = train_test_split(images, labels, test_size=0.3,
                                                                            random_state=42)
    validation_images, test_images, validation_labels, test_labels = train_test_split(temp_images, temp_labels,
                                                                                      test_size=0.5, random_state=42)

    # print the sizes of each set
    print(f"Training set size: {len(train_images)}")
    print(f"Validation set size: {len(validation_images)}")
    print(f"Test set size: {len(test_images)}")

    # specify filenames
    train_data_file = '../model_json_files/train_data_final.json'
    validation_data_file = '../model_json_files/validation_data_final.json'
    test_data_file = '../model_json_files/test_data_final.json'

    # save data to json files
    save_to_json({'images': train_images, 'labels': train_labels}, train_data_file)
    save_to_json({'images': validation_images, 'labels': validation_labels}, validation_data_file)
    save_to_json({'images': test_images, 'labels': test_labels}, test_data_file)





