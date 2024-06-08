import json
import csv
import os

# Define label mappings
label_mappings = {"angry": 1, "focused": 2, "happy": 3, "neutral": 4}


# Function to create CSV from JSON
def json_to_csv(json_file, csv_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    images = data['images']
    labels = data['labels']

    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'image_path', 'label_name', 'label_num']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for img_path, label_num in zip(images, labels):
            image_name = os.path.basename(img_path)
            label_name = [name for name, num in label_mappings.items() if num == label_num][0]
            writer.writerow(
                {'image_name': image_name, 'image_path': img_path, 'label_name': label_name, 'label_num': label_num})


# Directories
json_dir = 'model_json_files/'
csv_dir = 'csv_split_data/'

# Ensure the csv directory exists
os.makedirs(csv_dir, exist_ok=True)

# Convert JSON files to CSV files
json_to_csv(os.path.join(json_dir, 'train_data.json'), os.path.join(csv_dir, 'train_data.csv'))
json_to_csv(os.path.join(json_dir, 'validation_data.json'), os.path.join(csv_dir, 'validation_data.csv'))
json_to_csv(os.path.join(json_dir, 'test_data.json'), os.path.join(csv_dir, 'test_data.csv'))