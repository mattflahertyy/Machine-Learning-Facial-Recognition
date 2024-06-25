import os
import csv

# this file renames all images in the data_augmented/ then adds the labels to a new csv

base_dir = '../data_augmentation'

races = ['hispanic', 'asian', 'black']
genders = ['male', 'female']
emotions = ['angry', 'focused', 'happy', 'neutral']

emotion_labels = {'angry': 0, 'focused': 1, 'happy': 2, 'neutral': 3}
gender_labels = {'male': 0, 'female': 1}
race_labels = {'black': 1, 'asian': 2, 'hispanic': 3}

csv_file = 'augmented_labels.csv'

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['image_name', 'image_path', 'label_name_class', 'label_num_class', 'label_name_gender', 'label_num_gender',
         'label_name_race', 'label_num_race'])

global_count = 1000


def process_images():
    global global_count

    for race in races:
        for gender in genders:
            for emotion in emotions:
                dir_path = os.path.join(base_dir, race, gender, emotion)

                image_files = [file for file in os.listdir(dir_path) if
                               file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))]

                for image_file in image_files:
                    old_path = os.path.join(dir_path, image_file)
                    new_image_name = f"{emotion}___{global_count}{os.path.splitext(image_file)[1]}"
                    new_path = os.path.join(dir_path, new_image_name)

                    os.rename(old_path, new_path)

                    image_name = new_image_name
                    image_path = f"../{os.path.join(base_dir, race, gender, emotion, new_image_name)}"
                    label_name_class = emotion
                    label_num_class = emotion_labels[emotion]
                    label_name_gender = gender
                    label_num_gender = gender_labels[gender]
                    label_name_race = race
                    label_num_race = race_labels[race]

                    with open(csv_file, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([image_name, image_path, label_name_class, label_num_class, label_name_gender,
                                         label_num_gender, label_name_race, label_num_race])

                    global_count += 1


process_images()

print("Image renaming and CSV updating completed.")
