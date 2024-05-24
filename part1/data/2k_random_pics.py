import os
import random
import shutil

def select_and_copy_images(source_folder, destination_folder, num_images):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    all_files = os.listdir(source_folder)

    all_files = [file for file in all_files if os.path.isfile(os.path.join(source_folder, file))]

    selected_files = random.sample(all_files, min(num_images, len(all_files)))

    for file_name in selected_files:
        src_file = os.path.join(source_folder, file_name)
        dst_file = os.path.join(destination_folder, file_name)
        shutil.copyfile(src_file, dst_file)

source_folder = os.path.join(os.getcwd(), 'angry')  # Path to your source folder
destination_folder = os.path.join(os.getcwd(), 'angry_2k')

select_and_copy_images(source_folder, destination_folder, 2000)
