import os
import shutil
import random

# function to split dataset into train and test folders
def split_dataset(source_folder, train_folder, test_folder, train_ratio=0.8):
    # create train and test folders if they don't exist
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # list all images in the source folder
    images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    random.shuffle(images)  # shuffle the list of images

    # calculate the number of images for training
    train_count = int(len(images) * train_ratio)

    # split images into train and test sets
    train_images = images[:train_count]
    test_images = images[train_count:]

    # copy train images to train folder
    for img in train_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(train_folder, img))

    # copy test images to test folder
    for img in test_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(test_folder, img))

    # print number of images copied to train and test folders
    print(f"Copied {len(train_images)} images to {train_folder}")
    print(f"Copied {len(test_images)} images to {test_folder}")


if __name__ == "__main__":
    base_folder = "."  # specify the base folder
    classes = ["happy", "angry", "neutral", "focused"]  # list of classes
    output_folder = os.path.join(base_folder, "test_train_data")  # specify the output folder

    # iterate through classes and split dataset
    for cls in classes:
        source = os.path.join(base_folder, cls)  # source folder for class images
        train_dest = os.path.join(output_folder, f"{cls}_train")  # train folder for class
        test_dest = os.path.join(output_folder, f"{cls}_test")  # test folder for class
        split_dataset(source, train_dest, test_dest, train_ratio=0.8)  # split dataset for the class
