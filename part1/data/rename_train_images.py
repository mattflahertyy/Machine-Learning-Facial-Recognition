import os

# function to rename images in a folder based on the class name
def rename_images_in_folder(folder, class_name):
    images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]  # list of images in the folder

    for idx, img in enumerate(images):  # iterate through images
        extension = os.path.splitext(img)[1]  # get file extension
        new_name = f"{class_name}_{idx + 1}{extension}"  # generate new name with class name and index
        os.rename(os.path.join(folder, img), os.path.join(folder, new_name))  # rename the image

    print(f"Renamed {len(images)} images in {folder}")  # print number of images renamed

if __name__ == "__main__":
    base_folder = "test_train_data"  # specify the base folder

    classes = ["happy", "angry", "neutral", "focused"]  # list of classes

    for cls in classes:  # iterate through classes
        train_folder = os.path.join(base_folder, f"{cls}_train")  # generate path to class-specific train folder
        rename_images_in_folder(train_folder, cls)  # call function to rename images in the folder
