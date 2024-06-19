import os

# function to rename images in a folder based on the class name
def rename_images_in_folder(folder, class_name):
    images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]  # list of images in the folder

    for idx, img in enumerate(images):  # iterate through images
        extension = os.path.splitext(img)[1]  # get file extension
        new_name = f"{class_name}___{idx + 1}{extension}"  # generate new name with class name and index
        os.rename(os.path.join(folder, img), os.path.join(folder, new_name))  # rename the image

    print(f"Renamed {len(images)} images in {folder}")  # print number of images renamed

if __name__ == "__main__":
    base_folder = os.path.join(os.getcwd(), "../data_new")  # look for "data" folder in the current directory

    classes = ["angry"]  # specify the class folder to rename images (e.g., "happy")

    for cls in classes:  # iterate through classes
        class_folder = os.path.join(base_folder, cls)  # generate path to class-specific folder
        rename_images_in_folder(class_folder, cls)  # call function to rename images in the folder
