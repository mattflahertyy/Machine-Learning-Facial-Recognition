import os
from PIL import Image

# this method removes all images that cannot be opened (corrupted)
def check_and_remove_corrupted_images(folder):
    corrupted_files = []  # list to hold names of corrupted files
    for filename in os.listdir(folder):  # loop through files in folder
        try:
            with Image.open(os.path.join(folder, filename)) as img:  # try opening the image
                img.verify()  # verify image integrity
        except (IOError, SyntaxError) as e:  # if error occurs, file is corrupted
            corrupted_files.append(filename)  # add corrupted file to list

    for filename in corrupted_files:  # loop through corrupted files
        os.remove(os.path.join(folder, filename))  # remove corrupted file
    print(f'Removed {len(corrupted_files)} corrupted images.')  # print how many files removed

if __name__ == "__main__":
    folder = "data_new/neutral"  # specify the folder to check
    check_and_remove_corrupted_images(folder)  # call the function on the folder
