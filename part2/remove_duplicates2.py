import os
import hashlib

# this method removes all images with duplicate content. it computes the MD5 hash of each image and removes the most similar
def remove_duplicates(folder):
    hash_keys = {}  # dictionary to store MD5 hashes and corresponding filenames
    duplicates = []  # list to store duplicate filenames

    for filename in os.listdir(folder):  # iterate through files in the folder
        if os.path.isfile(os.path.join(folder, filename)):  # check if it's a file
            with open(os.path.join(folder, filename), 'rb') as f:  # open file in binary mode
                filehash = hashlib.md5(f.read()).hexdigest()  # compute MD5 hash of the file
            if filehash not in hash_keys:  # if hash is not already in dictionary
                hash_keys[filehash] = filename  # add hash and filename to dictionary
            else:
                duplicates.append(filename)  # add filename to duplicates list if hash already exists

    for filename in duplicates:  # loop through duplicate filenames
        os.remove(os.path.join(folder, filename))  # remove duplicate file
    print(f'Removed {len(duplicates)} duplicate images.')  # print number of duplicates removed


if __name__ == "__main__":
    folder = "data_new/neutral"  # specify the folder to check
    remove_duplicates(folder)  # call the function to remove duplicates
