import os
import hashlib


def remove_duplicates(folder):
    hash_keys = {}
    duplicates = []

    for filename in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, filename)):
            with open(os.path.join(folder, filename), 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash not in hash_keys:
                hash_keys[filehash] = filename
            else:
                duplicates.append(filename)

    for filename in duplicates:
        os.remove(os.path.join(folder, filename))
    print(f'Removed {len(duplicates)} duplicate images.')


if __name__ == "__main__":
    folder = "happy"
    remove_duplicates(folder)
