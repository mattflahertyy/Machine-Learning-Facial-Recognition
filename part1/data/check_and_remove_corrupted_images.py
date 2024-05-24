import os
from PIL import Image

# this method removes all images that cannot be opened (corrupted)
def check_and_remove_corrupted_images(folder):
    corrupted_files = []
    for filename in os.listdir(folder):
        try:
            with Image.open(os.path.join(folder, filename)) as img:
                img.verify()
        except (IOError, SyntaxError) as e:
            corrupted_files.append(filename)

    for filename in corrupted_files:
        os.remove(os.path.join(folder, filename))
    print(f'Removed {len(corrupted_files)} corrupted images.')


if __name__ == "__main__":
    folder = "angry"
    check_and_remove_corrupted_images(folder)
