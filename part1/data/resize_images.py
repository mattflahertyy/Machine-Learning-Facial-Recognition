import os
from PIL import Image

def resize_images(folder, target_size=(48, 48)):
    for filename in os.listdir(folder):
        try:
            # Open the image
            img = Image.open(os.path.join(folder, filename))
            # Resize the image to 48x48 pixels
            img = img.resize(target_size, resample=Image.BILINEAR)
            # Save the resized image, overwriting the original file
            img.save(os.path.join(folder, filename))
            # Close the image
            img.close()
        except Exception as e:
            print(f'Error resizing image {filename}: {e}')

if __name__ == "__main__":
    folder = "angry"
    resize_images(folder)
