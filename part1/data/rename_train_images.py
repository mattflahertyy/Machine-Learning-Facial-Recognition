import os

def rename_images_in_folder(folder, class_name):
    images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    for idx, img in enumerate(images):
        extension = os.path.splitext(img)[1]
        new_name = f"{class_name}_{idx + 1}{extension}"
        os.rename(os.path.join(folder, img), os.path.join(folder, new_name))

    print(f"Renamed {len(images)} images in {folder}")


if __name__ == "__main__":
    base_folder = "test_train_data"

    classes = ["happy", "angry", "neutral", "focused"]

    for cls in classes:
        train_folder = os.path.join(base_folder, f"{cls}_train")
        rename_images_in_folder(train_folder, cls)
