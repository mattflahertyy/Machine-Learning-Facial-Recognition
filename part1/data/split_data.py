import os
import shutil
import random

def split_dataset(source_folder, train_folder, test_folder, train_ratio=0.8):
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    random.shuffle(images)

    train_count = int(len(images) * train_ratio)

    train_images = images[:train_count]
    test_images = images[train_count:]

    for img in train_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(train_folder, img))

    for img in test_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(test_folder, img))

    print(f"Copied {len(train_images)} images to {train_folder}")
    print(f"Copied {len(test_images)} images to {test_folder}")


if __name__ == "__main__":
    base_folder = "."
    classes = ["happy", "angry", "neutral", "focused"]
    output_folder = os.path.join(base_folder, "test_train_data")

    for cls in classes:
        source = os.path.join(base_folder, cls)
        train_dest = os.path.join(output_folder, f"{cls}_train")
        test_dest = os.path.join(output_folder, f"{cls}_test")
        split_dataset(source, train_dest, test_dest, train_ratio=0.8)
