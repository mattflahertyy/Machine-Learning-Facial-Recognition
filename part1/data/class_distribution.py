import os
import matplotlib.pyplot as plt

data_path = os.path.dirname(os.path.abspath(__file__))

# Data and image count function
def load_data_custom(data_path):
    class_labels = []
    class_counts = []
    for class_dir in ['neutral', 'happy', 'focused', 'angry']:
        class_path = os.path.join(data_path, class_dir)
        if os.path.isdir(class_path):
            num_images = len([name for name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, name))])
            class_labels.append(class_dir)
            class_counts.append(num_images)
    return class_labels, class_counts

# Load the data
class_labels, class_counts = load_data_custom(data_path)

# Plot the distribution
plt.figure(figsize=(10, 6))
bars = plt.bar(class_labels, class_counts, color='skyblue')

# Add the counts on top of each bar
for bar, count in zip(bars, class_counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha='center', va='bottom')

plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.title('Distribution of Images Across Different Classes')
plt.show()