import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
augmented_labels = pd.read_csv('part3/data_augmentation/augmented_labels.csv')
kfold = pd.read_csv('part3/csv/kfold.csv')

# Combine the data from both CSV files
combined_data = pd.concat([augmented_labels, kfold])

# Count the number of each class
class_counts = combined_data['label_name_class'].value_counts()

# Print the counts
print("Class distribution:")
print(class_counts)

# Plot the distribution
plt.figure(figsize=(10, 6))
ax = class_counts.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(rotation=0)

# Add count labels on top of each bar
for i in ax.containers:
    ax.bar_label(i)

plt.show()