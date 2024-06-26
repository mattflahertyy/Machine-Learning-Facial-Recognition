import pandas as pd
from sklearn.model_selection import train_test_split

# Load the combined CSV file
all_data = pd.read_csv('all.csv')

# Define the target columns for stratification
target_columns = ['label_num_class', 'label_num_gender', 'label_num_race']

# Split the data into train (70%) and temp (30%)
train_data, temp_data = train_test_split(all_data, test_size=0.3, stratify=all_data[target_columns], random_state=42)

# Split the temp data into validation (50% of 30% = 15%) and test (50% of 30% = 15%)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data[target_columns], random_state=42)

# Save the new splits to CSV files
train_data.to_csv('train_augmented.csv', index=False)
validation_data.to_csv('validation_augmented.csv', index=False)
test_data.to_csv('test_augmented.csv', index=False)

print("Data has been split into train_augmented.csv, validation_augmented.csv, and test_augmented.csv.")
