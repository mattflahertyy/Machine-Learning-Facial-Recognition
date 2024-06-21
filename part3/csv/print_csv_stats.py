import pandas as pd

# Define the paths to your CSV files
csv_files = ['test.csv',
             'train.csv',
             'validation.csv']

# Initialize counters for races, genders, and class types
race_counts = {}
gender_counts = {}
class_counts = {}


# Function to update the counters
def update_counts(df):
    global race_counts, gender_counts, class_counts
    races = df['label_name_race'].value_counts().to_dict()
    genders = df['label_name_gender'].value_counts().to_dict()
    classes = df['label_name_class'].value_counts().to_dict()

    for race, count in races.items():
        race_counts[race] = race_counts.get(race, 0) + count

    for gender, count in genders.items():
        gender_counts[gender] = gender_counts.get(gender, 0) + count

    for class_type, count in classes.items():
        class_counts[class_type] = class_counts.get(class_type, 0) + count


# Process each CSV file
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    update_counts(df)

# Print the results
print("Race Counts:")
for race, count in race_counts.items():
    print(f"{race}: {count}")

print("\nGender Counts:")
for gender, count in gender_counts.items():
    print(f"{gender}: {count}")

print("\nClass Counts:")
for class_type, count in class_counts.items():
    print(f"{class_type}: {count}")
