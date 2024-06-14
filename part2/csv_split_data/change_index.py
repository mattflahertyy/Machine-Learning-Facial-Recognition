import pandas as pd


# this function changes the index of the labels to 1 less (from 1 to 4 based to 0 to 3)
def modify_csv(filename):
    # Read CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Subtract 1 from values in the 4th column
    df.iloc[:, 3] -= 1

    # Save the modified DataFrame back to the CSV file
    df.to_csv(filename, index=False)


# List of CSV files to modify
files_to_modify = ['test_data2.csv', 'train_data2.csv', 'validation_data2.csv']

# Loop through each file and modify
for file in files_to_modify:
    modify_csv(file)
