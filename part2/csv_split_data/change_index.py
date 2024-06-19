import pandas as pd


# this function changes the index of the labels to 1 less (from 1 to 4 based to 0 to 3)
def modify_csv(filename):
    # read csv file into a DataFrame
    df = pd.read_csv(filename)

    # subtract 1 from values in the 4th column
    df.iloc[:, 3] -= 1

    # save the modified DataFrame back to the CSV file
    df.to_csv(filename, index=False)


# list of csv files to modify
files_to_modify = ['test_data_final.csv', 'train_data_final.csv', 'validation_data_final.csv']

# loop through each file and modify
for file in files_to_modify:
    modify_csv(file)
