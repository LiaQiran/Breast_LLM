#%%
import pandas as pd
import os
import numpy as np


def load_csv(csv_file_path, key1, key2):
    # Load the CSV file
    csv_file = csv_file_path

    # Read the CSV file, loading only the specified columns
    df = pd.read_csv(csv_file, usecols=[key1, key2])
    print(df.head())

    # Convert the columns to arrays
    key1_value = df[key1].values
    key2_value = df[key2].values

    return key1_value, key2_value


def load_directory_keywords(path, keywords):
    # Get the list of all files in the directory
    all_files = os.listdir(path)

    # Filter filenames that contain the specified keyword
    filtered_files = [file for file in all_files if keywords in file]

    # Convert the filtered filenames to a NumPy array
    filtered_files_array = np.array(filtered_files)
    print(filtered_files_array)

    return filtered_files_array


def match_and_save_BUSBRA(filenames_csv, types_csv, filtered_files_array, output_csv):
    # Prepare a list to hold the matched data
    matched_data = []

    for file in filtered_files_array:
        # Extract the series number and side (e.g., L or R) from the filename
        # Assuming the format is like 'case0001_L_BUSBRA.png'
        series_number = file.split('_')[0][-4:]  # Get the '0001' part
        side = file.split('_')[1]  # Get the 'L' or 'R' part

        # Find the corresponding row in the CSV based on the series number and side
        for i, csv_filename in enumerate(filenames_csv):
            # Example CSV format: 'bus_0001-l.png'
            csv_series_number = csv_filename.split('_')[1][:4]  # Extract '0001'
            csv_side = csv_filename.split('-')[1][0].upper()  # Extract 'L' or 'R' and convert to upper case

            # Check if the series number and side match
            if csv_series_number == series_number and csv_side == side:
                # Get the corresponding type and add it to the matched data
                matched_data.append([file, types_csv[i]])
                break

    # Save the matched data to a CSV file
    matched_df = pd.DataFrame(matched_data, columns=['filename', 'type'])
    matched_df.to_csv(output_csv, index=False)
    print(f"Matched data saved to {output_csv}")


# # Load CSV data
# filename, type = load_csv('Dataset_label/BUSBRA.csv', 'filename', 'type')
#
# # Load directory files that contain the keyword "BUSBRA"
# filtered_files = load_directory_keywords('data/BUS_dataset/test/images/', 'BUSBRA')
#
# # Match the files and save the result to a new CSV file
# match_and_save_BUSBRA(filename, type, filtered_files, 'BUSBRA_test.csv')

#%% BUSUC
def save_filenames_with_types(BUSUCB_train, BUSUCM_train, output_csv):
    # Create lists to hold the filenames and corresponding types
    filenames = []
    types = []

    # Add BUSUCB_train filenames with type 0
    for file in BUSUCB_train:
        filenames.append(file)
        types.append(0)  # Assign type 0 for benign

    # Add BUSUCM_train filenames with type 1
    for file in BUSUCM_train:
        filenames.append(file)
        types.append(1)  # Assign type 1 for malignant

    # Create a DataFrame to store the filenames and types
    df = pd.DataFrame({'filename': filenames, 'type': types})

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")
# Load directory files that contain the keyword "BUSBRA"
# BUSUCB_train = load_directory_keywords('data/BUS_dataset/train/images/', 'BUSUCB')
# BUSUCM_train = load_directory_keywords('data/BUS_dataset/train/images/', 'BUSUCM')
# # Save the filenames with their corresponding types to a CSV file
# save_filenames_with_types(BUSUCB_train, BUSUCM_train, 'BUSUC_train.csv')
#
# BUSUCB_test = load_directory_keywords('data/BUS_dataset/test/images/', 'BUSUCB')
# BUSUCM_test = load_directory_keywords('data/BUS_dataset/test/images/', 'BUSUCM')
# # Save the filenames with their corresponding types to a CSV file
# save_filenames_with_types(BUSUCB_test, BUSUCM_test, 'BUSUC_test.csv')
#%% BUSUDIAT
def match_and_save_BUSUDIAT(filenames_csv, types_csv, filtered_files_array, output_csv):
    matched_data = []

    for file in filtered_files_array:
        # Extract the series number from the filtered file name, e.g., '001' from 'case001_BUSUDIAT.png'
        series_number = file.split('_')[0][-3:]  # Extracts the last 3 digits from the series part 'case001'

        # Match the series number to the filenames from the CSV
        for i, csv_filename in enumerate(filenames_csv):
            csv_series_number = csv_filename.split('.')[0][-3:]  # Extract the last 3 digits from the CSV filename

            if series_number == csv_series_number:
                # If the series numbers match, append the filename and type to the matched_data list
                matched_data.append([file, types_csv[i]])
                break

    # Create a DataFrame to save the matched filenames and types
    df = pd.DataFrame(matched_data, columns=['filename', 'type'])

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Matched data saved to {output_csv}")

# Match the filtered filenames with the CSV data and save the result to a new CSV file

# filename, type = load_csv('Dataset_label/UDIAT.csv', 'filename', 'type')
# BUSUDIAT_train = load_directory_keywords('data/BUS_dataset/train/images/', 'BUSUDIAT')
# match_and_save_BUSUDIAT(filename, type, BUSUDIAT_train, 'BUSUDIAT_train.csv')
# filename, type = load_csv('Dataset_label/UDIAT.csv', 'filename', 'type')
# BUSUDIAT_train = load_directory_keywords('data/BUS_dataset/test/images/', 'BUSUDIAT')
# match_and_save_BUSUDIAT(filename, type, BUSUDIAT_train, 'BUSUDIAT_test.csv')
#%% BUSUSG-BrRaST
# # Load CSV data
#
# import pandas as pd
# import os
# import numpy as np
#
#
# def load_csv(csv_file_path, key1, key2):
#     # Load the CSV file and read the specified columns
#     df = pd.read_csv(csv_file_path, usecols=[key1, key2])
#
#     # Replace 'normal' with 2 in the 'type' column
#    # df[key2] = df[key2].replace('normal', 2)
#
#     return df[key1].values, df[key2].values
#
#
# def load_directory_keywords(path, keywords):
#     # Get the list of all files in the directory
#     all_files = os.listdir(path)
#
#     # Filter filenames that contain the specified keyword
#     filtered_files = [file for file in all_files if keywords in file]
#
#     # Convert the filtered filenames to a NumPy array
#     return np.array(filtered_files)
#
#
# def match_and_save(filenames_csv, types_csv, filtered_files_array, output_csv):
#     matched_data = []
#
#     for file in filtered_files_array:
#         # Extract the series number from the filtered file name, e.g., '010' from 'case010_BUSUSG.png'
#         series_number = file.split('_')[0][-3:]  # Extracts the last 3 digits from the series part 'case010'
#
#         # Match the series number to the filenames from the CSV
#         for i, csv_filename in enumerate(filenames_csv):
#             csv_series_number = csv_filename.split('.')[0][-3:]  # Extract the last 3 digits from the CSV filename
#
#             if series_number == csv_series_number:
#                 # If the series numbers match, append the filename and type to the matched_data list
#                 matched_data.append([file, types_csv[i]])
#                 break
#
#     # Create a DataFrame to save the matched filenames and types
#     df = pd.DataFrame(matched_data, columns=['filename', 'type'])
#
#     # Save the DataFrame to a CSV file
#     df.to_csv(output_csv, index=False)
#     print(f"Matched data saved to {output_csv}")
#
#
# # Load the CSV data (filename and type), replacing 'normal' with 2
# filename_csv, type_csv = load_csv('Dataset_label/BrEaST.csv', 'filename', 'type')
#
# # Load the filtered filenames from the directory
# filtered_files = load_directory_keywords('data/BUS_dataset/train/images/', 'BUSUSG')
#
# # Match the filtered filenames with the CSV data and save the result to a new CSV file
# match_and_save(filename_csv, type_csv, filtered_files, 'BUSUSG_train.csv')
#%%
import pandas as pd

# List of CSV files to concatenate
csv_train =[
'BUSUSG_train.csv','BUSUDIAT_train.csv', 'BUSUC_train.csv', 'BUSBRA_train.csv'

]
csv_test = [
    'BUSUSG_test.csv',
     'BUSUDIAT_test.csv',
    'BUSUC_test.csv',
    'BUSBRA_test.csv',
]

def csv_combine(csv_files, savename):
    # Initialize an empty list to hold DataFrames
    dfs = []

    # Loop through the CSV files and append each one to the list
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    total_df = pd.concat(dfs, ignore_index=True)

    # Save the concatenated DataFrame to a new CSV file
    total_df.to_csv(savename, index=False)

    print("All files have been combined into total_combined_data.csv")


csv_combine(csv_train,'train.csv')

csv_combine(csv_test,'test.csv')
#%% BUSIS
import pandas as pd
import os
import numpy as np

def load_csv(csv_file_path, key1, key2):
    # Load the CSV file and read the specified columns
    df = pd.read_csv(csv_file_path, usecols=[key1, key2])

    # Replace 'B' with 0 and 'M' with 1 in the 'tumor type' column
    df[key2] = df[key2].replace({'B': 0, 'M': 1})

    return df[key1].values, df[key2].values

def load_directory_keywords(path, keywords):
    # Get the list of all files in the directory
    all_files = os.listdir(path)

    # Filter filenames that contain the specified keyword
    filtered_files = [file for file in all_files if keywords in file]

    # Convert the filtered filenames to a NumPy array
    return np.array(filtered_files)

def match_and_save(filenames_csv, types_csv, filtered_files_array, output_csv):
    matched_data = []

    for file in filtered_files_array:
        # Extract the series number from the filtered file name, e.g., '100' from 'case100_BUSIS.png'
        try:
            series_number = file.split('_')[0][4:]  # Extracts '100' from 'case100_BUSIS.png'
            print(series_number)
            # Pad the extracted series number with leading zeros to match the CSV format
            series_number_padded = series_number.zfill(4)  # Converts '100' to '0100'
        except (AttributeError, IndexError):
            continue  # Skip the file if it doesn't follow the expected format

        # Match the padded series number to the filenames from the CSV
        for i, csv_filename in enumerate(filenames_csv):
            if csv_filename.endswith(series_number_padded):  # Match the padded series number
                # If the series numbers match, append the filename and type to the matched_data list
                matched_data.append([file, types_csv[i]])
                break

    # Create a DataFrame to save the matched filenames and types
    df = pd.DataFrame(matched_data, columns=['filename', 'type'])

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Matched data saved to {output_csv}")

# Load the CSV data (filename and type), replacing 'B' with 0 and 'M' with 1
filename_csv, type_csv = load_csv('Dataset_label/BUSIS562.csv', 'img name', 'tumor type')

# # Load the filtered filenames from the directory
# BUSIS_train = load_directory_keywords('data/BUS_dataset/train/images/', 'BUSIS')
#
# # Match the filtered filenames with the CSV data and save the result to a new CSV file
# match_and_save(filename_csv, type_csv, BUSIS_train, 'BUSIS_train.csv')
# Load the filtered filenames from the directory
BUSIS_test = load_directory_keywords('data/BUS_dataset/test/images/', 'BUSIS')

# Match the filtered filenames with the CSV data and save the result to a new CSV file
match_and_save(filename_csv, type_csv, BUSIS_test, 'BUSIS_test.csv')
#%%
import pandas as pd

# List of CSV files to concatenate
csv_files = [
    'BUSUSG_train.csv',
    'BUSUDIAT_train.csv',
    'BUSUC_train.csv',
   'BUSBRA_train.csv',
    "BUSIS_train.csv"
]

# Initialize an empty list to hold DataFrames
dfs = []

# Loop through the CSV files and append each one to the list
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
total_df = pd.concat(dfs, ignore_index=True)

# Save the concatenated DataFrame to a new CSV file
total_df.to_csv('train.csv', index=False)

print("All files have been combined into train.csv")
#%%
import pandas as pd

# List of CSV files to concatenate
csv_files = [
    'BUSUSG_test.csv',
    'BUSUDIAT_test.csv',
    'BUSUC_test.csv',
   'BUSBRA_test.csv',
    "BUSIS_test.csv"
]

# Initialize an empty list to hold DataFrames
dfs = []

# Loop through the CSV files and append each one to the list
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
total_df = pd.concat(dfs, ignore_index=True)

# Save the concatenated DataFrame to a new CSV file
total_df.to_csv('test.csv', index=False)

print("All files have been combined into test.csv")