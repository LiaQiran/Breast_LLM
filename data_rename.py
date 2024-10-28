#%%
import os
import shutil
import os

def rename_files(directory,old,new):
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if it's a file
        if os.path.isfile(file_path):
            # Check if the file name contains '_GT'
            if '_GT' in filename:
                # Generate the new file name by removing '_GT'
                new_filename = filename.replace(old,new)
                new_file_path = os.path.join(directory, new_filename)

                # Rename the file
                os.rename(file_path, new_file_path)
                print(f'Renamed: {file_path} -> {new_file_path}')





def copy_files(source_dir, destination_dir):
    """
    Copy all files from the source directory to the destination directory.

    Parameters:
    - source_dir (str): Path to the source directory containing files to move.
    - destination_dir (str): Path to the destination directory where files will be moved.
    """
    # Ensure the destination directory exists, if not, create it
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Get list of all files in the source directory
    files = os.listdir(source_dir)

    # Loop over all files and move them to the destination directory
    for file_name in files:
        # Construct full file paths
        source = os.path.join(source_dir, file_name)
        destination = os.path.join(destination_dir, file_name)

        # Move file
        shutil.copy(source, destination)
        print(f"Moved: {file_name}")

def rename_files_with_suffix(directory, suffix):
    """
    Renames all files in the given directory by adding a suffix at the end of each file name.

    Parameters:
    - directory (str): The path to the directory containing the files.
    - suffix (str): The suffix to add to each file name (e.g., "_BUSIS").
    """
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if it's a file
        if os.path.isfile(file_path):
            # Split the filename and its extension
            name, ext = os.path.splitext(filename)

            # Generate the new file name by adding the suffix before the extension
            new_filename = f"{name}{suffix}{ext}"
            new_file_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(file_path, new_file_path)
            print(f'Renamed: {file_path} -> {new_file_path}')

import os

def remove_suffix_from_files(directory, suffix):
    """
    Removes the specified suffix from all files in the given directory.

    Parameters:
    - directory (str): The path to the directory containing the files.
    - suffix (str): The suffix to remove from each file name (e.g., "_BUSIS").
    """
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if it's a file
        if os.path.isfile(file_path):
            # Split the filename and its extension
            name, ext = os.path.splitext(filename)

            # Check if the filename ends with the suffix
            if name.endswith(suffix):
                # Remove the suffix from the filename
                new_name = name[:-len(suffix)]
                new_filename = f"{new_name}{ext}"
                new_file_path = os.path.join(directory, new_filename)

                # Rename the file
                os.rename(file_path, new_file_path)
                print(f'Renamed: {file_path} -> {new_file_path}')

# Example usage:
# remove_suffix_from_files('/path/to/directory', '_BUSIS')


#%%
directory_BUSIS_label = r'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUSIS/BUSIS/GT'
directory_BUSIS_image=  r'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUSIS/BUSIS/Original'
destination_label=r'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/labels'
destination_image=r'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/images'
#%%
# # Call the function
# rename_files(directory_BUSIS_label,'_GT', '_BUSIS')
# #%%
# remove_suffix_from_files(directory_BUSIS_label,'.png')
# #%%
# copy_files(directory_BUSIS_label,destination_label)
# copy_files(directory_BUSIS_image,destination_image)
#%%
directory_BUSBRA_label = r'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUSBRA/masks'
directory_BUSBRA_image=  r'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUSBRA/Images'
import os

import os


def rename_files(folder1, folder2):
    # List all files in the first folder
    files1 = os.listdir(folder1)
    # List all files in the second folder
    files2 = os.listdir(folder2)

    for file1, file2 in zip(files1, files2):
        # Extract the common part of the filename (everything after "bus_" and before "-")
        parts1 = file1.split("-")
        parts2 = file2.split("-")

        if len(parts1) > 1 and len(parts2) > 1:
            number = parts1[0].split("_")[1]  # Get the number part
            side1 = parts1[1][0].upper()  # Get the side part and capitalize it
            side2 = parts2[1][0].upper()  # Should be the same as side1, but just in case

            new_name = f"case{number}_{side1}_BUSBRA"

            # Rename file1
            new_file1 = os.path.join(folder1, new_name)
            os.rename(os.path.join(folder1, file1), new_file1)
            print(f'Renamed: {file1} -> {new_name}')

            # Rename file2
            new_file2 = os.path.join(folder2, new_name)
            os.rename(os.path.join(folder2, file2), new_file2)
            print(f'Renamed: {file2} -> {new_name}')



# Rename files in the mask directory
#rename_files(directory_BUSBRA_label, directory_BUSBRA_image)
#%%
# copy_files(directory_BUSBRA_label,destination_label)
# copy_files(directory_BUSBRA_image,destination_image)
#%% for STU
import os
import shutil


def rename_and_move_files(src_directory, labels_directory, images_directory):
    # Ensure destination directories exist
    os.makedirs(labels_directory, exist_ok=True)
    os.makedirs(images_directory, exist_ok=True)

    # Process mask files
    for filename in os.listdir(src_directory):
        if filename.startswith('mask_'):
            # Extract the number from the filename
            number = filename.split('_')[1].split('.')[0]
            # Format the new name as caseXX_STU.png
            new_name = f"case{int(number):02d}_STU.png"
            # Define source and destination paths
            src_path = os.path.join(src_directory, filename)
            dst_path = os.path.join(labels_directory, new_name)
            # Move the file
            shutil.move(src_path, dst_path)
            print(f"Renamed and moved: {src_path} -> {dst_path}")

    # Process Test_Image and Task_Image files
    for filename in os.listdir(src_directory):
        if filename.startswith('Test_Image_') or filename.startswith('Task_Image_'):
            # Extract the number from the filename
            number = filename.split('_')[-1].split('.')[0]
            # Format the new name as caseXX_STU.png
            new_name = f"case{int(number):02d}_STU.png"
            # Define source and destination paths
            src_path = os.path.join(src_directory, filename)
            dst_path = os.path.join(images_directory, new_name)
            # Move the file
            shutil.move(src_path, dst_path)
            print(f"Renamed and moved: {src_path} -> {dst_path}")


# Example usage
src_directory = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/STU-Hospital-master/STU-Hospital-master/Hospital'
labels_directory = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/labels'
images_directory = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/images'

rename_and_move_files(src_directory, labels_directory, images_directory)
#%% for BUSUCM
import os


def rename_files_BUSUCM(directory):
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if it's a file and matches the pattern of numbers
        if os.path.isfile(file_path) and filename.endswith('.png'):
            # Extract the number part of the filename
            number_part = filename.split('.')[0]

            # Format the number part with leading zeros
            new_number = number_part.zfill(3)

            # Generate the new file name
            new_filename = f"case{new_number}_BUSUCM.png"
            new_file_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(file_path, new_file_path)
            print(f'Renamed: {file_path} -> {new_file_path}')


# Example usage:
# BUSUCM_image = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_UC/BUS_UC/Malignant/images'
# BUSUCM_label = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_UC/BUS_UC/Malignant/masks'
# rename_files_BUSUCM(BUSUCM_image)
# rename_files_BUSUCM(BUSUCM_label)
#
# copy_files(BUSUCM_image,destination_image)
# copy_files(BUSUCM_label,destination_label)

#%% for BUSUCB
def rename_files_in_directory(directory, prefix, suffix):
    # List all files in the directory
    files = sorted(os.listdir(directory))

    # Loop through each file in the directory
    for i, filename in enumerate(files, start=1):
        # Generate the new filename
        new_filename = f"{prefix}{i:03d}{suffix}.png"

        # Create the full file paths
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {filename} -> {new_filename}")


# # Usage
# BUSUCB_image = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_UC/BUS_UC/Benign/images'
# BUSUCB_label = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_UC/BUS_UC/Benign/masks'
# rename_files_in_directory(BUSUCB_image, 'case', '_BUSUCB')
# rename_files_in_directory(BUSUCB_label, 'case', '_BUSUCB')
# #%%
# copy_files(BUSUCB_image,destination_image)
# copy_files(BUSUCB_label,destination_label)
#%% for BUS_UDIAT
import os


def rename_files_in_directory_BUSUDIAT(directory, prefix, suffix):
    # List all files in the directory
    files = sorted(os.listdir(directory))

    # Loop through each file in the directory
    for filename in files:
        # Ensure the file is a PNG and has the correct naming format
        if filename.endswith('.png') and filename[:6].isdigit():
            # Extract the number from the filename
            number = filename[:6]

            # Generate the new filename
            new_filename = f"{prefix}{int(number):03d}{suffix}.png"

            # Create the full file paths
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")



# # Usage
# BUSUDIAT_image = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS UDIAT only seg/BUS/original'
# BUSUDIAT_label = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS UDIAT only seg/BUS/GT'
# rename_files_in_directory_BUSUDIAT(BUSUDIAT_image, 'case', '_BUSUDIAT')
# rename_files_in_directory_BUSUDIAT(BUSUDIAT_label, 'case', '_BUSUDIAT')
#
# #%%
# copy_files(BUSUDIAT_image,destination_image)
# copy_files(BUSUDIAT_label,destination_label)
#%% for BUSUSG

import os
import shutil


def process_files(source_directory, images_directory, labels_directory):
    # Create directories if they don't exist
    os.makedirs(images_directory, exist_ok=True)
    os.makedirs(labels_directory, exist_ok=True)

    # Process files in the source directory
    for filename in os.listdir(source_directory):
        file_path = os.path.join(source_directory, filename)

        if os.path.isfile(file_path):
            # Split the filename and its extension
            name, ext = os.path.splitext(filename)

            # Check if it's an image or tumor file and process accordingly
            if '_tumor' in filename:
                # Generate the new file name for labels
                new_filename = f"{name.replace('_tumor', '')}_BUSUSG{ext}"
                new_file_path = os.path.join(labels_directory, new_filename)
                shutil.copy(file_path, new_file_path)
                print(f"Moved and renamed {filename} to {new_filename} in labels directory.")
            else:
                # Generate the new file name for images
                new_filename = f"{name}_BUSUSG{ext}"
                new_file_path = os.path.join(images_directory, new_filename)
                shutil.copy(file_path, new_file_path)
                print(f"Moved and renamed {filename} to {new_filename} in images directory.")


# Usage
# source_directory = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BrEaST-Lesions_USG-images_and_masks'
# images_directory = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/images'
# labels_directory = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/labels'
#
# process_files(source_directory, images_directory, labels_directory)
#%%
import os

def normalize_filename(filename):
    """
    Normalize the filename by removing the extension and converting to lowercase.
    """
    name, _ = os.path.splitext(filename)
    return name.lower()

def find_missing_files(folder1, folder2):
    # Get the list of files in both folders
    files1 = set(normalize_filename(f) for f in os.listdir(folder1))
    files2 = set(normalize_filename(f) for f in os.listdir(folder2))

    # Find files in folder1 that are not in folder2
    missing_in_folder2 = files1 - files2

    # Find files in folder2 that are not in folder1
    missing_in_folder1 = files2 - files1

    # Print results
    if missing_in_folder2:
        print("Files in folder1 but not in folder2:")
        for file in missing_in_folder2:
            print(file)

    if missing_in_folder1:
        print("Files in folder2 but not in folder1:")
        for file in missing_in_folder1:
            print(file)

# Example usage
folder1 = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/test/images'
folder2 = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/test/labels'

find_missing_files(folder1, folder2)
#%% for BUSSD
# import os

import os


def rename_files_BUSSD(directory):
    """
    Renames files in the given directory by removing the last three digits from the filename
    (before the extension) and using the remaining number to create new filenames in the format
    'caseXXX_BUSSD.png'.

    Parameters:
    - directory (str): The path to the directory containing the files to rename.
    """
    # Loop through all files in the directory, sorted to maintain order
    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, filename)

        # Check if it's a file
        if os.path.isfile(file_path):
            # Extract the name without the extension
            name_without_ext = os.path.splitext(filename)[0]

            # Remove the last three digits from the filename
            if len(name_without_ext) > 3:
                name_part = name_without_ext[:-3]  # Remove last three characters
            else:
                name_part = name_without_ext  # If less than 3 characters, keep as is

            try:
                # Convert the remaining name part to an integer
                number_part = int(name_part)

                # Generate the new filename
                new_filename = f"case{number_part:03d}_BUSSD.png"
                new_file_path = os.path.join(directory, new_filename)

                # Rename the file
                os.rename(file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_filename}")
            except ValueError:
                print(f"Skipping {filename}: '{name_part}' is not a valid number.")



# Example usage
directory_BUSSD_image = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS Synthetic Dataset/images'
directory_BUSSD_label = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS Synthetic Dataset/masks'
rename_files_BUSSD(directory_BUSSD_image)
rename_files_BUSSD(directory_BUSSD_label)

#%%

copy_files(directory_BUSSD_image,destination_image)
copy_files(directory_BUSSD_label,destination_label)
