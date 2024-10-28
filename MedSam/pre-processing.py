#%%
# for label,revalue them into 255 and change the format into .png
import os
from PIL import Image
import numpy as np

def process_images(directory, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a .png or .bmp
        if filename.endswith('.png') or filename.endswith('.bmp'):
            file_path = os.path.join(directory, filename)

            # Open the image
            with Image.open(file_path) as img:
                # Convert the image to grayscale (if not already)
                img = img.convert('L')

                # Convert the image to a NumPy array
                img_array = np.array(img)

                # Set all label values of 1 to 255
                img_array[img_array == 1] = 255

                # Convert the array back to an image
                processed_img = Image.fromarray(img_array)

                # Define the new filename as .png
                new_filename = os.path.splitext(filename)[0] + '.png'
                output_path = os.path.join(output_directory, new_filename)

                # Save the image in PNG format
                processed_img.save(output_path, 'PNG')

                print(f"Processed and saved: {new_filename}")

# Example usage:
input_directory = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/labels'
output_directory = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/labels_png'

process_images(input_directory, output_directory)
#%% resize to 512*512

import os
from PIL import Image

def resize_images_and_masks(image_directory, mask_directory, output_image_directory, output_mask_directory, size=(512, 512)):
    # Create output directories if they don't exist
    os.makedirs(output_image_directory, exist_ok=True)
    os.makedirs(output_mask_directory, exist_ok=True)

    # Loop through all files in the image directory
    for filename in os.listdir(image_directory):
        image_path = os.path.join(image_directory, filename)
        mask_path = os.path.join(mask_directory, filename)  # Assuming the mask has the same name as the image

        if os.path.isfile(image_path) and os.path.isfile(mask_path):
            # Open and resize the image
            with Image.open(image_path) as img:
                resized_image = img.resize(size, Image.LANCZOS)  # Resize image to 512x512, Image.LANCZOS is the preferred high-quality downscaling filter in newer versions of Pillow
                resized_image.save(os.path.join(output_image_directory, filename))

            # Open and resize the corresponding mask
            with Image.open(mask_path) as mask:
                resized_mask = mask.resize(size, Image.NEAREST)  # Resize mask using NEAREST to preserve label values,use Image.NEAREST to ensure that the label values (which are often integers) are preserved
                resized_mask.save(os.path.join(output_mask_directory, filename))

            print(f"Resized: {filename}")

# Example usage:
image_directory = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/images'
mask_directory = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/labels_png'
output_image_directory = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/image_resize'
output_mask_directory = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/labels_resize'

resize_images_and_masks(image_directory, mask_directory, output_image_directory, output_mask_directory, size=(512, 512))

#%%
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def overlay_image_and_mask(image_path, mask_path, alpha=0.5):
    # Open the grayscale image and mask
    image = Image.open(image_path).convert('L')  # Ensure the image is in grayscale
    mask = Image.open(mask_path).convert('L')  # Ensure the mask is grayscale

    # Convert the images to NumPy arrays
    image_array = np.array(image)
    mask_array = np.array(mask)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Display the grayscale image
    ax.imshow(image_array, cmap='gray')

    # Overlay the mask, where mask value is 255 (using red color)
    mask_overlay = np.zeros_like(image_array, dtype=np.uint8)
    mask_overlay[mask_array == 255] = 255  # Set the mask area to 255 (white)

    # Overlay the mask with red transparency
    ax.imshow(mask_overlay, cmap='Reds', alpha=alpha)

    # Turn off axis for a cleaner display
    plt.axis('off')

    # Show the plot
    plt.show()

# Example usage:
image_path = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/image_resize/case002_BUSUCM.png'
mask_path = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/labels_resize/case002_BUSUCM.png'

overlay_image_and_mask(image_path, mask_path)

#%% move images into test and train folder 1:9
import os
import shutil
import random


def split_images_and_masks(image_folder, mask_folder, train_image_folder, test_image_folder, train_mask_folder,
                           test_mask_folder, split_ratio=0.9):
    """
    Splits images and their corresponding masks into train and test sets based on the given split_ratio.

    Parameters:
    - image_folder: Path to the folder containing the images.
    - mask_folder: Path to the folder containing the masks with corresponding filenames.
    - train_image_folder: Path to the folder where training images will be stored.
    - test_image_folder: Path to the folder where test images will be stored.
    - train_mask_folder: Path to the folder where training masks will be stored.
    - test_mask_folder: Path to the folder where test masks will be stored.
    - split_ratio: Proportion of images to include in the training set (default is 0.9).

    The function creates the train and test folders for both images and masks if they don't exist,
    splits the images randomly, and moves both the images and their corresponding masks to the respective folders.
    """

    # Create directories if they do not exist
    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(test_image_folder, exist_ok=True)
    os.makedirs(train_mask_folder, exist_ok=True)
    os.makedirs(test_mask_folder, exist_ok=True)

    # Get the list of image files in the folder
    image_files = os.listdir(image_folder)

    # Filter out non-image files if necessary (adjust extensions as needed)
    image_files = [file for file in image_files if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Shuffle the image files randomly
    random.shuffle(image_files)

    # Calculate the split point for the given ratio (default is 9:1 or 90% for train)
    split_index = int(split_ratio * len(image_files))

    # Split the list into train and test sets
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]

    # Move image and corresponding mask files to their respective folders
    for file_name in train_files:
        # Move the image
        shutil.move(os.path.join(image_folder, file_name), os.path.join(train_image_folder, file_name))

        # Move the corresponding mask
        mask_name = file_name  # Assuming the mask has the same name
        shutil.move(os.path.join(mask_folder, mask_name), os.path.join(train_mask_folder, mask_name))

    for file_name in test_files:
        # Move the image
        shutil.move(os.path.join(image_folder, file_name), os.path.join(test_image_folder, file_name))

        # Move the corresponding mask
        mask_name = file_name  # Assuming the mask has the same name
        shutil.move(os.path.join(mask_folder, mask_name), os.path.join(test_mask_folder, mask_name))

    print(f"Moved {len(train_files)} images and masks to the train folders.")
    print(f"Moved {len(test_files)} images and masks to the test folders.")




#%%
# Example usage
split_images_and_masks('./data/BUS_dataset/images_resize',
                       './data/BUS_dataset/labels_resize',
                       './data/BUS_dataset/train/images',
                       './data/BUS_dataset/test/images',
                       './data/BUS_dataset/train/labels',
                       './data/BUS_dataset/test/labels')
