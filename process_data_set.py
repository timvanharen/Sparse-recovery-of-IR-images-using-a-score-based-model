# Process the data set to create the training and testing data sets from "data" folder
import os
import numpy as np
import cv2
from pathlib import Path

# max amount of images to process
MAX_TRAIN_IMAGE = 10742 # Max is around 10700
MAX_TEST_IMAGE = 1144
CLEAN_TRAIN_DIRS_FIRST = False # Set to True to clean the directories before processing the images
CLEAN_TEST_DIRS_FIRST = True

task = 'normal'

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print("Configuration:")
for k, v in config.items():
    print(f"{k}: {v}")



# Resolution of the images
# 1. High resolution (HR)
# 2. Frikandel resolution (2 (\sqrt(2) x \sqrt(2)) times smaller than HR)
# 3. Half resolution (HalfR) (4 (2x2) times smaller than HR)
# 4. Medium resolution (MR) (16 (4x4) times smaller than HR)
# 5. Low resolution (LR) (256 (16x16) times smaller than HR)
frikandel_reduce_factor = np.sqrt(2)
half_reduce_factor = 2
medium_reduce_factor = 4
low_reduce_factor = 16
# Put these factors in a list to make it easier to iterate over them
reduce_factors = [1, frikandel_reduce_factor, half_reduce_factor, medium_reduce_factor, low_reduce_factor]

# Dir paths
#data_dir = Path("images/data/data")
#data_val_dir = Path("images/data/data_val")
HR_train_data_output_dir = Path("./images/high_res_train")
HR_test_data_output_dir = Path("./images/high_res_test")
frikandel_train_data_output_dir = Path("./images/frikandel_train")
frikandel_test_data_output_dir = Path("./images/frikandel_test")
half_train_data_output_dir = Path("./images/half_res_train")
half_test_data_output_dir = Path("./images/half_res_test")
MR_train_data_output_dir = Path("./images/medium_res_train")
MR_test_data_output_dir = Path("./images/medium_res_test")
LR_train_data_output_dir = Path("./images/low_res_train")
LR_test_data_output_dir = Path("./images/low_res_test")

if task == 'square': 
    print("Square task selected.")
    train_file_list = os.listdir(HR_train_data_output_dir)
    test_file_list = os.listdir(HR_test_data_output_dir)
    train_image_count = len(train_file_list)
    test_image_count = len(test_file_list)  
    if test_image_count > 0:
        print("Using the train images for the square task.")
        data_dir = HR_train_data_output_dir
    if train_image_count > 0:
        print("Using the test images for the square task.")
        data_val_dir = HR_test_data_output_dir
    else:
        print("No images found in the data directory. This is needd for making square images.")
        print("Please put the images in the data directory.")
        print("Exiting...")
        exit()

    HR_train_data_output_dir = Path("./images-square/high_res_train")
    HR_test_data_output_dir = Path("./images-square/high_res_test")
    frikandel_train_data_output_dir = Path("./images-square/frikandel_train")
    frikandel_test_data_output_dir = Path("./images-square/frikandel_test")
    half_train_data_output_dir = Path("./images-square/half_res_train")
    half_test_data_output_dir = Path("./images-square/half_res_test")
    MR_train_data_output_dir = Path("./images-square/medium_res_train")
    MR_test_data_output_dir = Path("./images-square/medium_res_test")
    LR_train_data_output_dir = Path("./images-square/low_res_train")
    LR_test_data_output_dir = Path("./images-square/low_res_test")
    

# Put these paths in lists per train/test to make it easier to iterate over them
output_train_dirs = [HR_train_data_output_dir, frikandel_train_data_output_dir, half_train_data_output_dir, MR_train_data_output_dir, LR_train_data_output_dir]
output_test_dirs = [HR_test_data_output_dir, frikandel_test_data_output_dir, half_test_data_output_dir, MR_test_data_output_dir, LR_test_data_output_dir]

# File names for the training and testing data sets
HR_train_file_name = "high_res_train"
HR_test_file_name = "high_res_test"
frikandel_train_file_name = "frikandel_train"
frikandel_test_file_name = "frikandel_test"
half_train_file_name = "half_res_train"
half_test_file_name = "half_res_test"
MR_train_file_name = "medium_res_train"
MR_test_file_name = "medium_res_test"
LR_train_file_name = "low_res_train"
LR_test_file_name = "low_res_test"

# Put these file names in lists per train/test to make it easier to iterate over them
output_train_file_names = [HR_train_file_name, frikandel_train_file_name, half_train_file_name, MR_train_file_name, LR_train_file_name]
output_test_file_names = [HR_test_file_name, frikandel_test_file_name, half_test_file_name, MR_test_file_name, LR_test_file_name]

# Check if data and data_val directories exist
if not os.path.exists(data_dir):
    print("Data directory does not exist.")
    exit()
if not os.path.exists(data_val_dir):
    print("Data validation directory does not exist.")
    exit()

# Iterate over the directories and check if they exist otherwise create them
for dir in output_train_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

for dir in output_test_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# List the files in the data directory
train_file_list = os.listdir(data_dir)
test_file_list = os.listdir(data_val_dir)
train_image_count = len(train_file_list)
test_image_count = len(test_file_list)
print("Number of training samples available: ", train_image_count)
print("Number of testing samples available: ", test_image_count)

# show the resolution of 1 image
image = cv2.imread(os.path.join(data_dir, train_file_list[0]))
height, width = image.shape[:2]
print("Resolution of the images: ", height, "x", width)
input("Press Enter to continue...")

# Clear the directories before processing the images
if CLEAN_TRAIN_DIRS_FIRST == True:
    for dir in output_train_dirs:
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))
    print("Train directories cleaned.")
if CLEAN_TEST_DIRS_FIRST == True:
    for dir in output_test_dirs:
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))
    print("Test directories cleaned.")

# Save a list of the number of files in all the directories
train_data_dir_num = []
test_data_dir_num = []

# For training images
for dir in output_train_dirs:
    train_data_dir_num.append(len(os.listdir(dir)))
    print("Number of images stored in: ", dir, "is: ", train_data_dir_num[-1])

# For testing images
for dir in output_test_dirs:
    test_data_dir_num.append(len(os.listdir(dir)))
    print("Number of images stored in: ", dir, "is: ", test_data_dir_num[-1])

# This function downsamples the image to a smaller size of columns and rows specified by the user
def downsample_image(image, rows, cols):
    # Resize the image to the specified number of rows and columns
    downsampled_image = cv2.resize(image, (rows, cols))
    return downsampled_image

def making_square_image(image):
    # Check if the image is already square
    if image.shape[0] == image.shape[1]:
        return image
    # If not, make it square by cropping the larger dimension
    height, width = image.shape[:2]
    size = min(height, width)

    # Calculate crop offsets to center the square
    y_offset = (height - size) // 2
    x_offset = (width - size) // 2

    return image[y_offset:y_offset+size, x_offset:x_offset+size]


# Check if both processes need to be done by summing the number of images in the directories and multiplying by the max amount of images to process
if (sum(train_data_dir_num)) >= (MAX_TRAIN_IMAGE * len(output_train_dirs)):
    print("Train directory is full. No need to process the images.")
else:
    # Process the traing data set
    for file_count, file in enumerate(os.listdir(data_dir)): # check every file in the input dir
        if file.endswith(".jpg") and file_count < MAX_TRAIN_IMAGE:
            # Read the image file in grayscale
            image = cv2.imread(os.path.join(data_dir, file), cv2.IMREAD_GRAYSCALE)
            for i, dir in enumerate(output_train_dirs): # check every dir individually
                if test_data_dir_num[i] < MAX_TRAIN_IMAGE: # check if we need to fill a dir at all
                    # Change the name to a numbered name
                    new_file = output_train_file_names[i] + "_" + str(file_count) + ".jpg"
                    if task == 'square':
                        image = making_square_image(image)
                    if i == 0: #We want to store the original image in the HR directory
                        cv2.imwrite(os.path.join(dir, new_file), image)
                    else: # Resize the image to the specified number of rows and columns
                        # Downsample the image
                        rows = int(image.shape[1] // reduce_factors[i])
                        cols = int(image.shape[0] // reduce_factors[i])
                        downsampled_image = downsample_image(image, rows, cols)
                        # Write the image to the dir
                        cv2.imwrite(os.path.join(dir, new_file), downsampled_image)
                

            # Print the progress every 100 images by printing the index of the image
            if file_count % (train_image_count//10) == 0:
                print("progress: ", file_count, "/", train_image_count, " For all resolutions")

# Process the testing data set
for file_count, file in enumerate(os.listdir(data_val_dir)): # check every file in the input dir
    if file.endswith(".jpg") and file_count < MAX_TEST_IMAGE:
        # Read the image file
        image = cv2.imread(os.path.join(data_val_dir, file), cv2.IMREAD_GRAYSCALE)
        for i, dir in enumerate(output_test_dirs): # check every dir individually
            if test_data_dir_num[i] < MAX_TEST_IMAGE : # check if we need to fill a dir at all
                # Change the name to a numbered name
                new_file = output_test_file_names[i] + "_" + str(file_count) + ".jpg"
                if task == 'square':
                    image = making_square_image(image)
                if i == 0: #We want to store the original image in the HR directory
                    cv2.imwrite(os.path.join(dir, new_file), image)
                else: # Resize the image to the specified number of rows and columns
                    # Downsample the image
                    rows = int(image.shape[1] // reduce_factors[i])
                    cols = int(image.shape[0] // reduce_factors[i])
                    downsampled_image = downsample_image(image, rows, cols)
                    # Write the image to the dir
                    cv2.imwrite(os.path.join(dir, new_file), downsampled_image)

        # Print the progress every 100 images by printing the index of the image
        if file_count % (test_image_count//10) == 0:
            print("progress: ", file_count, "/", test_image_count, " For all resolutions")

print("Finished processing training data set.")
# For training images
for dir in output_train_dirs:
    train_data_dir_num.append(len(os.listdir(dir)))
    print("Number of images stored in: ", dir, "is: ", train_data_dir_num[-1])

# For testing images
for dir in output_test_dirs:
    test_data_dir_num.append(len(os.listdir(dir)))
    print("Number of images stored in: ", dir, "is: ", test_data_dir_num[-1])