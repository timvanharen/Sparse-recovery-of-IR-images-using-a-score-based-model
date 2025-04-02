# Process the data set to create the training and testing data sets from "data" folder
import os
import numpy as np
import cv2

# max amount of images to process
MAX_IMAGE = 2000
TEST_RATIO = 0.2

# Dir paths
data_dir = "images\data\data"
HR_train_data_output_dir = ".\images\high_res_train\HR_train\\"
HR_test_data_output_dir = ".\images\high_res_test\HR_test\\"
LR_train_data_output_dir = ".\images\low_res_train\LR_train\\"
LR_test_data_output_dir = ".\images\low_res_test\LR_test\\"

# Also make a directory for the discrete cosine transform of all the images
DCT_HR_train_data_output_dir = ".\images\dct_high_res_train\DCT_HR_train\\"
DCT_HR_test_data_output_dir = ".\images\dct_high_res_test\DCT_HR_test\\"
DCT_LR_train_data_output_dir = ".\images\dct_low_res_train\DCT_LR_train\\"
DCT_LR_test_data_output_dir = ".\images\dct_low_res_test\DCT_LR_test\\"

# Split the data set into training and testing sets
def process_data_set(data_dir, test_ratio=0.2):
    """
    Process the data set to create the training and testing data sets.
    
    Args:
    - data_dir: Directory containing the data set.
    - test_ratio: Ratio of the data set to use for testing.
    
    Returns:
    - train_data: Training data set.
    - test_data: Testing data set.
    
    """
    
    # Get the list of files in the data directory
    file_list = os.listdir(data_dir)
    
    # Shuffle the list of files
    np.random.shuffle(file_list)
    
    # Calculate the number of test samples
    num_test_samples = int(len(file_list) * test_ratio)
    
    # Split the data set into training and testing sets
    test_data = file_list[:num_test_samples]
    train_data = file_list[num_test_samples:]
    
    return train_data, test_data

# This function downsamples the image to a smaller size of columns and rows specified by the user
def downsample_image(image, rows, cols):
    """
    Downsample the image to a smaller size.
    
    Args:
    - image: Input image as a NumPy array.
    - rows: Number of rows in the downsampled image.
    - cols: Number of columns in the downsampled image.
    
    Returns:
    - downsampled_image: Downsampled image.
    
    """
    
    # Resize the image to the specified number of rows and columns
    downsampled_image = cv2.resize(image, (rows, cols))
    
    return downsampled_image


# Test the splitting of the data set
train_data, test_data = process_data_set(data_dir, TEST_RATIO)
print("Number of training samples available: ", len(train_data))
print("Number of testing samples available: ", len(test_data))

# Check if "images/data" directory exists
if not os.path.exists(data_dir):
    print("images/data directory does not exist, so the data set is probably missing.")
    exit()
else:
    print("Data directory exists.")

# Check if the directories are already created
if not os.path.exists(HR_train_data_output_dir):
    os.makedirs(HR_train_data_output_dir)
if not os.path.exists(HR_test_data_output_dir):
    os.makedirs(HR_test_data_output_dir)
if not os.path.exists(LR_train_data_output_dir):
    os.makedirs(LR_train_data_output_dir)
if not os.path.exists(LR_test_data_output_dir):
    os.makedirs(LR_test_data_output_dir)

# Check how much files are already in the directories
print("Number of files in HR_train_data_output_dir: ", len(os.listdir(HR_train_data_output_dir)))
print("Number of files in HR_test_data_output_dir: ", len(os.listdir(HR_test_data_output_dir)))
print("Number of files in LR_train_data_output_dir: ", len(os.listdir(LR_train_data_output_dir)))
print("Number of files in LR_test_data_output_dir: ", len(os.listdir(LR_test_data_output_dir)))
input("Press Enter to continue...")

image_count = len(os.listdir(HR_train_data_output_dir))
print("Images stored in high_res_train: ", image_count)
# Write the training and testing data sets to the corresponding directories
for file in train_data:
    if image_count < MAX_IMAGE:
        # Read the image file
        image = cv2.imread(os.path.join(data_dir, file))
        
        # Change the name to a numbered name
        new_file = "high_res_train_" + str(train_data.index(file)) + ".jpg"
            
        # Write the image to the HR_train_data_output_dir directory
        cv2.imwrite(os.path.join(HR_train_data_output_dir, new_file), image)

        # Print the progress every 100 images by printing the index of the image
        if train_data.index(file) % 100 == 0:
            print("progress: ", train_data.index(file))
        
        image_count += 1

image_count = len(os.listdir(HR_test_data_output_dir))
print("Images stored in high_res_test: ", image_count)
for file in test_data:
    if image_count < MAX_IMAGE*TEST_RATIO:
        # Read the image file
        image = cv2.imread(os.path.join(data_dir, file))
        
        # Change the name to a numbered name
        new_file = "high_res_test_" + str(test_data.index(file)) + ".jpg"

        # Write the image to the HR_test_data_output_dir directory
        cv2.imwrite(os.path.join(HR_test_data_output_dir, new_file), image)

        # Print the progress every 100 images
        if test_data.index(file) % 100 == 0:
            print("progress: ", test_data.index(file))

        image_count += 1

# And then also write the downsampled images to the corresponding directories called "low_res_train" and "low_res_test"

# First load one image to get the size of the downsampled image
image = cv2.imread(os.path.join(data_dir, train_data[0]))

# Show the different channels of the image separately
b, g, r = cv2.split(image)
cv2.imshow('Blue Channel', b)
cv2.imshow('Green Channel', g)
cv2.imshow('Red Channel', r)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Size of image: ", image.shape)
factor = 16
rows = image.shape[1]//factor
cols = image.shape[0]//factor
print("rows: ", rows)
print("cols: ", cols)

image_count = len(os.listdir(LR_train_data_output_dir))
print("Images stored in low_res_train: ", image_count)
for file in train_data:
    if image_count < MAX_IMAGE:
        # Read the image file
        image = cv2.imread(os.path.join(data_dir, file))
        
        # Downsample the image
        downsampled_image = downsample_image(image, rows, cols)

        # Change the name to a numbered name
        new_file = "low_res_train_" + str(train_data.index(file)) + ".jpg"
        
        # Write the downsampled image to the low resolution training directory
        cv2.imwrite(os.path.join(LR_train_data_output_dir, new_file), downsampled_image)

        # Print the progress every 100 images
        if train_data.index(file) % 100 == 0 and train_data.index(file) < MAX_IMAGE:
            print("progress: ", train_data.index(file))
        
        image_count += 1

image_count = len(os.listdir(LR_test_data_output_dir))
print("Images stored in low_res_test: ", image_count)
for file in test_data:
    if image_count < MAX_IMAGE*TEST_RATIO:
        # Read the image file
        image = cv2.imread(os.path.join(data_dir, file))
        
        # Downsample the image
        downsampled_image = downsample_image(image, rows, cols)

        # Change the name to a numbered name
        new_file = "low_res_test_" + str(test_data.index(file)) + ".jpg"
        
        # Write the downsampled image to the low resolution testing directory
        cv2.imwrite(os.path.join(LR_test_data_output_dir, new_file), downsampled_image)

        # Print the progress every 100 images
        if test_data.index(file) % 100 == 0 and test_data.index(file) < MAX_IMAGE:
            print("progress: ", test_data.index(file))

        image_count += 1


# Now enter the second stage of the script, which is to perform the discrete cosine transform on the images and store them in the corresponding directories
# Check if the directories are already created
if not os.path.exists(DCT_HR_train_data_output_dir):
    os.makedirs(DCT_HR_train_data_output_dir)
if not os.path.exists(DCT_HR_test_data_output_dir):
    os.makedirs(DCT_HR_test_data_output_dir)
if not os.path.exists(DCT_LR_train_data_output_dir):
    os.makedirs(DCT_LR_train_data_output_dir)
if not os.path.exists(DCT_LR_test_data_output_dir):
    os.makedirs(DCT_LR_test_data_output_dir)

# Check how much files are already in the directories
print("Number of files in DCT_HR_train_data_output_dir: ", len(os.listdir(DCT_HR_train_data_output_dir)))
print("Number of files in DCT_HR_test_data_output_dir: ", len(os.listdir(DCT_HR_test_data_output_dir)))
print("Number of files in DCT_LR_train_data_output_dir: ", len(os.listdir(DCT_LR_train_data_output_dir)))
print("Number of files in DCT_LR_test_data_output_dir: ", len(os.listdir(DCT_LR_test_data_output_dir)))
input("Press Enter to continue...")

image_count = len(os.listdir(DCT_HR_train_data_output_dir))
print("Images stored in DCT_HR_train: ", image_count)
# Write the training and testing data sets to the corresponding directories
for file in train_data:
    if image_count < MAX_IMAGE:
        # Read the image file
        image = cv2.imread(os.path.join(data_dir, file))
        
        # Change the name to a numbered name
        new_file = "dct_train_" + str(train_data.index(file)) + ".jpg"
        
        # Perform the discrete cosine transform on the image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dct_image = cv2.dct(np.float32(gray_image), cv2.DCT_INVERSE)

        # Write the image to the DCT_HR_train_data_output_dir directory
        cv2.imwrite(os.path.join(DCT_HR_train_data_output_dir, new_file), dct_image)

        # Print the progress every 100 images by printing the index of the image
        if train_data.index(file) % 100 == 0:
            print("progress: ", train_data.index(file))
        
        image_count += 1

image_count = len(os.listdir(DCT_HR_test_data_output_dir))
print("Images stored in DCT_HR_test: ", image_count)

for file in test_data:
    if image_count < MAX_IMAGE*TEST_RATIO:
        # Read the image file
        image = cv2.imread(os.path.join(data_dir, file))
        
        # Change the name to a numbered name
        new_file = "dct_test_" + str(test_data.index(file)) + ".jpg"
        
        # Perform the discrete cosine transform on the image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dct_image = cv2.dct(np.float32(gray_image), cv2.DCT_INVERSE)

        # Write the image to the DCT_HR_test_data_output_dir directory
        cv2.imwrite(os.path.join(DCT_HR_test_data_output_dir, new_file), dct_image)

        # Print the progress every 100 images
        if test_data.index(file) % 100 == 0:
            print("progress: ", test_data.index(file))

        image_count += 1

# Now dct the low resolution images
image_count = len(os.listdir(DCT_LR_train_data_output_dir))
print("Images stored in DCT_LR_train: ", image_count)

for file in train_data:
    if image_count < MAX_IMAGE:
        # Read the image file
        image = cv2.imread(os.path.join(data_dir, file))
        
        # Change the name to a numbered name
        new_file = "dct_low_res_train_" + str(train_data.index(file)) + ".jpg"
        
        # Perform the discrete cosine transform on the image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dct_image = cv2.dct(np.float32(gray_image), cv2.DCT_INVERSE)

        # Write the image to the DCT_LR_train_data_output_dir directory
        cv2.imwrite(os.path.join(DCT_LR_train_data_output_dir, new_file), dct_image)

        # Print the progress every 100 images by printing the index of the image
        if train_data.index(file) % 100 == 0:
            print("progress: ", train_data.index(file))
        
        image_count += 1

image_count = len(os.listdir(DCT_LR_test_data_output_dir))
print("Images stored in DCT_LR_test: ", image_count)

for file in test_data:
    if image_count < MAX_IMAGE*TEST_RATIO:
        # Read the image file
        image = cv2.imread(os.path.join(data_dir, file))
        
        # Change the name to a numbered name
        new_file = "dct_low_res_test_" + str(test_data.index(file)) + ".jpg"
        
        # Perform the discrete cosine transform on the image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dct_image = cv2.dct(np.float32(gray_image), cv2.DCT_INVERSE)

        # Write the image to the DCT_LR_test_data_output_dir directory
        cv2.imwrite(os.path.join(DCT_LR_test_data_output_dir, new_file), dct_image)

        # Print the progress every 100 images
        if test_data.index(file) % 100 == 0:
            print("progress: ", test_data.index(file))

        image_count += 1