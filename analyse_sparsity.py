# This file is meant to measure the sparsity of the transform domain of the images, namley the DCT domain.
# 1. loads the images from the low_res, medium_res and high_res directories.
# 2. computes the DCT of the images and then calculates the sparsity of the DCT coefficients by thresholding the coefficients based on a given energy threshold.
# 3. Then it reconstructs the image using the lesser coefficients and measures the quality of the reconstructed image compared to the original.
# TODO: Add wavelet transform and other transforms to the analysis to check if these transforms are more sparse than the DCT.
# 4. Finally, it plots the original image, the reconstructed image and the DCT coefficients for comparison.

# Process the data set to create the training and testing data sets from "data" folder
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags

SHOW_DOWNSAMPLE_IMAGES = True

# Directory paths
HR_train_data_output_dir = ".\images\high_res_train"
HR_test_data_output_dir = ".\images\high_res_test"
MR_train_data_output_dir = ".\images\medium_res_train"
MR_test_data_output_dir = ".\images\medium_res_test"
LR_train_data_output_dir = ".\images\low_res_train"
LR_test_data_output_dir = ".\images\low_res_test"

# File names for the training and testing data sets
HR_train_file_name = "high_res_train_0.jpg"
HR_test_file_name = "high_res_test_0.jpg"
MR_train_file_name = "medium_res_train_0.jpg"
MR_test_file_name = "medium_res_test_0.jpg"
LR_train_file_name = "low_res_train_0.jpg"
LR_test_file_name = "low_res_test_0.jpg"

def create_binomial_filter_1d(k=2):
    """Create 1D binomial filter coefficients"""
    # Simple binomial filter [1,2,1]/4 for k=2
    # For larger k, use coefficients from Pascal's triangle
    coeffs = np.array([1, 2, 1])/4
    return coeffs

def create_filter_matrix(N, filter_coeffs, mode='reflect'):
    """Create sparse filter matrix"""
    half_len = len(filter_coeffs) // 2
    diagonals = []
    offsets = []
    
    for i in range(-half_len, half_len+1):
        diagonals.append(np.ones(N-abs(i)))
        offsets.append(i)
    
    F = diags(diagonals, offsets, shape=(N, N)).toarray()
    # Apply filter coefficients
    for i, offset in enumerate(offsets):
        F[F == (i+1)] = filter_coeffs[i]
    
    return F

def create_decimation_matrix(N, M, factor):
    """Create decimation matrix"""
    S = np.zeros((M, N))
    for i in range(M):
        S[i, i*factor] = 1
    return S

def create_downsample_matrix(N, factor):
    """Combine filter and decimation"""
    filter_coeffs = create_binomial_filter_1d()
    F = create_filter_matrix(N, filter_coeffs)
    M = N // factor
    S = create_decimation_matrix(N, M, factor)
    return S @ F

# This function compresses the energies of the DCT coefficients by thresholding the coefficients based on a given energy threshold.
def dct_thresholding(dct_image, energy_threshold=0.95, clip_max=None, threshold=None):
    # Calculate the total energy of the DCT coefficients
    total_energy = np.sum(dct_image ** 2)

    # Calculate the thresholded energy
    if threshold is None:
        thresholded_energy = total_energy * energy_threshold
    else:
        thresholded_energy = threshold

    print("Total energy: ", total_energy)
    print("Thresholded energy: ", thresholded_energy)

    # Sort the DCT coefficients by their absolute values in descending order
    flat_dct = np.abs(dct_image.flatten())
    print("Max DCT coefficient: ", np.max(flat_dct))

    # # # Clip the DCT coefficients to a certain range
    if clip_max is not None:
        clip_min = 0
        clip_max = clip_max
        flat_dct = np.clip(flat_dct, clip_min, clip_max)

    # Get the indices that would sort the array in descending order
    sorted_indices = np.argsort(flat_dct)[::-1]
    sorted_dct = flat_dct[sorted_indices]

    # Find at which index the sorted dct exceeds the thresholded energy
    threshold_index = np.argmax(sorted_dct < thresholded_energy)
    print("Threshold index: ", threshold_index)
    print("Ratio of coefficients used: ", threshold_index / len(flat_dct))

    # Create a mask for the DCT coefficients above the threshold
    dct_energy = flat_dct[sorted_indices]

    # Show a plot of the DCT energy distribution of the first 100 coefficients
    plt.plot(dct_energy)
    plt.title('DCT Energy Distribution')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Energy')
    plt.yscale('log')
    plt.axhline(y=dct_energy[threshold_index], color='r', linestyle='--', label='Thresholded Energy')
    plt.axvline(x=threshold_index, color='g', linestyle='--', label='Threshold Index')
    plt.legend()
    plt.grid()
    plt.show()

    thresholded_coeff = dct_energy[threshold_index]
    mask = np.abs(dct_image) > thresholded_coeff

    # Reshape the mask to the original DCT image shape
    mask = mask.reshape(dct_image.shape)

    # Apply the mask to the DCT coefficients
    thresholded_dct = dct_image * mask
    thresholded_dct[~mask] = 0  # Set the coefficients below the threshold to zero

    # Return the mask and the thresholded DCT coefficients
    return mask, thresholded_dct

# Example usage
if __name__ == '__main__':
    # Load the first low res, medium res and high res images to check the size of the images
    hr_image = cv2.imread(os.path.join(HR_train_data_output_dir, HR_train_file_name), cv2.IMREAD_GRAYSCALE)
    mr_image = cv2.imread(os.path.join(MR_train_data_output_dir, MR_train_file_name), cv2.IMREAD_GRAYSCALE)
    lr_image = cv2.imread(os.path.join(LR_train_data_output_dir, LR_train_file_name), cv2.IMREAD_GRAYSCALE)
    if hr_image is None or mr_image is None or lr_image is None:
        raise FileNotFoundError("Please place the images in the correct directories.")
    
    # Read all the image shapes
    hr_shape = hr_image.shape
    mr_shape = mr_image.shape
    lr_shape = lr_image.shape
    print("High Resolution Image Shape: ", hr_shape)
    print("Medium Resolution Image Shape: ", mr_shape)
    print("Low Resolution Image Shape: ", lr_shape)

    if SHOW_DOWNSAMPLE_IMAGES:
        # Show the images in subplots with the DCT underneath and the thresholded DCT underneath
        plt.subplot(2, 3, 1)
        plt.imshow(hr_image, cmap='gray')
        plt.title('High Resolution Image')
        
        plt.subplot(2, 3, 2)
        plt.imshow(mr_image, cmap='gray')
        plt.title('Medium Resolution Image')
        
        plt.subplot(2, 3, 3)
        plt.imshow(lr_image, cmap='gray')
        plt.title('Low Resolution Image')

        plt.subplot(2, 3, 4)
        plt.imshow(hr_image, cmap='gray')
        plt.title('High Resolution Image')

        # Create downsample matrices for the images
        # For 512×640 → 128×160
        D_height = create_downsample_matrix(512, 4)  # 128×512
        D_width = create_downsample_matrix(640, 4)   # 160×640

        # For 128×160 → 32×40
        D_height2 = create_downsample_matrix(128, 4)  # 32×128
        D_width2 = create_downsample_matrix(160, 4)   # 40×160

        mr_image_down = D_height @ hr_image @ D_width.T
        plt.subplot(2, 3, 5)
        plt.imshow(mr_image_down, cmap='gray')
        plt.title('downsampled matrix Medium Resolution Image')
        
        lr_image_down = D_height2 @ mr_image @ D_width2.T
        plt.subplot(2, 3, 6)
        plt.imshow(lr_image_down, cmap='gray')
        plt.title('downsampled matrixLow Resolution Image')
        plt.show()

    # Calculate the DCT of the images
    hr_dct = cv2.dct(np.float32(hr_image))
    mr_dct = cv2.dct(np.float32(mr_image))
    lr_dct = cv2.dct(np.float32(lr_image))

    # Compress the DCT coefficients using thresholding
    print("\n ==== Compressing hr_dct coefficients ====")
    hr_mask, hr_thresholded_coeff = dct_thresholding(hr_dct, energy_threshold=1*(10e-9)) # Tuned for good contruction this is a sparsity of 1.5% of the coefficients at 27PSNR
    print("\n ==== Compressing mr_dct coefficients ====")
    mr_mask, mr_thresholded_coeff = dct_thresholding(mr_dct, energy_threshold=1*(10e-8), clip_max=10e3) # TODO: Needs tuning 4.4% of the coefficients at 25PSNR, is clipping useful?
    print("\n ==== Compressing lr_dct coefficients ====")
    lr_mask, lr_thresholded_coeff = dct_thresholding(lr_dct, energy_threshold=1*(10e-7)) # tuning is cumbersome TODO: add L-curve fitting to find the best threshold 22% at 26PSNR

    # Reconstruct the images using the thresholded DCT coefficients
    hr_reconstructed = cv2.idct(hr_dct * hr_mask)
    mr_reconstructed = cv2.idct(mr_dct * mr_mask)
    lr_reconstructed = cv2.idct(lr_dct * lr_mask)
    
    # Clip the original DCT values to (0, 1) for better visualization
    hr_dct = np.clip(hr_dct, 0, 1)
    mr_dct = np.clip(mr_dct, 0, 1)
    lr_dct = np.clip(lr_dct, 0, 1)

    # Show the images in subplots with the DCT underneath and the thresholded DCT underneath
    plt.subplot(4, 3, 1)
    plt.imshow(hr_image, cmap='gray')
    plt.title('High Resolution Image')
    
    plt.subplot(4, 3, 2)
    plt.imshow(mr_image, cmap='gray')
    plt.title('Medium Resolution Image')
    
    plt.subplot(4, 3, 3)
    plt.imshow(lr_image, cmap='gray')
    plt.title('Low Resolution Image')
    
    # Original dct images
    plt.subplot(4, 3, 4)
    plt.imshow(hr_dct, cmap='gray')
    plt.title('Clipped (0,1) DCT of High Resolution Image')
    
    plt.subplot(4, 3, 5)
    plt.imshow(mr_dct, cmap='gray')
    plt.title('Clipped (0,1) DCT of Medium Resolution Image')
    
    plt.subplot(4, 3, 6)
    plt.imshow(lr_dct, cmap='gray')
    plt.title('Clipped (0,1) DCT of Low Resolution Image')

    # Thresholded dct images
    plt.subplot(4, 3, 7)
    plt.imshow(hr_mask, cmap='gray')
    plt.title('Thresholded DCT of High Resolution Image')

    plt.subplot(4, 3, 8)
    plt.imshow(mr_mask, cmap='gray')
    plt.title('Thresholded DCT of Medium Resolution Image')

    plt.subplot(4, 3, 9)
    plt.imshow(lr_mask, cmap='gray')
    plt.title('Thresholded DCT of Low Resolution Image')
    
    # Reconstructed images
    plt.subplot(4, 3, 10)
    plt.imshow(hr_reconstructed, cmap='gray')
    plt.title('Reconstructed High Resolution Image')

    plt.subplot(4, 3, 11)
    plt.imshow(mr_reconstructed, cmap='gray')
    plt.title('Reconstructed Medium Resolution Image')

    plt.subplot(4, 3, 12)
    plt.imshow(lr_reconstructed, cmap='gray')
    plt.title('Reconstructed Low Resolution Image')
    
    # Resize the window to the size of the image
    plt.gcf().set_size_inches(18, 10)
    plt.tight_layout()
    plt.show()

    input("Press Enter to continue...")
    
    # make reconstructed image matlike uint8
    hr_reconstructed = np.clip(hr_reconstructed, 0, 255).astype(np.uint8)
    mr_reconstructed = np.clip(mr_reconstructed, 0, 255).astype(np.uint8)
    lr_reconstructed = np.clip(lr_reconstructed, 0, 255).astype(np.uint8)

    # Calculate the PSNR of the reconstructed images
    hr_psnr = cv2.PSNR(hr_image, hr_reconstructed)
    mr_psnr = cv2.PSNR(mr_image, mr_reconstructed)
    lr_psnr = cv2.PSNR(lr_image, lr_reconstructed)
    print("PSNR of High Resolution Image: ", hr_psnr)
    print("PSNR of Medium Resolution Image: ", mr_psnr)
    print("PSNR of Low Resolution Image: ", lr_psnr)
