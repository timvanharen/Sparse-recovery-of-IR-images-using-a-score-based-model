import cv2
import numpy as np

def compress_color_depth(image, strength=2):
    """
    Reduces the color depth of an image by equalizing intensity levels.

    Args:
    - image: input image as a NumPy array.
    - strength: Strength of compression of intensity levels to quantize to.

    """
    
    # measure the maximum intensity level in the image
    max_intensity = np.max(image)

    # Determine the distribution of intensity levels
    intensity_distribution = np.bincount(image.flatten(), minlength=max_intensity + 1)

    # Calculate the cumulative distribution of intensity levels
    cumulative_distribution = np.cumsum(intensity_distribution)

    # Normalize the cumulative distribution to the range [0, 255]
    normalized_distribution = (cumulative_distribution * 255) // cumulative_distribution[-1]

    # Apply the normalized distribution to the image
    compressed_image = normalized_distribution[image]
    
    return compressed_image.astype(np.uint8)

def dct_thresholding(image, energy_threshold=0.95):
    # Convert image to grayscale if it's not already
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply DCT to the image
    dct_image = cv2.dct(np.float32(gray_image))
    print("size of dct_image: ", dct_image.shape)

    # Flatten the DCT coefficients
    flat_dct = dct_image.flatten()
    print("size of flat_dct: ", flat_dct.shape)
    
    # Calculate energy of each coefficient
    dct_energy = np.abs(flat_dct)
    print("DCT energy: ", dct_energy)

    total_energy = np.sum(dct_energy)
    print("Total energy: ", total_energy)
    
    # Sort coefficients by energy (descending)
    sorted_coeffs = np.sort(dct_energy)[::-1]
    
    # Calculate cumulative energy and find the cutoff index
    cumulative_energy = np.cumsum(sorted_coeffs) / total_energy
    cutoff_index = np.argmax(cumulative_energy >= energy_threshold)
    print("Cutoff index: ", cutoff_index)
    print("Ratio of coefficients used: ", cutoff_index / len(flat_dct))
    print("cumulative energy: ", cumulative_energy)
    
    # Use the coefficients above the threshold
    thresholded_coeff = sorted_coeffs[cutoff_index]
    print("thresholded_coefff: ", thresholded_coeff)

    # Save the coefficients above the threshold
    masked_dct = flat_dct * (dct_energy > thresholded_coeff)
    print("size of masked_dct: ", masked_dct.shape)

    # # Reshape to original DCT shape
    masked_dct = masked_dct.reshape(dct_image.shape)
    
    # Reconstruct the image using inverse DCT
    idct_image = cv2.idct(masked_dct)

    # Rescale to 255 using the min and max of the image and convert back to uint8
    reconstructed_image = cv2.normalize(idct_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
    return reconstructed_image

def downsample_image(image, factor):
    # Make sure the size of DS image is an integer, multiple of 2
    DS_size = (image.shape[0]//factor,image.shape[1]//factor)
    print("Size of DS:", DS_size)
    image_DS = np.zeros((DS_size[0], DS_size[1], 3))
    print("size of image:", image.shape)
    print("size of image_DS:", image_DS.shape)

    for i in range(DS_size[0]):
        for j in range(DS_size[1]):
            image_DS[i][j] = image[i*factor][j*factor]
    print("Size of DS image: ", image_DS.shape)
    return image_DS

# Example usage
if __name__ == '__main__':
    # Read the input image
    input_image_path = 'IR_image_car_street.jpg'
    image = cv2.imread(input_image_path)
    
    # Compress an image by downsampling.
    factor = 16
    rows = image.shape[1]//factor
    cols = image.shape[0]//factor
    print("rows: ", rows)
    print("cols: ", cols)
    image_DS = cv2.resize(image, (rows, cols))
   
    # Show the image at the same window size as the original image (618, 236, 640, 512)
    cv2.imshow('Original Image', image)
    # Resize the window to the size of the image
    image_DS = cv2.resize(image_DS, (image.shape[1], image.shape[0]))
    cv2.imshow('Downsampled Image', image_DS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Perform color depth reduction
    image_depth_compressed = compress_color_depth(image, strength=50)

    # Perform DCT thresholding with 95% energy retention
    reconstructed_image = dct_thresholding(image_depth_compressed, energy_threshold=0.99)
    
    # Display or save the reconstructed image
    cv2.imshow('Original Image', image)
    cv2.imshow('Compressed colour depth Image', image_depth_compressed)
    cv2.imshow('Reconstructed Image', reconstructed_image)
    # Print the size of the window
    print("Size of the window: ", cv2.getWindowImageRect('Original Image'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
