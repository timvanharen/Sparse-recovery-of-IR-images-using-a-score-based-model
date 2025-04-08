import numpy as np
import pandas as pd
import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from scipy.fftpack import dctn, idctn
from scipy.sparse.linalg import LinearOperator

class ImageDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir):
        self.low_res_files = sorted(glob.glob(os.path.join(low_res_dir, '*.jpg')))
        self.high_res_files = sorted(glob.glob(os.path.join(high_res_dir, '*.jpg')))
        
        # Detect image size from first sample of the low_res directory
        sample_img = np.array(Image.open(self.low_res_files[0]))
        self.img_lr_height, self.img_lr_width = sample_img.shape[0], sample_img.shape[1]
        print(f"Detected low-resolution image size: {self.img_lr_height}x{self.img_lr_width}")

        # Detect image size from first sample of the high_res directory
        sample_img = np.array(Image.open(self.high_res_files[0]))
        self.img_hr_height, self.img_hr_width = sample_img.shape[0], sample_img.shape[1]
        print(f"Detected high-resolution image size: {self.img_hr_height}x{self.img_hr_width}")
        
        
    def __len__(self):
        return min(len(self.low_res_files), len(self.high_res_files))
    
    def __getitem__(self, idx):
        # Load image pair
        lr_img = np.array(Image.open(self.low_res_files[idx]).convert('L'))  # Grayscale
        hr_img = np.array(Image.open(self.high_res_files[idx]).convert('L'))
        
        # Normalize to [0,1]
        lr_img = lr_img.astype(np.float32) / 255.0
        hr_img = hr_img.astype(np.float32) / 255.0
        
        # Convert to DCT domain
        lr_dct = cv2.dct(np.float32(lr_img))
        hr_dct = cv2.dct(np.float32(hr_img))
        
        # Show the DCT images for debugging
        # cv2.imshow('High Res DCT', hr_dct)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # lr_dct = dctn(lr_img) # norm='ortho'
        # hr_dct = dctn(hr_img) # norm='ortho'
        #return torch.tensor(lr_img), torch.tensor(hr_img) # Train with dense representation / image
        return torch.tensor(lr_dct), torch.tensor(hr_dct), torch.tensor(lr_img), torch.tensor(hr_img)



def A_operator(x):
    """Applies A to vector x: Downsampling after inverse DCT"""
    # Reshape to 2D DCT coefficients
    x_2d = x.reshape(hr_shape)
    
    # Inverse 2D DCT (to get real-space image)
    image = idctn(x_2d)
    
    # Downsample (using simple slicing - modify if needed)
    downsampled = image[::4, ::4]  
    
    return downsampled.ravel()

def A_transpose_operator(y):
    """Applies A^T to vector y: DCT after upsampling"""
    # Reshape to 2D low-res image
    y_2d = y.reshape(lr_shape)
    
    # Upsample (with zero insertion)
    upsampled = np.zeros(hr_shape)
    upsampled[::4, ::4] = y_2d
    
    # Forward 2D DCT (to get coefficients)
    coeffs = dctn(upsampled)
    
    return coeffs.ravel()


def construct_A_explicit(m,n):
    """Constructs explicit A matrix (memory intensive!)"""
    A = np.zeros((m, n))
    basis_vec = np.zeros(n)
    
    for i in range(n):
        basis_vec[i] = 1
        A[:, i] = A_operator(basis_vec)
        basis_vec[i] = 0
        
    return A


if __name__ == "__main__":
    # Load data
    dataset = ImageDataset(
        low_res_dir='images/low_res_train/LR_train',
        high_res_dir='images/medium_res_train/MR_train'
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    n_components = dataset.img_hr_height * dataset.img_hr_width
    print(f"Number of components: {n_components}") 
    n_nonzero_coefs = dataset.img_lr_height * dataset.img_lr_width 
    print(f"Number of non-zero coefficients: {n_nonzero_coefs}")


    _,_, lr_image, hr_image = dataset[0]
    # Test with given matrix
    # Dimensions
    hr_shape = hr_image.shape  # High resolution shape (x)
    lr_shape = lr_image.shape   # Low resolution shape (y)
    n = hr_shape[0] * hr_shape[1]  # 327,680 (x size)
    m = lr_shape[0] * lr_shape[1]  # 1,280 (y size) 
    factor = hr_shape[0] // lr_shape[0]  
    print(f"Factor: {factor}")
    # # Create LinearOperator for A (memory efficient)
    # A = LinearOperator(shape=(m, n), matvec=A_operator, rmatvec=A_transpose_operator)
    # print("A.shape:", A.shape)
    # print("A:", A)

    A_exp = construct_A_explicit(m,n)  # Uncomment to construct explicit A (not recommended for large images)
    print("A_exp.shape:", A_exp.shape)
    print("A_exp:", A_exp)


    # y is the lr_dct and y_HR is the hr_dct
    for batch_idx, (y, y_HR, y_img_lr, y_img_hr) in enumerate(dataloader):
        if batch_idx == 0:
            break
    print(f"Low-res DCT image shape: {y.shape}")
    print(f"High-res DCT image shape: {y_HR.shape}")


    y = np.clip(y, -1, 1)
    y_HR = np.clip(y_HR, -1, 1)
    yFlat = y.flatten()
    
    y_HR_flat = y_HR.flatten().reshape(1, -1)
    print(f"Low-res DCT image flattened shape: {yFlat.shape}")
    print(f"High-res DCT image flattened shape: {y_HR_flat.shape}")
    # distort the clean signal
    y_noisy = yFlat + 0.05 * np.random.randn(len(y))
    X = y_HR_flat
    # plot the sparse signal
    # plt.figure(figsize=(7, 7))
    # plt.subplot(4, 1, 1)
    # plt.xlim(0, n_nonzero_coefs)
    # plt.title("Sparse signal")
    # plt.stem(yFlat[0, :])
    yFlat = yFlat.numpy()

    print("yFlat shape:", yFlat.shape)

    print(A_exp.shape)
    omp = OrthogonalMatchingPursuit(tol= 1000, n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(A_exp, yFlat)
    coef = omp.coef_
    (idx_r,) = coef.nonzero()
    plt.subplot(4, 1, 2)
    plt.xlim(0, 512)
    plt.title("Recovered signal from noise-free measurements")
    plt.stem(idx_r, coef[idx_r])

    # plot the noisy reconstruction
    omp.fit(X, y_noisy)
    coef = omp.coef_
    (idx_r,) = coef.nonzero()
    plt.subplot(4, 1, 3)
    plt.xlim(0, 512)
    plt.title("Recovered signal from noisy measurements")
    plt.stem(idx_r, coef[idx_r])

    # plot the noisy reconstruction with number of non-zeros set by CV
    omp_cv = OrthogonalMatchingPursuitCV()
    omp_cv.fit(X, y_noisy)
    coef = omp_cv.coef_
    (idx_r,) = coef.nonzero()
    plt.subplot(4, 1, 4)
    plt.xlim(0, 512)
    plt.title("Recovered signal from noisy measurements with CV")
    plt.stem(idx_r, coef[idx_r])

    plt.subplots_adjust(0.06, 0.04, 0.94, 0.90, 0.20, 0.38)
    plt.suptitle("Sparse signal recovery with Orthogonal Matching Pursuit", fontsize=16)
    plt.show()
