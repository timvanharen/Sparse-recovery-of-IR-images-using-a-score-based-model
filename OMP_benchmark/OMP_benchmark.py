import numpy as np
import pandas as pd
import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
import matplotlib.pyplot as plt

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
        return torch.tensor(lr_img), torch.tensor(hr_img) # Train with dense representation / image
        return torch.tensor(lr_dct), torch.tensor(hr_dct)


if __name__ == "__main__":
    # Load data
    dataset = ImageDataset(
        low_res_dir='images/low_res_train/LR_train',
        high_res_dir='images/medium_res_train/MR_train'
    )
    
    n_components, n_features = 512, 100
    n_nonzero_coefs = 17    

    # distort the clean signal
    y_noisy = y + 0.05 * np.random.randn(len(y))

    # plot the sparse signal
    plt.figure(figsize=(7, 7))
    plt.subplot(4, 1, 1)
    plt.xlim(0, 512)
    plt.title("Sparse signal")
    plt.stem(idx, w[idx])

    # plot the noise-free reconstruction
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(X, y)
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
