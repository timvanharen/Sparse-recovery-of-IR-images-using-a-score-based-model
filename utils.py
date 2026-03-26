"""Shared utility functions for sparse recovery of IR images."""

import numpy as np
from scipy.sparse import diags


def create_binomial_filter_1d(k=2):
    """Create 1D binomial filter coefficients [1,2,1]/4."""
    coeffs = np.array([1, 2, 1]) / 4
    return coeffs


def create_filter_matrix(N, filter_coeffs):
    """Create a Toeplitz-like convolution matrix for 1D filtering.
    
    Correctly applies the binomial filter as a convolution matrix with
    reflected boundary conditions.
    """
    half_len = len(filter_coeffs) // 2
    F = np.zeros((N, N))
    for i in range(N):
        for j, c in enumerate(filter_coeffs):
            idx = i + (j - half_len)
            # Reflect at boundaries
            if idx < 0:
                idx = -idx
            elif idx >= N:
                idx = 2 * (N - 1) - idx
            F[i, idx] += c
    return F


def create_decimation_matrix(N, M, factor):
    """Create decimation (subsampling) matrix of shape (M, N)."""
    S = np.zeros((M, N))
    for i in range(M):
        S[i, i * factor] = 1
    return S


def create_downsample_matrix(N, factor):
    """Combine binomial anti-aliasing filter and decimation.
    
    Returns a (N//factor, N) matrix that low-pass filters then subsamples.
    """
    filter_coeffs = create_binomial_filter_1d()
    F = create_filter_matrix(N, filter_coeffs)
    M = N // factor
    S = create_decimation_matrix(N, M, factor)
    return S @ F


def calculate_nmse(original, reconstructed):
    """Calculate Normalized Mean Squared Error."""
    mse = np.mean((original - reconstructed) ** 2)
    power = np.mean(original ** 2)
    return mse / (power + 1e-10)


def calculate_psnr(original, reconstructed, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))
