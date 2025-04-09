import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import orthogonal_mp
from scipy.fftpack import dct, idct
from pathlib import Path
import torch
import torch.fft
import time

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def dct2d_dict(block_size):
    """Build DCT dictionary for given block size using PyTorch"""
    P = block_size[0]
    Q = block_size[1]
    n = P * Q
    D = []
    
    for i in range(P):
        for j in range(Q):
            basis = torch.zeros(block_size)
            basis[i, j] = 1.0
            # 2D DCT using torch.fft is not dct
            #dct_basis = torch.fft.fftn(basis, norm='ortho').real
            basis = basis.cpu().numpy()
            dct_basis = dct(dct(basis.T, norm='ortho').T, norm='ortho')
            dct_basis = torch.from_numpy(dct_basis).float().to(device)
            D.append(dct_basis.flatten())
    
    # Stack and move to GPU
    D = torch.stack(D, dim=1).to(device)
    return D  # shape: (n, n)

# ========
#  ariellubonja implementation of OMP
# ========
# This is a modified version of the original code to work with PyTorch tensors and GPU.
def innerp(x, y=None, out=None):
    if y is None:
        y = x
    if out is not None:
        out = out[:, None, None]  # Add space for two singleton dimensions.
    return torch.matmul(x[..., None, :], y[..., :, None], out=out)[..., 0, 0]

def cholesky_solve(ATA, ATy):
    if ATA.dtype == torch.half or ATy.dtype == torch.half:
        return ATy.to(torch.float).cholesky_solve(torch.cholesky(ATA.to(torch.float))).to(ATy.dtype)
    return ATy.cholesky_solve(torch.cholesky(ATA)).to(ATy.dtype)

def omp_v0(X, y, XTX, n_nonzero_coefs=None, tol=None, inverse_cholesky=True):
    B = y.shape[0]
    normr2 = innerp(y)  # Norm squared of residual.
    projections = (X.transpose(1, 0) @ y[:, :, None]).squeeze(-1)
    sets = y.new_zeros(n_nonzero_coefs, B, dtype=torch.int64)

    if inverse_cholesky:
        # Doing the inverse-cholesky iteratively uses more memory,
        # but takes less time than waiting till solving the problem in the end it seems.
        # (Of course this may also just be because we have not optimized it extensively,
        #  but also since F is triangular it could be faster to multiply, prob. not on GPU tho.)
        F = torch.eye(n_nonzero_coefs, dtype=y.dtype, device=y.device).repeat(B, 1, 1)
        a_F = y.new_zeros(n_nonzero_coefs, B, 1)

    D_mybest = y.new_empty(B, n_nonzero_coefs, XTX.shape[0])
    temp_F_k_k = y.new_ones((B, 1))

    if tol:
        result_lengths = sets.new_zeros(y.shape[0])
        result_solutions = y.new_zeros((y.shape[0], n_nonzero_coefs, 1))
        finished_problems = sets.new_zeros(y.shape[0], dtype=torch.bool)

    for k in range(n_nonzero_coefs+bool(tol)):
        # STOPPING CRITERIA
        if tol:
            problems_done = normr2 <= tol
            if k == n_nonzero_coefs:
                problems_done[:] = True

            if problems_done.any():
                new_problems_done = problems_done & ~finished_problems
                finished_problems.logical_or_(problems_done)
                result_lengths[new_problems_done] = k
                if inverse_cholesky:
                    result_solutions[new_problems_done, :k] = F[new_problems_done, :k, :k].permute(0, 2, 1) @ a_F[:k, new_problems_done].permute(1, 0, 2)
                else:
                    assert False, "inverse_cholesky=False with tol != None is not handled"
                if problems_done.all():
                    return sets.t(), result_solutions, result_lengths

        sets[k] = projections.abs().argmax(1)
        # D_mybest[:, k, :] = XTX[gamma[k], :]  # Same line as below, but significantly slower. (prob. due to the intermediate array creation)
        torch.gather(XTX, 0, sets[k, :, None].expand(-1, XTX.shape[1]), out=D_mybest[:, k, :])
        if k:
            D_mybest_maxindices = D_mybest.permute(0, 2, 1)[torch.arange(D_mybest.shape[0], dtype=sets.dtype, device=sets.device), sets[k], :k]
            torch.rsqrt(1 - innerp(D_mybest_maxindices), out=temp_F_k_k[:, 0])  # torch.exp(-1/2 * torch.log1p(-inp), temp_F_k_k[:, 0])
            D_mybest_maxindices *= -temp_F_k_k  # minimal operations, exploit linearity
            D_mybest[:, k, :] *= temp_F_k_k
            D_mybest[:, k, :, None].baddbmm_(D_mybest[:, :k, :].permute(0, 2, 1), D_mybest_maxindices[:, :, None])


        temp_a_F = temp_F_k_k * torch.gather(projections, 1, sets[k, :, None])
        normr2 -= (temp_a_F * temp_a_F).squeeze(-1)
        projections -= temp_a_F * D_mybest[:, k, :]
        if inverse_cholesky:
            a_F[k] = temp_a_F
            if k:  # Could maybe a speedup from triangular mat mul kernel.
                torch.bmm(D_mybest_maxindices[:, None, :], F[:, :k, :], out=F[:, k, None, :])
                F[:, k, k] = temp_F_k_k[..., 0]
    else:
        if inverse_cholesky:
            solutions = F.permute(0, 2, 1) @ a_F.squeeze(-1).transpose(1, 0)[:, :, None]
        else:
            AT = X.T[sets.T]
            solutions = cholesky_solve(AT @ AT.permute(0, 2, 1), AT @ y.T[:, :, None])

    return sets.t(), solutions, None

#  ================

# Slow and inaccurate implementation of OMP
def omp(A, y, k, tol=1e-6):
    """
    Orthogonal Matching Pursuit
    A: measurement matrix (m x n)
    y: measurement vector (m)
    k: sparsity level
    tol: residual tolerance
    """
    m, n = A.shape
    residual = y.clone()
    idx = []
    x_hat = torch.zeros(n, device=device)
    
    start_time = time.time()
    iter_time = time.time()
    for _ in range(k):
        # Find the atom most correlated with residual
        corr = torch.abs(A.T @ residual)

        # Avoid already selected atoms
        if len(idx) > 0:
            corr[idx] = -1
        new_idx = torch.argmax(corr).item()

        # Check if correlation is too small
        if corr[new_idx] < tol:
            break
            
        idx.append(new_idx)
        
        # Solve least squares problem with selected atoms
        A_selected = A[:, idx]
        least_sq_time = time.time()
        x_ls = torch.linalg.lstsq(A_selected, y).solution
        print(f"Least squares time: {time.time() - least_sq_time:.4f} seconds")
        
        # Update residual
        residual = y - A_selected @ x_ls

        # Print per iteration the time
        print("Iteration:", len(idx), "Time elapsed:", time.time() - iter_time)
        iter_time = time.time()
    print("Total time elapsed:", time.time() - start_time)
    # Construct solution
    x_hat = torch.zeros(n, device=device)
    if len(idx) > 0:
        x_hat[idx] = x_ls
    
    return x_hat

def compressed_sensing_block_reconstruct(y, Phi, D, k):
    """ reconstruction using the OMP implementation"""
    A = Phi @ D  # measurement matrix
    #x_hat = omp(A, y, k)
    
    # Convert to numpy for sklearn's OMP (could implement custom GPU OMP, but it's not trivial and results are slow and inaccurate)
    A_np = A.cpu().numpy()
    y_np = y.cpu().numpy()
    
    x_hat = orthogonal_mp(A_np, y_np, n_nonzero_coefs=k)  # sparse code
    #x_hat = omp_v0(A_np, y_np, A_np.T @ A_np, n_nonzero_coefs=k, tol=None, inverse_cholesky=True)
    x_hat = torch.from_numpy(x_hat).float().to(device)

    recon_block = D @ x_hat
    return recon_block

def block_process_cs(img, block_size, m, k, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    h, w = img.shape
    n = block_size[0] * block_size[1]
    D = dct2d_dict(block_size)
    print("D shape", D.shape)
    
    # Initialize reconstruction on GPU
    recon = torch.zeros((h, w), dtype=torch.float32, device=device)

    for i in range(0, h, block_size[0]):
        for j in range(0, w, block_size[1]):
            block = img[i:i+block_size[0], j:j+block_size[1]]
            if block.shape != (block_size[0], block_size[1]):
                continue
            
            # Move data to GPU
            x = torch.from_numpy(block.flatten().astype(np.float32)).to(device)

            # Generate measurement matrix on GPU
            Phi = torch.randn(m, n, device=device)
            Phi = Phi / torch.norm(Phi, dim=1, keepdim=True)  # normalize rows

            # Simulate compressed measurement y
            y = Phi @ x

            # Reconstruct
            recon_block = compressed_sensing_block_reconstruct(y, Phi, D, k)
            recon_block = recon_block.reshape((block_size[0], block_size[1]))

            # Place reconstructed block back
            recon[i:i+block_size[0], j:j+block_size[1]] = recon_block

    # Move result back to CPU for visualization
    return torch.clamp(recon, 0, 255).cpu().numpy().astype(np.uint8)

def show_images(imgs, titles):
    plt.figure(figsize=(14, 5))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- Main ---

start_time = time.time()
print("Starting compressed sensing reconstruction...")
# Load original image
image_path = Path('medium_res_train_0.jpg')
original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if original is None:
    raise FileNotFoundError("Please place 'original.png' in the current directory.")

block_size_height = original.shape[0]
block_size_width = original.shape[1]
block_size = (block_size_height, block_size_width)
n = block_size_height * block_size_width
# sparsity level
# From analysis: 
# low res: 32x40 (1280) * 0.2234 = 285.12 (285)
# medium res: 128*160 (20480) * 0.0446 = 912.64 (913)
# high res: 512*640 (327680) * 0.0129 = 4915.2 (4916)
k = 2000
m = k * 4 # Best we can do it seems

# PRINT ALL THE VARIABLES
print("Block size:", block_size)
print("m (measurements):", m)
print("k (sparsity level):", k)

reconstructed = block_process_cs(original, block_size, m, k)
print("Reconstruction completed.")
print("Total time elapsed:", time.time() - start_time)

# Calculate PSNR
psnr = cv2.PSNR(original, reconstructed)
print(f"PSNR: {psnr:.2f} dB")

# Show result
show_images([original, reconstructed], ["Original", "Compressed Sensing Reconstruction"])