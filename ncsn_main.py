import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import glob
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global configuration variables
task = 'train' # 'test  # Task name
batch_size = 32
num_epochs = 100
learning_rate = 1e-3
num_scales = num_epochs
sigma_min = 0.01
sigma_max = 1.0
num_measurements = 32 # Number of measurements
snr_db = 20 # Signal-to-noise ratio in dB

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print("Configuration:")
for k, v in config.items():
    print(f"{k}: {v}")

# TODO:
# - Where is the gradient of the log likelihood and how can we test that it works?
# - How to add noise to the measurements? (Gaussian noise, Poisson noise, etc.)
# - How does the reconstruction work? Is it possible to compare with annealed Langevin dynamics existing code?
# - How to add the SNR to the measurements? (e.g. using a Gaussian noise model)
# - How can we show pictures of the creation process? (e.g. using matplotlib or OpenCV)
# - What happens if we use a lower resolution image? To train a bit faster?


# =====================
# Dataset Class
# =====================
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
        return torch.tensor(lr_dct), torch.tensor(hr_dct)

# =====================
# Data Generator
# =====================
class CSImageGenerator:
    def __init__(self, dataset, M, snr_db):
        self.dataset = dataset
        self.M = M          # Measurements
        self.snr_db = snr_db
        
    def make_measurements(self, lr_dct, hr_dct):
        N = hr_dct.numel()
        print(f"Generating measurements for N={N} and M={self.M}")
        # Ensure M < N
        if self.M >= N:
            raise ValueError("Number of measurements M must be less than the number of pixels N.")
        
        # Print shapes
        print(f"lr_dct shape: {lr_dct.shape}")
        print(f"hr_dct shape: {hr_dct.shape}")
        # Calculate the sensing matrix
        P = lr_dct.flatten()[:, np.newaxis]*np.linalg.pinv(hr_dct).flatten()[:, np.newaxis].T
        y = P @ hr_dct.flatten()

        # check if y and lr_dct.flatten() are equal
        if torch.allclose(y, lr_dct.flatten(), atol=1e-6):
            print("y and lr_dct.flatten() are equal.")
        else:
            print("y and lr_dct.flatten() are NOT equal.")
        
        return hr_dct.flatten(), P

        if hr_dct.is_complex():
            # Generate complex sensing matrix
            P_complex = (torch.randn(self.M, N, device=device) + 
                        1j*torch.randn(self.M, N, device=device)) / np.sqrt(2*self.M)
            print(f"Generated sensing matrix P_complex with shape: {P_complex.shape}")

            # Convert to real representation
            P_real = torch.cat([
                torch.cat([P_complex.real, -P_complex.imag], dim=1),
                torch.cat([P_complex.imag, P_complex.real], dim=1)
            ], dim=0)
            print(f"Converted sensing matrix P_real with shape: {P_real.shape}")
        else:
            # Generate real sensing matrix
            P_real = (torch.randn(self.M, N, device=device) / np.sqrt(self.M))
            print(f"Generated sensing matrix P_real with shape: {P_real.shape}")
        
        # Flatten and convert to real
        h = hr_dct.view(-1)
        h_real = torch.cat([h.real, h.imag]) if h.is_complex() else h
        print(f"Flattened h with shape: {h_real.shape}")

        # Add noise
        signal_power = torch.mean(torch.abs(h_real)**2)
        noise_power = signal_power / (10**(self.snr_db/10))
        if hr_dct.is_complex():
            noise = torch.sqrt(noise_power/2) * torch.randn(2*self.M, device=device)
        else:
            noise = torch.sqrt(noise_power) * torch.randn(self.M, device=device)
        noise = noise.view(self.M, -1)
        print(f"Generated noise with shape: {noise.shape}")

        # Measurement
        y_real = P_real @ h_real.to(device) + noise.squeeze()

        return y_real, P_real

# =====================
# NCSN Implementation
# =====================
class NoiseSchedule:
    def __init__(self, num_scales=10, sigma_min=0.01, sigma_max=1.0):
        self.sigmas = torch.exp(torch.linspace(np.log(sigma_min), 
                              np.log(sigma_max), num_scales)).to(device)
        
    def sample_sigma(self, batch_size):
        idx = torch.randint(0, len(self.sigmas), (batch_size,))
        return self.sigmas[idx]

class NCSN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.GroupNorm(8, 256),
            nn.SiLU()
        )
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, 3, padding=1, output_padding=1, stride=2),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1, output_padding=1, stride=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
        
    def forward(self, x, sigma):
        h = self.encoder(x)
        sigma_embed = self.sigma_embed(sigma.view(-1, 1))
        sigma_embed = sigma_embed.view(-1, 256, 1, 1).expand(-1, -1, h.shape[2], h.shape[3])
        h = torch.cat([h, sigma_embed], dim=1)
        return self.decoder(h)

# =====================
# Training monitoring functions
# =====================

def visualize_progress(model, test_sample, epoch):
    model.eval()
    with torch.no_grad():
        # Create noisy versions at different σ levels
        noise_levels = torch.linspace(0.1, 1.0, 5).to(device)
        noisy_samples = test_sample + torch.randn_like(test_sample) * noise_levels.view(-1, 1, 1, 1)
        
        # Predict scores
        scores = model(noisy_samples, noise_levels)
        
        # Denoise using score
        denoised = noisy_samples - scores * noise_levels.view(-1, 1, 1, 1)
        
    # Plot results
    plt.figure(figsize=(15, 3))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(denoised[i][0].cpu().numpy(), cmap='gray')
        plt.title(f"sigma={noise_levels[i]:.2f}")
    plt.suptitle(f"Epoch {epoch} Denoising Results")
    plt.show()
    model.train()

def calculate_metrics(pred, target, sigma):
    # 1. Relative Error
    rel_error = torch.mean(torch.abs(pred - target) / (torch.abs(target).mean() + 1e-5))
    
    # 2. Angle Alignment (cosine similarity)
    cosine_sim = F.cosine_similarity(pred.flatten(1), target.flatten(1))
    
    return {
        'loss': torch.mean((pred - target)**2 / sigma**2),
        'rel_error': rel_error,
        'cosine_sim': cosine_sim.mean()
    }

def plot_gradients(model):
    gradients = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append(param.grad.abs().mean().item())
    
    plt.figure(figsize=(10, 4))
    plt.plot(gradients, 'o-')
    plt.yscale('log')
    plt.title("Gradient Magnitudes")
    plt.xlabel("Parameter Index")
    plt.ylabel("Gradient (log scale)")
    plt.show()

# =====================
# Training with Noise Conditioning
# =====================
def train_ncsn(dataset, batch_size=32, num_epochs=10, num_scales=10, sigma_min=0.01, sigma_max=1.0, lr = 1e-4, loss_stopping_criterion=1.3e-6):
    print("\nTraining NCSN model...")
    noise_schedule = NoiseSchedule(num_scales=num_epochs, sigma_min=sigma_min, sigma_max=sigma_max)
    model = NCSN(dataset.img_hr_height, dataset.img_hr_width).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Increase sigma for each epoch
        sigma = torch.ones(batch_size).to(device) * noise_schedule.sigmas[epoch]

        start_time = time.time()
        for batch_idx, (_, hr_dct) in enumerate(dataloader):
            # Move data to device and add channel dimension
            hr_dct = hr_dct.unsqueeze(1).to(device)  # [B, 1, H, W]
            
            # Add noise to clean images
            noise = torch.randn_like(hr_dct)  # [B, 1, H, W]
            noisy_dct = hr_dct + noise * sigma.view(-1, 1, 1, 1)  # Scale noise by sigma

            # Compute score target (-noise/sigma)
            target = -noise / (sigma.view(-1, 1, 1, 1) + 1e-5)
            
            # Forward pass
            pred = model(noisy_dct, sigma)
            
            # Loss with 1/sigma^2 weighting (importance weighting)
            loss = torch.mean((pred - target)**2 / (sigma.view(-1, 1, 1, 1)**2))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # # After loss.backward()
            # if batch_idx % 50 == 0:
            #     plot_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # TODO: Why is this needed?
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(dataloader)} "
                      f"| Loss: {loss.item():.4f} | Sigma: {sigma.mean().item():.4f}±{sigma.std().item():.4f}")
        
        show_resized_dct_image(hr_dct[0, 0].detach().cpu().numpy(), "original DCT", wait=True)
        show_resized_dct_image(noisy_dct[0, 0].detach().cpu().numpy(), "noisy DCT", wait=True)

        # Model training evaluation
        if epoch % 1 == 0:  # Every epoch
            test_sample = dataset[0][1].unsqueeze(0).unsqueeze(1).to(device)  # Get first HR sample
            visualize_progress(model, test_sample, epoch)

            metrics = calculate_metrics(pred, target, sigma.view(-1, 1, 1, 1))
            print(f"Epoch {epoch} | Loss: {metrics['loss']:.4f} | "
                f"RelError: {metrics['rel_error']:.4f} | "
                f"CosSim: {metrics['cosine_sim']:.4f}")

        # Epoch summary
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s | "
              f"Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'ncsn_checkpoint_epoch{epoch+1}.pth')

    return model

# =====================
# Annealed Langevin Dynamics
# =====================
def annealed_langevin_dynamics(y, P, model, img_shape, steps_per_noise=100):
    noise_schedule = NoiseSchedule()
    current_h = torch.randn(1, 1, *img_shape, device=device, requires_grad=True)
    show_resized_dct_image(current_h[0, 0].detach().cpu().numpy(), "initial DCT", wait=True)
    # Store reconstruction process
    reconstruction_process = []
    
    for sigma in reversed(noise_schedule.sigmas):
        # Step size depends on noise level
        step_size = 0.25/steps_per_noise * sigma**2 #* 0.01  # Adjust this value as needed
        
        for step in range(steps_per_noise):
            # Data fidelity term
            h_flat = current_h.view(-1)
            residual = y - P @ h_flat
            grad_likelihood = P.T @ residual

            # Prior term
            with torch.no_grad():
                score = model(current_h, sigma * torch.ones(1, device=device))

            # Langevin update
            noise_term = torch.sqrt(2 * step_size) * torch.randn_like(current_h)

            current_h = current_h + step_size * (score + grad_likelihood.view_as(current_h)) + noise_term
            
            if step % 20 == 0:
                reconstruction_process.append(current_h.detach().cpu().numpy())
    
    # Final reconstruction
    reconstructed_dct = current_h.squeeze().detach().cpu().numpy()
    reconstructed_img = cv2.idct(reconstructed_dct)
    return np.clip(reconstructed_img, 0, 1), reconstruction_process

# =====================
# Evaluation Metrics
# =====================
def calculate_nmse(original, reconstructed):
    """Calculate Normalized Mean Squared Error"""
    mse = np.mean((original - reconstructed)**2)
    power = np.mean(original**2)
    return mse / (power + 1e-10)

def psnr(original, reconstructed):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - reconstructed)**2)
    return 20 * np.log10(1.0 / np.sqrt(mse + 1e-10))

def reconstruct_and_evaluate(y, P, model, hr_dct):
    """Evaluate reconstruction method"""
    results = {}
    # NCSN method
    start = time.time()
    ncsn_recon, process = annealed_langevin_dynamics(y, P, model, hr_dct.shape)
    results['ncsn'] = {
        'time': time.time() - start,
        'nmse': calculate_nmse(hr_dct.numpy(), ncsn_recon),
        'image': ncsn_recon,
        'process': process
    }
    
    return results

# =====================
# 5. Visualization
# =====================
def plot_results(results, hr_dct):
    plt.figure(figsize=(18, 6))

    # NCSN
    plt.subplot(1, 3, 2)
    plt.imshow(results['ncsn']['image'], cmap='gray')
    plt.title(f"NCSN Method\nNMSE: {results['ncsn']['nmse']:.4f}")
    
    # Ground Truth
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.idct(hr_dct.numpy()), cmap='gray')
    plt.title("Ground Truth")
    
    plt.tight_layout()
    plt.savefig('comparison_results.jpg', dpi=300)
    plt.show()
    
    # Plot reconstruction process
    if 'process' in results['ncsn']:
        plot_reconstruction_process(results['ncsn']['process'])

def plot_reconstruction_process(process):
    plt.figure(figsize=(12, 8))
    for i, img in enumerate(process[:10]):  # Show first 10 steps
        plt.subplot(2, 5, i+1)
        plt.imshow(cv2.idct(img.squeeze()), cmap='gray')
        plt.title(f"Step {i*20}")
        plt.axis('off')
    plt.suptitle("Reconstruction Process")
    plt.tight_layout()
    plt.savefig('reconstruction_process.jpg', dpi=300)
    plt.show()

def show_resized_dct_image(img_dct, title, wait=False):
    # let's upscale the image using new  width and height
    image = cv2.idct(img_dct)
    up_width = 600
    up_height = 400
    up_points = (up_width, up_height)
    resized_up = cv2.resize(image, up_points, interpolation= cv2.INTER_LINEAR)
    resized_up_DCT = cv2.resize(img_dct, up_points, interpolation= cv2.INTER_LINEAR)

    resized_up_DCT = np.clip(resized_up_DCT, 0, 1)

    plt.subplot(1, 2, 1)
    plt.imshow(resized_up, cmap='gray', norm=None)
    plt.title(title)
    plt.subplot(1, 2, 2)
    plt.imshow(resized_up_DCT, cmap='gray', norm=None)
    plt.title(title + ' DCT')
    plt.savefig('resized_image_and_dct.jpg', dpi=300)
    if wait:
        plt.show()

# =====================
# Main Workflow
# =====================
if __name__ == "__main__":
    # Load data
    dataset = ImageDataset(
        low_res_dir='images/low_res_train/LR_train',
        high_res_dir='images/medium_res_train/MR_train'
    )
    
    # Train ncsn model
    if task == 'train':
        print("Training score model...")
        ncsn_model = train_ncsn(dataset, batch_size=batch_size, lr=learning_rate, num_epochs=num_epochs, 
                                num_scales=num_scales, sigma_min=sigma_min, sigma_max=sigma_max)
        torch.save(ncsn_model.state_dict(), 'ncsn_model.pth')
    else: # Load pre-trained model
        print("Loading pre-trained score model...")
        ncsn_model = NCSN(dataset.img_hr_height, dataset.img_hr_width).to(device)
        ncsn_model.load_state_dict(torch.load('ncsn_model.pth'))
        ncsn_model.eval()
    
    # Test reconstruction
    test_generator = CSImageGenerator(dataset, M=num_measurements, snr_db=snr_db)
    lr_dct, hr_dct = dataset[0]
    y, P = test_generator.make_measurements(lr_dct, hr_dct)
    
    # Evaluate both methods
    results = reconstruct_and_evaluate(y, P, ncsn_model, hr_dct)

    # Print metrics
    print("\nEvaluation Results:")
    print(f"NCSN Method - NMSE: {results['ncsn']['nmse']:.4f}, Time: {results['ncsn']['time']:.2f}s")
    
    # Visualize
    plot_results(results, hr_dct)