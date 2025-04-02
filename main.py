import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import OrthogonalMatchingPursuit
# from scipy.fft import dctn, idctn
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import glob
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global configuration variables
task = 'train' # 'test  # Task name
batch_size = 32
num_epochs = 10
learning_rate = 1e-3
num_measurements = 32

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
# 1. Image Data Loading
# =====================
class ImageDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None):
        self.low_res_files = sorted(glob.glob(os.path.join(low_res_dir, '*.jpg')))
        self.high_res_files = sorted(glob.glob(os.path.join(high_res_dir, '*.jpg')))

        # Detect image size from first sample of the low_res directory
        sample_img = np.array(Image.open(self.low_res_files[0]))
        self.img_lr_height, self.img_lr_width = sample_img.shape[0], sample_img.shape[1]
        print(f"Detected low-resolution image size: {self.img_lr_height}x{self.img_lr_width}")

        # Detect image size from first sample of the high_res directory
        sample_img = np.array(Image.open(self.high_res_files[0]))
        self.img_hr_height, self.img_hr_width = sample_img.shape[0], sample_img.shape[1]
        print(f"Detected high-resolution image size: {self.img_lr_height}x{self.img_lr_width}")
        self.transform = transform
        
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
        
        if self.transform:
            lr_dct = self.transform(lr_dct)
            hr_dct = self.transform(hr_dct)
            
        return torch.tensor(lr_dct), torch.tensor(hr_dct)

# =====================
# 2. Data Generator (Modified for Images)
# =====================
class CSImageGenerator:
    def __init__(self, dataset, M, snr_db):
        self.dataset = dataset
        self.M = M          # Measurements
        self.snr_db = snr_db
        
    def make_measurements(self, hr_dct):
        N = hr_dct.numel()
        print(f"Generating measurements for N={N} and M={self.M}")
        # Ensure M < N
        if self.M >= N:
            raise ValueError("Number of measurements M must be less than the number of pixels N.")
        
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
        y_real = P_real @ h_real.to(device) + noise
        return y_real, P_real

# =====================
# 3. Score Model (Enhanced for Images)
# =====================
class ImageScoreNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # convolutional layers using input_dim as the number of input channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.Softplus(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Softplus(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Softplus()
        )

        self.fc = nn.Sequential(
            nn.Linear(128 + 1, 256),
            nn.Softplus(),
            nn.Linear(256, 128),
            nn.Softplus()
        )
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.Softplus(),
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.Softplus(),
            nn.ConvTranspose2d(32, 1, 3, padding=1)
        )
        
    def forward(self, x, sigma):
        # x shape: (batch, channels, height, width)
        sigma = sigma.view(-1, 1)
        x = self.conv(x)
        # print("x shape after conv:", x.shape)
        # input("Press Enter to continue...")
        # Global average pooling + sigma conditioning
        #x_pooled = torch.mean(x, dim=(0), keepdim=True)
        x_pooled = torch.mean(x, dim=(2,3))
        # print("x_pooled shape after conv:", x_pooled.shape)
        # print("sigma shape after conv:", sigma.shape)
        # print("Shape of concatenated tensor:", torch.cat([x_pooled, sigma], dim=1).shape)
        # input("Press Enter to continue...")
        x_cond = self.fc(torch.cat([x_pooled, sigma], dim=1))
        x_cond = x_cond.view(x.shape[0], -1, 1, 1)
        
        # Broadcast conditioning
        x = x + x_cond
        return self.deconv(x)

# =====================
# 4. Training Loop (Image-optimized)
# =====================
def train_score_model(dataset, batch_size=32, num_epochs=10, lr = 1e-4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ImageScoreNet(input_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    loss_stopping_criterion = 1.3e-6
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        runs = 0
        # Print the size of one epoch
        print(f"Epoch {epoch+1}/{num_epochs} - Batch size: {batch_size} - Dataset size: {len(dataset)}")
        for lr_dct, hr_dct in dataloader:

            # Move to device and add channel dimension
            hr_dct = hr_dct.unsqueeze(1).to(device)  # (batch, 1, h, w)
            # print("hr_dct.shape:", hr_dct.shape)
            # show_resized_dct_image(hr_dct[0, 0].cpu().numpy(), "hr_dct")

            # Add noise
            sigma = torch.rand(hr_dct.shape[0], device=device) * 0.2
            # print("sigma.shape:", sigma.shape)

            noise = torch.randn_like(hr_dct) * sigma.view(-1, 1, 1, 1)
            # print("noise.shape:", noise.shape)

            noisy_dct = hr_dct + noise
            # print("noisy_dct shape:", noisy_dct.shape)

            # show_resized_dct_image(noisy_dct[0, 0].cpu().numpy(), "Noisy")

            # Score matching
            target = -noise / (sigma.view(-1, 1, 1, 1) + 1e-5)
            # print("target shape:", target.shape)
            # show_resized_dct_image(target[0, 0].cpu().numpy(), "target", wait=True)

            pred = model(noisy_dct, sigma)
            loss = torch.mean((pred - target)**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Print the loss and time per batch
            print(f"Epoch {epoch+1}/{num_epochs}, run: {runs}/{len(dataset)//batch_size}, Batch Loss: {loss.item():.4f}")
            runs += 1
            if loss.item() < loss_stopping_criterion:
                print(f"Stopping early at epoch {epoch+1}, run {runs}/{len(dataset)//batch_size} with loss: {loss.item():.4f}")
                break

        
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss/len(dataloader):.4f}")
        if epoch_loss/len(dataloader) < loss_stopping_criterion:
            print(f"Stopping early at epoch {epoch+1} with average loss: {epoch_loss/len(dataloader):.4f}")
            break
    return model

# =====================
# 5. Image Reconstruction
# =====================
def reconstruct_image(y, P, model, img_shape, steps=200):
    current_h = torch.randn(1, 1, *img_shape, device=device, requires_grad=True)
    optimizer = optim.Adam([current_h], lr=0.01)
    
    for step in range(steps):
        # Flatten for measurement
        h_flat = current_h.view(-1)
        h_real = torch.cat([h_flat.real, h_flat.imag]) if h_flat.is_complex() else h_flat
        
        # Data fidelity term
        residual = y - P @ h_real
        data_loss = torch.sum(residual**2)
        
        # Prior term
        sigma = torch.tensor(0.1, device=device)
        score = model(current_h, sigma)
        prior_loss = torch.sum(current_h * score)
        
        total_loss = data_loss + prior_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step} Loss: {total_loss.item():.4f}")
    
    # Convert back to image space
    reconstructed_dct = current_h.squeeze().detach().cpu().numpy()
    reconstructed_img = cv2.idct(reconstructed_dct)  # Inverse DCT
    #reconstructed_img = idctn(reconstructed_dct, norm='ortho')
    return np.clip(reconstructed_img, 0, 1)

def show_resized_dct_image(img_dct, title, wait=False):
    # let's upscale the image using new  width and height
    image = cv2.idct(img_dct)
    up_width = 600
    up_height = 400
    up_points = (up_width, up_height)
    resized_up = cv2.resize(image, up_points, interpolation= cv2.INTER_LINEAR)
    resized_up_DCT = cv2.resize(img_dct, up_points, interpolation= cv2.INTER_LINEAR)
    cv2.imshow(title, resized_up)
    cv2.imshow(title + ' DCT', resized_up_DCT)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# =====================
# Main Workflow
# =====================
if __name__ == "__main__":
    # 1. Load datasets
    train_dataset = ImageDataset(
        low_res_dir='images/low_res_train/LR_train',
        high_res_dir='images/medium_res_train/MR_train' #high_res_dir='images/high_res_train/HR_train'
    )
    
    test_dataset = ImageDataset(
        low_res_dir='images/low_res_test/LR_test',
        high_res_dir='images/medium_res_test/MR_test' #high_res_dir='images/high_res_test/HR_test'
    )

    # Print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # 2. Train score model
    if task == 'train':
        print("Training score model...")
        score_model = train_score_model(train_dataset, batch_size=batch_size, lr=learning_rate)
        torch.save(score_model.state_dict(), 'image_score_model.pth')
    else: # Load pre-trained model
        print("Loading pre-trained score model...")
        score_model = ImageScoreNet(input_dim=128).to(device)
        score_model.load_state_dict(torch.load('image_score_model.pth'))
        score_model.eval()
    
    # 3. Test reconstruction
    test_generator = CSImageGenerator(test_dataset, M=num_measurements, snr_db=20) #batch_size=32, 
    lr_dct, hr_dct = test_dataset[0]  # Get first test sample
    print(f"Low Res DCT shape: {lr_dct.shape}, High Res DCT shape: {hr_dct.shape}")

    # Create measurements
    y, P = test_generator.make_measurements(hr_dct)
    
    # Reconstruction
    print("\nReconstructing image...")
    start_time = time.time()
    reconstructed_img = reconstruct_image(y, P, score_model, hr_dct.shape)
    print(f"Reconstruction took {time.time()-start_time:.2f} seconds")
    
    # Visualize
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.idct(lr_dct.numpy()))
    plt.title('Low Resolution Input')
    
    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title('Reconstructed Image')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.idct(hr_dct.numpy()))
    plt.title('Ground Truth High Res')
    
    plt.tight_layout()
    plt.savefig('reconstruction_comparison.jpg', dpi=300)
    plt.show()