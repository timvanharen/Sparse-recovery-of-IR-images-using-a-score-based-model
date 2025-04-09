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
import functools
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

USE_NCSN_MODEL = False  # Set to False to use ScoreNet instead

# Global configuration variables
task = 'train' #'train' # 'test  # Task name
batch_size = 32
num_epochs = 5
learning_rate = 1e-4
num_scales = num_epochs
sigma_min = 0.01
sigma_max = 1.0
num_measurements = 32 # Number of measurements
snr_db = 20 # Signal-to-noise ratio in dB
anneal_power = 2. # Annealing power for Langevin dynamics

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print("Configuration:")
for k, v in config.items():
    print(f"{k}: {v}")

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

# TODO:
# - Where is the gradient of the log likelihood and how can we test that it works?
# - How to add noise to the measurements?
# - How does the reconstruction work? Is it possible to compare with annealed Langevin dynamics existing code?
# - How to add the SNR to the measurements? (e.g. using a Gaussian noise model)
# - How can we show pictures of the creation process? (e.g. using matplotlib or OpenCV)
# - (Done) What happens if we use a lower resolution image? To train a bit faster?

# - (Done) FInd out which NN architerture must be used for the NCSN model (e.g. UNet, ResNet, etc.)

# - Find out why:
#    - gradients are only in one axis direction, the should be vectors in all kinds of directions right?
#    - the output image shows a cross in the middle of the image


# TODO: Sturcture the code base and remove unused code and files
# - why use glob?
# - Save analytic images in a dedicated folder (e.g. 'images/analytic_images')
# - Save the model in a dedicated folder (e.g. 'models/ncsn_model.pth')
# - Save checkpoints during training and save in a dedicated folder (e.g. 'checkpoints/')
# - Fix the names of the h and y bitches and lr_dct etc.
# - Structure the code base and remove unused code and files

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
        lr_img = lr_img.astype(np.float32) / 255 # TODO: ? lr_img.max()
        hr_img = hr_img.astype(np.float32) / 255 # TODO: ? hr_img.max()
        
        # # Convert to DCT domain
        # lr_dct = cv2.dct(np.float32(lr_img))
        # hr_dct = cv2.dct(np.float32(hr_img))

        return torch.tensor(lr_img), torch.tensor(hr_img) # Train with dense representation / image

# =====================
# Data Generator
# =====================
class CSImageGenerator:
    def __init__(self, dataset, M, snr_db):
        self.dataset = dataset
        self.M = M          # Measurements
        self.snr_db = snr_db
        
    def make_measurements(self, lr_img, hr_img):
        print("lr_img.shape:", lr_img.shape)
        N = lr_img.shape[0]*lr_img.shape[1]  # Number of pixels in the image

        print(f"Generating measurements for N={N} and M={self.M}")
        # Ensure M < N
        if self.M >= N:
            raise ValueError("Number of measurements M must be less than the number of pixels N.")
        
        # Print shapes
        print(f"lr_img shape: {lr_img.shape}")
        print(f"hr_img shape: {hr_img.shape}")

        # Calculate the sensing matrix
        # First move the lr_img to a tensor and device
        lr_img = torch.tensor(lr_img, dtype=torch.float32).to(device)  # Convert to tensor and move to device
        y = lr_img.flatten()[:, torch.newaxis].to(device)  # Flatten and move to device
        h = hr_img.flatten()[:, torch.newaxis].to(device)  # Flatten and move to device
        P = (y*torch.linalg.pinv(h)).to(device)  # Pseudo-inverse for sensing matrix
        #y = P @ h #TODO: Add noise # P Must be a downscaling measurement matrix.

        return y.squeeze(), P

# =====================
# ScoreNet Implementation
# =====================
PRINT_SIZE = False

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]

class DynamicPadCat(nn.Module):
    def forward(self, x1, x2):
        # x1: decoder feature, x2: encoder feature
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
        diffY // 2, diffY - diffY // 2])
        return torch.cat([x1, x2], dim=1)

class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, img_height, img_width, channels=[16, 32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
            marginal_prob_std: A function that takes time t and gives the standard
                deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
            channels: The number of channels for feature maps of each resolution.
            embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        self.channels = channels
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
        nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=4, stride=2, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=4, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=4, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        self.conv5 = nn.Conv2d(channels[3], channels[4], kernel_size=2, stride=2, padding=0, bias=False)
        self.dense5 = Dense(embed_dim, channels[4])
        self.gnorm5 = nn.GroupNorm(32, num_channels=channels[4])

        # Decoding layers where the resolution increases
        self.tconv5 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2, padding=0, bias=False)

        self.dense6 = Dense(embed_dim, channels[3])
        self.tgnorm5 = nn.GroupNorm(32, num_channels=channels[3])
        self.tconv4 = nn.ConvTranspose2d(channels[3]*2, channels[2], kernel_size=4, stride=2, padding=1, bias=False)

        self.dense7 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2]*2, channels[1], kernel_size=4, stride=2, padding=1, bias=False)

        self.dense8 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1]*2, channels[0], kernel_size=4, stride=2, padding=1, bias=False)

        self.dense9 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(16, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0]*2, 1, kernel_size=4, stride=2, padding=1)
        # Modify transpose convs for non-square outputs
        self.padcat = DynamicPadCat()

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t

        embed = self.act(self.embed(t))

        # Encoding path
        h1 = self.conv1(x)

        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)

        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)

        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)

        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)
        h5 = self.conv5(h4)

        h5 += self.dense5(embed)
        h5 = self.gnorm5(h5)
        h5 = self.act(h5)

        if PRINT_SIZE:
            print("t.shape:", t.shape)
            print("x.shape:", x.shape)  
            print("embed.shape:", embed.shape)
            print("h1.shape:", h1.shape)
            print("h2.shape:", h2.shape)
            print("h3.shape:", h3.shape)
            print("h4.shape:", h4.shape)
            print("h5.shape:", h5.shape)

        # DeConv 5
        h = self.tconv5(h5)
        if PRINT_SIZE: print("h.shape: self.tconv5(h5)", h.shape)

        h += self.dense6(embed)
        if PRINT_SIZE:  print("h.shape self.dense5(embed):", h.shape)

        h = self.tgnorm5(h)
        if PRINT_SIZE:  print("h.shape: self.tgnorm5(h)", h.shape)

        h = self.act(h)
        if PRINT_SIZE:  print("h.shape: self.act(h)", h.shape)

        # DeConv 4
        h = self.tconv4(self.padcat(h, h4))  # Modified
        if PRINT_SIZE:  print("h.shape: self.tconv4(self.padcat(h, h4))  # Modified", h.shape)

        h += self.dense7(embed)
        if PRINT_SIZE:  print("h.shape: self.dense6(embed)", h.shape)

        h = self.tgnorm4(h)
        if PRINT_SIZE:  print("h.shape: self.tgnorm4(h)", h.shape)

        h = self.act(h)
        if PRINT_SIZE:  print("h.shape: self.act(h)", h.shape)

        # DeConv 3
        h = self.tconv3(self.padcat(h, h3))  # Modified
        if PRINT_SIZE:  print("h.shape: self.tconv3(self.padcat(h, h3))", h.shape)

        h += self.dense8(embed)
        if PRINT_SIZE:  print("h.shape: self.dense7(embed)", h.shape)

        h = self.tgnorm3(h)
        if PRINT_SIZE:  print("h.shape: self.tgnorm3(h)", h.shape)

        h = self.act(h)
        if PRINT_SIZE:  print("h.shape: self.act(h)", h.shape)

        # DeConv 2
        h = self.tconv2(self.padcat(h, h2))  # Modified
        if PRINT_SIZE:  print("h.shape: self.tconv2(self.padcat(h, h2))", h.shape)

        h += self.dense9(embed)
        if PRINT_SIZE:  print("h.shape: self.dense8(embed)", h.shape)

        h = self.tgnorm2(h)
        if PRINT_SIZE:  print("h.shape: self.tgnorm2(h)", h.shape)

        h = self.act(h)
        if PRINT_SIZE:  print("h.shape: self.act(h)", h.shape)

        # DeConv 1
        h = self.tconv1(self.padcat(h, h1))  # Modified
        if PRINT_SIZE:  print("h.shape: self.tconv1(self.padcat(h, h1))  # Modified", h.shape)

        return h / self.marginal_prob_std(t)[:, None, None, None]

def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

    Returns:
        The standard deviation.
    """
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

    Returns:
        The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)

sigma =  25.0#@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
        model: A PyTorch model instance that represents a
            time-dependent score-based model.
        x: A mini-batch of training data.
        marginal_prob_std: A function that gives the standard deviation of
            the perturbation kernel.
        eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    return loss

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
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.GroupNorm(8, 256),
            nn.ReLU()
        )
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, 3, padding=1, output_padding=1, stride=2),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1, output_padding=1, stride=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
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
def train_ncsn(dataset, batch_size=32, num_epochs=10, num_scales=10, sigma_min=0.01, sigma_max=1.0, lr = 1e-4, loss_stopping_criterion=1.3e-6, anneal_power=2.):
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
        for batch_idx, (_, hr_img) in enumerate(dataloader):
            # Move data to device and add channel dimension
            hr_img = hr_img.unsqueeze(1).to(device)  # [B, 1, H, W]
            
            # Add noise to clean images
            noise = torch.randn_like(hr_img)  # [B, 1, H, W]
            noisy_dct = hr_img + noise * sigma.view(-1, 1, 1, 1)  # Scale noise by sigma

            # Compute score target (-noise/sigma)
            target = -noise / (sigma.view(-1, 1, 1, 1)**2) # + 1e-5 # Avoid division by zero
            
            # Forward pass
            pred = model(noisy_dct, sigma)
            
            # Loss with 1/sigma^2 weighting (importance weighting)
            loss = torch.mean(1/2.* ((pred - target)**2).sum(dim = (2,3)) * sigma ** anneal_power)
            #loss = torch.mean((pred - target)**2 / (sigma.view(-1, 1, 1, 1)**2))
            
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
        
        show_resized_dct_image(hr_img[0, 0].detach().cpu().numpy(), "train original", wait=True)
        show_resized_dct_image(noisy_dct[0, 0].detach().cpu().numpy(), "train noisy", wait=True)

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
def annealed_langevin_dynamics(y, P, model, img_shape, num_epochs, num_scales, sigma_min, sigma_max, anneal_power, steps_per_noise=10):
    noise_schedule = NoiseSchedule(num_scales=num_epochs, sigma_min=sigma_min, sigma_max=sigma_max)
    current_h = torch.rand(1, 1, *img_shape, device=device, requires_grad=True)
    
    # Store reconstruction process
    reconstruction_process = []
    
    for sigma in reversed(noise_schedule.sigmas):
        # Step size depends on noise level
        step_size = 1/steps_per_noise * sigma**2 #* 0.01  # Adjust this value as needed
        
        for step in range(steps_per_noise):
            
            # Data fidelity term
            h_flat = current_h.view(-1) # Flatten the current estimate from ([1, 1, 128, 160]) to ([20480])
            # show_resized_dct_image(current_h[0, 0].detach().cpu().numpy(), f"h_flat Step {step}", wait=True)

            residual = y - P @ h_flat

            grad_likelihood = residual @ P
            
            # Prior term
            with torch.no_grad():
                score = model(current_h, sigma * torch.ones(1, device=device))
            
            # Langevin update
            noise_term = torch.sqrt(2 * step_size) * torch.randn_like(current_h)
            # show_resized_dct_image(noise_term[0, 0].detach().cpu().numpy(), f"noise_term Step {step}", wait=True)
            # # print("noise_term.shape:", noise_term.shape)
            
            current_h = current_h + step_size * (score + 0.1*grad_likelihood.view_as(current_h)) + noise_term # is mean of grad_likelihood a logical choice? TODO: check
            # show_resized_dct_image(current_h[0, 0].detach().cpu().numpy(), f"current_h Step {step}", wait=True)

        reconstruction_process.append(current_h.detach().cpu().numpy())
        #show_resized_dct_image(current_h[0, 0].detach().cpu().numpy(), f"Reconstruction Step {step}", wait=True)
        print("current_h.shape:", current_h.shape)
    
    # Final reconstruction
    reconstructed_dct = current_h.squeeze().detach().cpu().numpy()
    cv2.imshow('Reconstructed DCT', reconstructed_dct)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    reconstructed_img = cv2.idct(reconstructed_dct)
    return reconstructed_img, reconstruction_process

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

def reconstruct_and_evaluate(y, P, model, hr_img, num_epochs, num_scales, sigma_min, sigma_max, anneal_power):
    """Evaluate reconstruction method"""
    results = {}
    # NCSN method
    start = time.time()
    ncsn_recon, process = annealed_langevin_dynamics(y, P, model, hr_img.shape, num_epochs=num_epochs, 
                                num_scales=num_scales, sigma_min=sigma_min, sigma_max=sigma_max, anneal_power=anneal_power)
    results['ncsn'] = {
        'time': time.time() - start,
        'nmse': calculate_nmse(hr_img.numpy(), ncsn_recon),
        'image': ncsn_recon,
        'process': process
    }
    
    return results

# =====================
# 5. Visualization
# =====================
def plot_results(results, hr_img):
    plt.figure(figsize=(18, 6))

    # NCSN
    plt.subplot(1, 3, 2)
    plt.imshow(results['ncsn']['image'], cmap='gray')
    plt.title(f"NCSN Method\nNMSE: {results['ncsn']['nmse']:.4f}")
    
    # Ground Truth
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.idct(hr_img.numpy()), cmap='gray')
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

    #resized_up_DCT = np.clip(resized_up_DCT, 0, 1)

    plt.subplot(1, 2, 1)
    plt.imshow(resized_up, cmap='gray', norm=None)
    plt.title(title)
    plt.subplot(1, 2, 2)
    plt.imshow(resized_up_DCT, cmap='gray', norm=None)
    plt.title(title + ' DCT')
    plt.savefig('resized_image_and_dct.jpg', dpi=300)
    if wait:
        plt.show()

def train_score_model():
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn, img_height=hr_shape[0], img_width=hr_shape[1]))
    score_model = score_model.to(device)
    optimizer = optim.Adam(score_model.parameters(), lr=learning_rate)

    for epoch in range(0, num_epochs):
        avg_loss = 0.
        num_items = 0
        start_epoch_time = time.time()
        batch_idx = 0
        for _, hr_img in data_loader:
            hr_img = hr_img.to(device)
            hr_img = hr_img.view(-1, 1, hr_img.shape[1], hr_img.shape[2]) # Reshape to (batch_size, channels, height, width)
            
            loss = loss_fn(score_model, hr_img, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * hr_img.shape[0]
            num_items += hr_img.shape[0]
            batch_idx += 1
            if batch_idx % 10 == 0:
                print('Epoch:', epoch, '| Batch:', batch_idx, '| Loss: {:5f}'.format(loss.item()))\
        
        print('Epoch:', epoch, '/', num_epochs, '| Average Loss: {:5f}'.format(avg_loss / num_items), '| Time:', time.time() - start_epoch_time)
        
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), '/checkpoints/ckpt.pth')
    return score_model
from scipy import integrate

# TODO: Clean up the code and remove unnecessary comments
## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5 #@param {'type': 'number'}
def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64,
                atol=error_tolerance,
                rtol=error_tolerance,
                device='cuda',
                z=None,
                eps=1e-3):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.ones(batch_size, device=device)
  # Create the latent code
  if z is None:
    print('hr_img.shape:', hr_img.shape)
    init_x = torch.randn(batch_size, 1, hr_img.shape[0], hr_img.shape[1], device=device) \
      * marginal_prob_std(t)[:, None, None, None]
  else:
    init_x = z

  shape = init_x.shape
  # print('init_x.shape:', init_x.shape)
  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    # print('sample.shape:', sample.shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape(batch_size)# .reshape(sample.shape[0], sample.shape[1])
    # print('time_steps.shape:', time_steps.shape)
    with torch.no_grad():
      score = score_model(sample, time_steps) # HERE
    # print('score.shape:', score.shape)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)

  def ode_func(t, x):
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    # print("score_eval_wrapper(x, time_steps).shape:", score_eval_wrapper(x, time_steps).shape)
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

  # Run the black-box ODE solver.
  # print("init_x.flatten().cpu().numpy().shape:", init_x.flatten().cpu().numpy().shape)
  # print("init_x.reshape(-1).cpu().numpy().shape:", init_x.reshape(-1).cpu().numpy().shape)
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.flatten().cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x

from torchvision.utils import make_grid


# =====================
# Main Workflow
# =====================
if __name__ == "__main__":

    # Load data
    dataset = ImageDataset(
        low_res_dir='images/low_res_train',
        high_res_dir='images/high_res_train'
    )

    # Get image dimensions
    data_loader = DataLoader(dataset, batch_size=batch_size) #, num_workers=2)

    # Get first image from dataset
    lr_img, hr_img = dataset[0]
    print('lr_img.shape:', lr_img.shape)
    print('hr_img.shape:', hr_img.shape)

    # Dimensions
    hr_shape = hr_img.shape     # High resolution shape (x)
    lr_shape = lr_img.shape     # Low resolution shape (y)
    n = hr_img.shape[0] * hr_img.shape[1]
    m = lr_img.shape[0] * lr_img.shape[1]
    print("N: ", n, "M: ", m)

    # Train ncsn model
    if task == 'train':
        print("Training score model...")
        if USE_NCSN_MODEL:
            score_model = train_ncsn(dataset, batch_size=batch_size, lr=learning_rate, num_epochs=num_epochs, 
                                    num_scales=num_scales, sigma_min=sigma_min, sigma_max=sigma_max, anneal_power=anneal_power)
            torch.save(score_model.state_dict(), 'ncsn_model.pth')
        else:
            score_model = train_score_model()
            torch.save(score_model.state_dict(), 'score_model.pth')

    else: # Load pre-trained model
        print("Loading pre-trained score model...")
        if USE_NCSN_MODEL:
            score_model = NCSN(dataset.img_hr_height, dataset.img_hr_width).to(device)
        else:
            score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn, img_height=hr_shape[0], img_width=hr_shape[1]))
            score_model = score_model.to(device)
        score_model.load_state_dict(torch.load('score_model.pth'))
        score_model.eval()
    
    if USE_NCSN_MODEL:
        # Generate measurements
        test_generator = CSImageGenerator(dataset, M=num_measurements, snr_db=snr_db)
        y, P = test_generator.make_measurements(lr_img, hr_img) #lr_dct
        print("y.shape:", y.shape)
        print("P.shape:", P.shape)

        # Evaluate both methods
        results = reconstruct_and_evaluate(y, P, score_model, hr_img, num_epochs=num_epochs, 
                                    num_scales=num_scales, sigma_min=sigma_min, sigma_max=sigma_max, anneal_power=anneal_power)

        # Print metrics
        print("\nEvaluation Results:")
        print(f"NCSN Method - NMSE: {results['ncsn']['nmse']:.4f}, Time: {results['ncsn']['time']:.2f}s")
        
        # Visualize
        plot_results(results, hr_img)
    else:
        # ## Load the pre-trained checkpoint from disk.
        # device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
        # ckpt = torch.load('/checkpoints/ckpt.pth', map_location=device)
        # score_model.load_state_dict(ckpt)

        sample_batch_size = 64 #@param {'type':'integer'}
        sampler = ode_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

        ## Generate samples using the specified sampler.
        samples = sampler(score_model,
                        marginal_prob_std_fn,
                        diffusion_coeff_fn,
                        sample_batch_size,
                        device=device)

        ## Sample visualization.
        #samples = samples.clamp(0.0, 1.0)
        import matplotlib.pyplot as plt
        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

        plt.figure(figsize=(6,6))
        plt.axis('off')
        plt.imshow(sample_grid.permute(1, 2, 0).cpu()) #, vmin=0., vmax=1.)
        plt.show()
        plt.savefig("batch generated")