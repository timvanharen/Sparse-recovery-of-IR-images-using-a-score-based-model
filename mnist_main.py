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

# Global configuration variables
task = 'train' #'train' # 'test  # Task name
batch_size = 256
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

# Create checkpoint directory if it doesn't exist
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

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
# Data Generator
# =====================
class CSImageGenerator:
    def __init__(self, dataset, M, snr_db):
        self.dataset = dataset
        self.M = M          # Measurements
        self.snr_db = snr_db
        
    def make_measurements(self, x, y):
        print("x.shape:", x.shape)
        N = x.shape[0]*x.shape[1]  # Number of pixels in the image

        print(f"Generating measurements for N={N} and M={self.M}")
        # Ensure M < N
        if self.M >= N:
            raise ValueError("Number of measurements M must be less than the number of pixels N.")
        
        # Print shapes
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")

        # Calculate the sensing matrix
        # First move the x to a tensor and device
        x = torch.tensor(x, dtype=torch.float32).to(device)  # Convert to tensor and move to device
        y = x.flatten()[:, torch.newaxis].to(device)  # Flatten and move to device
        h = y.flatten()[:, torch.newaxis].to(device)  # Flatten and move to device
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

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
    
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
    if PRINT_SIZE:
        print("t.shape:", t.shape)
        print("x.shape:", x.shape)  
        print("embed.shape:", embed.shape)
        print("h1.shape:", h1.shape)
        print("h2.shape:", h2.shape)
        print("h3.shape:", h3.shape)
        print("h4.shape:", h4.shape)
        
    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]
    return h
  
class ScoreNet_poepje(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[16, 32, 64, 128, 256], embed_dim=256):
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

        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1, bias=False)
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
        self.tconv3 = nn.ConvTranspose2d(channels[2]*2, channels[1], kernel_size=3, stride=2, padding=1, bias=False)

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
        if PRINT_SIZE:
            print("t.shape:", t.shape)
            print("x.shape:", x.shape)  
            print("embed.shape:", embed.shape)
            print("h1.shape:", h1.shape)
            print("h2.shape:", h2.shape)
            print("h3.shape:", h3.shape)
            print("h4.shape:", h4.shape)
        h5 = self.conv5(h4)
        
        h5 += self.dense5(embed)
        h5 = self.gnorm5(h5)
        h5 = self.act(h5)

        if PRINT_SIZE:
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
    return (torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma)))#/np.sqrt(sigma) - 1./2.) * np.sqrt(sigma) # TODO: Check this formula

def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

    Returns:
        The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)

sigma_noise =  25 #@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma_noise)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma_noise)

# This fuction generated 10000 times the marginal prob standard and then plots the distribution
def plot_marginal_prob_std():
    t = torch.linspace(0, 1, 10000)
    std = marginal_prob_std(t, sigma_noise).cpu().numpy()
    #std = std/np.sqrt(sigma_noise) - np.ones(std.shape)/2
    plt.hist(std, bins=100, density=True)
    plt.title("Marginal Probability Standard Deviation, sigma_noise={}".format(sigma_noise))
    plt.xlabel("Standard Deviation")
    plt.ylabel("Density")
    plt.show()

def loss_fn(model, x, marginal_prob_std, eps=1e-8, show_image=False):
    """The loss function for training score-based generative models.

    Args:
        model: A PyTorch model instance that represents a
            time-dependent score-based model.
        x: A mini-batch of training data.
        marginal_prob_std: A function that gives the standard deviation of
            the perturbation kernel.
        eps: A tolerance value for numerical stability.
    """
    #plot_marginal_prob_std()
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x, )
    std = marginal_prob_std(random_t)
    
    z = z * std[:, None, None, None]  # Broadcasting to match x's shape
    perturbed_x = x + z
    
    if show_image:
        # print min max of image values
        print("x.min()", x[0,0].min(), "x.max()", x[0,0].max())
        # Also print min max of noise
        print("z.min()", z[0,0].min(), "z.max()", z[0,0].max())
        print("random_t.shape:", random_t)
        plt.imshow(x[0, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Clean Image before pertubation")
        plt.axis('off')
        plt.show()
        # Show the first pertubed image in the set
        plt.imshow(perturbed_x[0, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Perturbed Image with noise {std[0]:.2f}")
        plt.axis('off')
        plt.show()

    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    return loss

# =====================
# Noise Schedule
# =====================
class NoiseSchedule:
    def __init__(self, num_scales=10, sigma_min=0.01, sigma_max=1.0):
        self.sigmas = torch.exp(torch.linspace(np.log(sigma_min), 
                              np.log(sigma_max), num_scales)).to(device)
        
    def sample_sigma(self, batch_size):
        idx = torch.randint(0, len(self.sigmas), (batch_size,))
        return self.sigmas[idx]

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

def reconstruct_and_evaluate(y, P, model, x, num_epochs, num_scales, sigma_min, sigma_max, anneal_power):
    """Evaluate reconstruction method"""
    results = {}
    start = time.time()
    ncsn_recon, process = annealed_langevin_dynamics(y, P, model, x.shape, num_epochs=num_epochs, 
                                num_scales=num_scales, sigma_min=sigma_min, sigma_max=sigma_max, anneal_power=anneal_power)
    results['score_model'] = {
        'time': time.time() - start,
        'nmse': calculate_nmse(y.numpy(), ncsn_recon),
        'image': ncsn_recon,
        'process': process
    }
    return results

# =====================
# 5. Visualization
# =====================
def plot_results(results, y):
    plt.figure(figsize=(18, 6))

    # score model
    plt.subplot(1, 2, 1)
    plt.imshow(results['score_model']['image'], cmap='gray')
    plt.title(f"NCSN Method\nNMSE: {results['score_model']['nmse']:.4f}")
    
    # Ground Truth
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.idct(y.numpy()), cmap='gray')
    plt.title("Ground Truth")
    
    plt.tight_layout()
    plt.savefig('comparison_results.jpg', dpi=300)
    plt.show()
    
    # Plot reconstruction process
    if 'process' in results['score_model']:
        plot_reconstruction_process(results['score_model']['process'])

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
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)
    optimizer = optim.Adam(score_model.parameters(), lr=learning_rate)

    for epoch in range(0, num_epochs):
        avg_loss = 0.
        num_items = 0
        start_epoch_time = time.time()
        batch_idx = 0
        for x, y in data_loader:
            x = x.to(device)
            #x = x.view(-1, 1, x.shape[1], x.shape[2]) # Reshape to (batch_size, channels, height, width)
            
            loss = loss_fn(score_model, x, marginal_prob_std_fn, show_image=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            batch_idx += 1
            #print("batch_idx:", batch_idx)
            #break
            if batch_idx % 10 == 0:
                print('Epoch:', epoch, '| Batch:', batch_idx, '| Loss: {:5f}'.format(loss.item()))\
        
        print('Epoch:', epoch, '/', num_epochs, '| Average Loss: {:5f}'.format(avg_loss / num_items), '| Time:', time.time() - start_epoch_time)
        
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), 'checkpoints/ckpt.pth')
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
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
      * marginal_prob_std(t)[:, None, None, None]
  else:
    init_x = z
    
  shape = init_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
    with torch.no_grad():    
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)
  
  def ode_func(t, x):        
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t    
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
  
  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x

from torchvision.utils import make_grid


from torchvision.datasets import MNIST
import torchvision.transforms as transforms

# =====================
# Main Workflow
# =====================
if __name__ == "__main__":
    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)

    # Get image dimensions
    data_loader = DataLoader(dataset, batch_size=batch_size) #, num_workers=2)
    # Get the first batch of images
    x, y = next(iter(data_loader))
    print("x.shape:", x.shape)
    print("y:", y)

    # Train ncsn model
    if task == 'train':
        print("Training score model...")
        score_model = train_score_model()
        torch.save(score_model.state_dict(), 'checkpoints/score_model.pth')

    elif task == 'reconstruct': # Load pre-trained model
        print("Loading pre-trained score model...")

        score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
        score_model = score_model.to(device)
        score_model.load_state_dict(torch.load('checkpoints/score_model.pth'))
        score_model.eval()

        # Generate measurements
        test_generator = CSImageGenerator(dataset, M=num_measurements, snr_db=snr_db)
        y, P = test_generator.make_measurements(x, y) #lr_dct
        print("y.shape:", y.shape)
        print("P.shape:", P.shape)

        # Evaluate both methods
        results = reconstruct_and_evaluate(y, P, score_model, x, num_epochs=num_epochs, 
                                    num_scales=num_scales, sigma_min=sigma_min, sigma_max=sigma_max, anneal_power=anneal_power)

        # Print metrics
        print("\nEvaluation Results:")
        print(f"NCSN Method - NMSE: {results['score_model']['nmse']:.4f}, Time: {results['score_model']['time']:.2f}s")
        # Visualize
        plot_results(results, y)
    
    else: # Just generate from noise
        score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
        score_model = score_model.to(device)
        score_model.load_state_dict(torch.load('checkpoints/score_model.pth'))
        score_model.eval()

        start_inference_time = time.time()

        sample_batch_size = 4 #@param {'type':'integer'}
        sampler = ode_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

        ## Generate samples using the specified sampler.
        samples = sampler(score_model,
                        marginal_prob_std_fn,
                        diffusion_coeff_fn,
                        sample_batch_size,
                        device=device)
        
        print("end inference time:", time.time() - start_inference_time)
        ## Sample visualization.
        #samples = samples.clamp(0.0, 1.0)
        import matplotlib.pyplot as plt
        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

        plt.figure(figsize=(6,6))
        plt.axis('off')
        plt.imshow(sample_grid.permute(1, 2, 0).cpu()) #, vmin=0., vmax=1.)
        plt.savefig("batch generated.jpg")
        plt.show()
            