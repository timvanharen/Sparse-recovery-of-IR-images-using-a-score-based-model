import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
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
train_batch_size = 32
num_epochs = 50
learning_rate = 1e-4
num_scales = 100
sigma_min = 0.01
sigma_max = 1.0
steps_per_noise_lvl = 5 # Number of steps per noise level
recon_batch_size = 1
anneal_power = 1. # Annealing power for Langevin dynamics

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print("Configuration:")
for k, v in config.items():
    print(f"{k}: {v}")

# Create checkpoint directory if it doesn't exist
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

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

def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

    Returns:
        The standard deviation.
    """
    t = t.to(device)
    return (torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))) #/np.sqrt(sigma) - 1./2.) * np.sqrt(sigma) # TODO: Check this formula, it rescales, moves and then rescales the distribution

def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.
    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.
    Returns:
        The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t).to(device)

sigma =  25.0 #@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

# This fuction generated 10000 times the marginal prob standard and then plots the distribution
def plot_marginal_prob_std():
    t = torch.linspace(0, 1, 10000)
    std = marginal_prob_std(t, sigma).cpu().numpy()
    #std = std/np.sqrt(sigma) - np.ones(std.shape)/2
    plt.hist(std, bins=100, density=True)
    plt.title("Marginal Probability Standard Deviation, sigma={}".format(sigma))
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
    noise_schedule = NoiseSchedule(num_scales=num_scales, sigma_min=sigma_min, sigma_max=sigma_max)

    #plot_marginal_prob_std()
    # Generate a sigma from the noise schedule from a number from 0 to num_scales-1
    random_t = torch.rand(x.shape[0]).to(device) * (1. - eps) + eps
    z = torch.randn_like(x, )
    random_t_schedule = torch.floor(random_t * num_scales).long()
    
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    #std = torch.zeros(x.shape[0]).to(device)
    #std = noise_schedule.sigmas[random_t_schedule].view(-1, 1, 1, 1) # Reshape to match x's shape
    #perturbed_x = x + z * std#[:, None, None, None] # Broadcasting to match x's shape
    
    if show_image:

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

    loss = torch.mean(torch.sum((score*std[:, None, None, None] + z)**2, dim=(1,2,3)))
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
# Annealed Langevin Dynamics
# =====================
def annealed_langevin_dynamics(y, D_height, D_width, x, model):
    noise_schedule = NoiseSchedule(num_scales=num_scales, sigma_min=sigma_min, sigma_max=sigma_max)
    
    current_h = torch.rand(recon_batch_size, 1, x.shape[0], x.shape[1], device=device, requires_grad=True)
    
    # Store reconstruction process
    reconstruction_process = []

    # Annealing hyper parameter for Langevin dynamics
    alpha = 0.1
    beta = 0.2
    r = 0.9

    # COnvert to tesnor and to device
    alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
    beta = torch.tensor(beta, dtype=torch.float32, device=device)
    r = torch.tensor(r, dtype=torch.float32, device=device)

    i = 0 #step counter
    for sigma in reversed(noise_schedule.sigmas):
        # Step size depends on noise level
        step_size = alpha * r**i
        
        for step in range(steps_per_noise_lvl):

            # Compute the residual of the estimated image and the measurements
            residual = y.flatten() - (D_height @ current_h[0, 0] @ D_width.T).view(-1) # Flatten the current estimate from ([1, 1, 128, 160]) to ([20480])

            # Compute the gradient of the likelihood term
            grad_likelihood = D_height.T @ residual.view(14, 14) @ D_width
            
            # Prior term
            with torch.no_grad():
                score = model(current_h, sigma * torch.ones(1, device=device)) # TODO: we need t
            
            # Noise term
            noise_term = torch.sqrt(2*beta * step_size) * sigma * torch.randn_like(current_h)
            
            # Update the current estimate using annealed Langevin dynamics
            current_h = current_h + step_size * (score + grad_likelihood.view_as(current_h)) / (anneal_power**2 + sigma**2) + noise_term # is mean of grad_likelihood a logical choice? TODO: check
        i += 1
        reconstruction_process.append(current_h.detach().cpu().numpy())
        show_reconstructed_image(x.detach().cpu().numpy(), current_h[0, 0].detach().cpu().numpy(), f"Reconstruction Step {step}", wait=True)
    
    # Final reconstruction
    reconstructed_img = current_h.squeeze().detach().cpu().numpy()
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

def reconstruct_and_evaluate(y, D_heigth, D_width, x, model):
    """Evaluate reconstruction method"""
    results = {}
    start = time.time()
    recon, process = annealed_langevin_dynamics(y, D_heigth, D_width, x, model)
    print("recon.shape:", recon.shape)
    print("y.shape:", y.shape)
    results['score_model'] = {
        'time': time.time() - start,
        'nmse': calculate_nmse(x.detach().cpu().numpy(), recon),
        'image': recon,
        'process': process
    }
    return results

# =====================
# 5. Visualization
# =====================
def plot_results(results, x):
    plt.figure(figsize=(18, 6))

    # Ground Truth
    plt.subplot(1, 2, 1)
    plt.imshow(x, cmap='gray')
    plt.title("Ground Truth")

    # score model
    plt.subplot(1, 2, 2)
    plt.imshow(results['score_model']['image'], cmap='gray')
    plt.title(f"NCSN Method\nNMSE: {results['score_model']['nmse']:.4f}")
    
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
        plt.imshow(img[0,0], cmap='gray')
        plt.title(f"Step {i}")
        plt.axis('off')

    plt.suptitle("Reconstruction Process")
    plt.tight_layout()
    plt.savefig('reconstruction_process.jpg', dpi=300)
    plt.show()

def show_reconstructed_image(x, y_recon, title, wait=False):
    # let's upscale the image using new  width and height
    up_width = 600
    up_height = 600
    up_points = (up_width, up_height)
    resized_up_x = cv2.resize(x, up_points, interpolation= cv2.INTER_LINEAR)
    resized_up_recon = cv2.resize(y_recon, up_points, interpolation= cv2.INTER_LINEAR)
    plt.subplot(1, 2, 1)
    plt.imshow(resized_up_x, cmap='gray', norm=None)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(resized_up_recon, cmap='gray', norm=None)
    plt.title("reconstruction in progress")
    plt.savefig('reconstruction.jpg', dpi=300)
    if wait:
        plt.show()

def train_score_model():
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)
    optimizer = Adam(score_model.parameters(), lr=learning_rate)
    
    num_of_batches = len(data_loader)
    print("Number of batches:", num_of_batches)
    for epoch in range(0, num_epochs):
        avg_loss = 0.
        num_items = 0
        start_epoch_time = time.time()

        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            #x = x.view(-1, 1, x.shape[1], x.shape[2]) # Reshape to (batch_size, channels, height, width)
            
            loss = loss_fn(score_model, x, marginal_prob_std_fn, show_image=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            batch_idx += 1

            if batch_idx % (num_of_batches // 10) == 0:
                print('Epoch:', epoch, '| Batch:', batch_idx, '| Loss: {:5f}'.format(loss.item()))
        print('Epoch:', epoch, '/', num_epochs, '| Average Loss: {:5f}'.format(avg_loss / num_items), '| Time:', time.time() - start_epoch_time)
        
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), 'checkpoints/score_model.pth')
    return score_model
from scipy import integrate

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
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape(batch_size)# .reshape(sample.shape[0], sample.shape[1])

    with torch.no_grad():
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)

  def ode_func(t, x):
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.flatten().cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x

from scipy.sparse import diags

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

# =====================
# Main Workflow
# =====================
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
if __name__ == "__main__":
    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)

    # Get image dimensions
    data_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    # Get the first batch of images
    x, y = next(iter(data_loader))
    print("Desired signal shape:", x.shape)
    print("Measurement shape:", y.shape)

    # Train the model
    if task == 'train':
        print("Training score model...")
        score_model = train_score_model()
        torch.save(score_model.state_dict(), 'checkpoints/score_model.pth')
        print("Model saved to checkpoints/score_model.pth")
        print("Training completed.")
        exit()

    if task == 'reconstruct': # Load pre-trained model
        print("Loading pre-trained score model...")

        score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
        score_model = score_model.to(device)
        score_model.load_state_dict(torch.load('checkpoints/score_model.pth'))
        #score_model.eval()

        # Generate measurements by downscaling the image in x to y, dummy TODO: Use compressed measuremnts
        y_meas = cv2.resize(x[0, 0].cpu().numpy(), (x.shape[2]//2, x.shape[3]//2), interpolation=cv2.INTER_LINEAR)
        y_meas = torch.tensor(y_meas, dtype=torch.float32).to(device)  # Convert to tensor and move to device
        y_meas = y_meas.flatten().to(device)  # Flatten and move to device

        # Create downsample matrices for the images
        # For 28×28 → 14×14
        D_height = create_downsample_matrix(28, 2)  # 32×128
        D_width = create_downsample_matrix(28, 2)   # 40×160

        y_down = D_height @ x[0, 0].cpu().numpy() @ D_width.T
        y_down = torch.tensor(y_down, dtype=torch.float32).to(device)  # Convert to tensor and move to device
        D_height = torch.tensor(D_height, dtype=torch.float32).to(device)  # Convert to tensor and move to device
        D_width = torch.tensor(D_width, dtype=torch.float32).to(device)  # Convert to tensor and move to device

        # Reconstruct and evaluate compressed measurement
        results = reconstruct_and_evaluate(y_down, D_height, D_width, x[0,0], score_model)

        # Print metrics
        print("\nEvaluation Results:")
        print(f"NCSN + Annealed Langevin Dynamics Method - NMSE: {results['score_model']['nmse']:.4f}, Time: {results['score_model']['time']:.2f}s")
        # Visualize
        plot_results(results, x[0,0].cpu().numpy())
        print("Reconstruction completed.")
        exit()
    
    if task == "generate": # Just generate from noise
        score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
        score_model = score_model.to(device)
        score_model.load_state_dict(torch.load('checkpoints/score_model.pth'))
        score_model.eval()

        start_inference_time = time.time()

        sample_batch_size = 64 #@param {'type':'integer'}
        sampler = ode_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

        ## Generate samples using the specified sampler.
        samples = sampler(score_model,
                        marginal_prob_std_fn,
                        diffusion_coeff_fn,
                        sample_batch_size,
                        device=device)
        
        print("end inference time:", time.time() - start_inference_time)
        ## Sample visualization.
        samples = samples.clamp(0.0, 1.0)
        import matplotlib.pyplot as plt
        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

        plt.figure(figsize=(6,6))
        plt.axis('off')
        plt.imshow(sample_grid.permute(1, 2, 0).cpu()) #, vmin=0., vmax=1.)
        plt.savefig("batch generated.jpg")
        plt.show()
            
        print("Sample generation completed.")
        exit()