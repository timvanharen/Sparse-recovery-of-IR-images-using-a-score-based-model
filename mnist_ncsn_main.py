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

from torchvision.utils import make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global configuration variables
task = 'train_ncsn' #'train' / 'reconstruct' / 'generate'
model = 'ScoreNet' #@param ['ScoreNet', 'ncsn'] {'type': 'raw'}
train_batch_size = 512
num_epochs = 400
learning_rate = 1e-3
weight_decay = 1e-1
stopping_criterion = 0.1 # Stop training if the loss is less than this value
num_scales = 20
sigma_min = 0.01
sigma_max = 1.0
steps_per_noise_lvl = 3 # Number of steps per noise level
recon_batch_size = 1
anneal_power = 1. # Annealing power for Langevin dynamics
eps = 1e-7 # Small value for numerical stability

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print("Configuration:")
for k, v in config.items():
    print(f"{k}: {v}")

# Create checkpoint directory if it doesn't exist
if not os.path.exists('checkpoints/MNIST'):
    os.makedirs('checkpoints/MNIST')

# =======================
# Code from MIMO Paper to train score network:
#   M. Arvinte, J. I. Tamir, "MIMO channel estimation using score-based generative models," 
#   IEEE Transactions on Wireless Communications, vol. 22, no. 6, pp.3698-3713, June 2023. 
# =======================
from ncsnv2.models        import get_sigmas
from ncsnv2.models.ema    import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from ncsnv2.losses        import get_optimizer
from ncsnv2.losses.dsm    import anneal_dsm_score_estimation

from loaders          import Channels
from dotmap           import DotMap #dotmap==1.3.30
import os, copy, argparse
from tqdm import tqdm as tqdm

import sys
#sys.path.append('./')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train', type=str, default='CDL-C')
args = parser.parse_args()

# Disable TF32 due to potential precision issues
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Model config
config          = DotMap()
config.device   = 'cuda:0'
# Inner model
config.model.ema           = True
config.model.ema_rate      = 0.999
config.model.normalization = 'InstanceNorm++'
config.model.nonlinearity  = 'elu'
config.model.sigma_dist    = 'geometric'
config.model.num_classes   = 2311 # Number of train sigmas and 'N'
config.model.ngf           = 32

# Optimizer
config.optim.weight_decay  = 0.000 # No weight decay
config.optim.optimizer     = 'Adam'
config.optim.lr            = 0.0001
config.optim.beta1         = 0.9
config.optim.amsgrad       = False
config.optim.eps           = 0.001

# Training
config.training.batch_size     = 32
config.training.num_workers    = 0
config.training.n_epochs       = 400
config.training.anneal_power   = 2
config.training.log_all_sigmas = False

# Data
# FOr MNIST
# config.data.channel        = args.train
# config.data.channels       = 1
# config.data.noise_std      = 0
# config.data.image_size     = [28, 28] # [Nt, Nr] for the transposed channel
# config.data.num_pilots     = config.data.image_size[1]
# config.data.norm_channels  = 'global'
# config.data.spacing_list   = [0.5] # Training and validation

# For MIMO
config.data.channel        = args.train
config.data.channels       = 1
config.data.noise_std      = 0
config.data.image_size     = [28, 18] # [Nt, Nr] for the transposed channel
config.data.num_pilots     = config.data.image_size[1]
config.data.norm_channels  = 'global'
config.data.spacing_list   = [0.5] # Training and validation

# Seeds for train and test datasets
train_seed, val_seed = 4321, 4321

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

        return torch.tensor(hr_img), torch.tensor(lr_img) # Train with dense representation / image and use low resoultuon as compressed measurement
    
# Get datasets and loaders for channels
dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
# dataset     = Channels(train_seed, config, norm=config.data.norm_channels)

# Load data
dataset = ImageDataset(
    low_res_dir='images-square/low_res_train',
    high_res_dir='images-square/medium_res_train'
)

# Get image dimensions
data_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)

data_loader  = DataLoader(dataset, batch_size=config.training.batch_size, 
         shuffle=True, num_workers=config.training.num_workers, drop_last=True)

# # Validation data
# val_datasets, val_loaders, val_iters = [], [], []
# for idx in range(len(config.data.spacing_list)):
#     # Validation config
#     val_config = copy.deepcopy(config)
#     val_config.data.spacing_list = [config.data.spacing_list[idx]]
#     # Create locals
#     val_datasets.append(Channels(val_seed, val_config, norm=[dataset.mean, dataset.std]))
#     val_loaders.append(DataLoader(
#         val_datasets[-1], batch_size=len(val_datasets[-1]),
#         shuffle=False, num_workers=0, drop_last=True))
#     val_iters.append(iter(val_loaders[-1])) # For validation

# Construct pairwise distances
if False: # Set to true to follow [Song '20] exactly
    dist_matrix   = np.zeros((len(dataset), len(dataset)))
    flat_channels = dataset.channels.reshape((len(dataset), -1))
    for idx in tqdm(range(len(dataset))):
        dist_matrix[idx] = np.linalg.norm(
            flat_channels[idx][None, :] - flat_channels, axis=-1)
# Pre-determined values from 'Mixed' setting
config.model.sigma_begin = 39.15
config.model.sigma_rate  = 0.995
config.model.sigma_end   = config.model.sigma_begin * \
    config.model.sigma_rate ** (config.model.num_classes - 1)

# Choose the inference step size (epsilon) according to [Song '20]
candidate_steps = np.logspace(-13, -8, 1000)
step_criterion  = np.zeros((len(candidate_steps)))
gamma_rate      = 1 / config.model.sigma_rate
for idx, step in enumerate(candidate_steps):
    step_criterion[idx] = (1 - step / config.model.sigma_end ** 2) \
        ** (2 * config.model.num_classes) * (gamma_rate ** 2 -
            2 * step / (config.model.sigma_end ** 2 - config.model.sigma_end ** 2 * (
                1 - step / config.model.sigma_end ** 2) ** 2)) + \
            2 * step / (config.model.sigma_end ** 2 - config.model.sigma_end ** 2 * (
                1 - step / config.model.sigma_end ** 2) ** 2)
best_idx = np.argmin(np.abs(step_criterion - 1.))
config.model.step_size = candidate_steps[best_idx]

# Instantiate model
diffuser = NCSNv2Deepest(config)
diffuser = diffuser.cuda()

# Instantiate optimizer
optimizer = get_optimizer(config, diffuser.parameters())

# Instantiate counters and EMA helper
start_epoch, step = 0, 0
if config.model.ema:
    ema_helper = EMAHelper(mu=config.model.ema_rate)
    ema_helper.register(diffuser)

# Get all sigma values for the discretized VE-SDE
sigmas = get_sigmas(config)

# # Sample fixed validation data
# val_H_list = []
# for idx in range(len(config.data.spacing_list)):
#     val_sample = next(val_iters[idx])
#     val_H_list.append(val_sample['H_herm'].cuda())

# Logging
config.log_path = './models/score/%s' % args.train
os.makedirs(config.log_path, exist_ok=True)
train_loss, val_loss  = [], []

if task == 'train_ncsn':
    # For each epoch
    for epoch in tqdm(range(start_epoch, config.training.n_epochs)):
        # For each batch
        # for i, sample in tqdm(enumerate(data_loader)):
        #     # for a in sample:
        #     #     print(a)
        #     #print("y.shape:", y.shape)
        #     diffuser.train()
        #     step += 1
        #     # # Move data to device
        #     for key in sample:
        #         sample[key] = sample[key].cuda()

            # print("sample['H_herm'].shape:", sample['H_herm'].shape)
            # # Compute DSM loss using Hermitian channels
            # loss = anneal_dsm_score_estimation(
            #     diffuser, sample['H_herm'], sigmas, None, 
            #     config.training.anneal_power)
            
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            x = x.view(-1, 1, x.shape[1], x.shape[2]) # Reshape to (batch_size, channels, height, width)

            diffuser.train()
            step += 1

            # Compute DSM loss using Hermitian channels
            loss = anneal_dsm_score_estimation(
                diffuser, x, sigmas, None, 
                config.training.anneal_power)    
            
            # Logging
            if step == 1:
                running_loss = loss.item()
            else:
                running_loss = 0.99 * running_loss + 0.01 * loss.item()
            train_loss.append(loss.item())
            
            # Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # EMA update
            if config.model.ema:
                ema_helper.update(diffuser)
                
    #         # Verbose
    #         if step % 100 == 0:
    #             if config.model.ema:
    #                 val_score = ema_helper.ema_copy(diffuser)
    #             else:
    #                 val_score = diffuser
                
    #             # For each validation setup
    #             local_val_losses = []
    #             for idx in range(len(config.data.spacing_list)):
    #                 with torch.no_grad():
    #                     val_dsm_loss = \
    #                         anneal_dsm_score_estimation(
    #                             val_score, val_H_list[idx],
    #                             sigmas, None,
    #                             config.training.anneal_power)
    #                 # Store
    #                 local_val_losses.append(val_dsm_loss.item())
    #             # Sanity delete
    #             del val_score
    #             # Log
    #             val_loss.append(local_val_losses)
                    
    #             # Print
    #             if len(local_val_losses) == 1:
    #                 print('Epoch %d, Step %d, Train Loss (EMA) %.3f, \
    # Val. Loss %.3f' % (
    #                     epoch, step, running_loss, 
    #                     local_val_losses[0]))
    #             elif len(local_val_losses) >= 2:
    #                 print('Epoch %d, Step %d, Train Loss (EMA) %.3f, \
    # Val. Loss (Split) %.3f %.3f' % (
    #                     epoch, step, running_loss, 
    #                     local_val_losses[0], local_val_losses[1]))

    input("End of training, are you sure about the file name?")

    # Save final weights
    torch.save({'model_state': diffuser.state_dict(),
                'optim_state': optimizer.state_dict(),
                'config': config,
                'train_loss': train_loss,
                'val_loss': val_loss}, 
    os.path.join(config.log_path, 'mnist_ncsn_model.pt'))

    input("End of mimo paper training implementation. Press Enter to continue...")

task = 'reconstruct' # Set to reconstruct to run the reconstruction code

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
def annealed_langevin_dynamics(y, D_height, D_width, x, model, eps):
    noise_schedule = NoiseSchedule(num_scales=num_scales, sigma_min=sigma_min, sigma_max=sigma_max)
    
    current_h = torch.rand(recon_batch_size, 1, x.shape[0], x.shape[1], device=device, requires_grad=True)

    # Store reconstruction process
    reconstruction_process = []

    # Solving R from a end value of 0.12, r**i = 0.12
    log_end_r = np.log10(0.1) / num_scales
    r = 10**(log_end_r) # This value should be set according to the number of scales, a nice end value is r**i = 0.12
    print("r:", r)
    # Annealing hyper parameter for Langevin dynamics
    alpha = 0.9
    beta = 1- alpha

    anneal_power = 0.9 # Annealing power for Langevin dynamics, !!! greater than 1!!!

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
            grad_likelihood = D_height.T @ residual.view_as(y) @ D_width

            # Prior term

            t = torch.rand(current_h.shape[0]).to(device) * (1. - eps) + eps
            t[0] = ((num_scales-i) / num_scales) *(1.-eps) + eps
            #t = t.view(-1, 1).float() # Reshape to match the input shape of the model
            with torch.no_grad():
                score = model(current_h, t)

            # Noise term
            noise_term = torch.sqrt(2*beta * step_size) * sigma * torch.randn_like(current_h)
            
            # Update the current estimate using annealed Langevin dynamics
            current_h = current_h + step_size * (score + grad_likelihood.view_as(current_h)) / (anneal_power**2 + sigma**2) + noise_term # is mean of grad_likelihood a logical choice? TODO: check
        i += 1
        if i % (num_scales // 20) == 0: # save 20 steps
            reconstruction_process.append(current_h.detach().cpu().numpy())
        # show_reconstructed_image(x.detach().cpu().numpy(), y.detach().cpu().numpy(), current_h[0, 0].detach().cpu().numpy(), f"Reconstruction Step {step}", wait=True)
    
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

def reconstruct_and_evaluate(y, D_heigth, D_width, x, model, eps):
    """Evaluate reconstruction method"""
    results = {}
    start = time.time()
    recon, process = annealed_langevin_dynamics(y, D_heigth, D_width, x, model, eps)
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
def plot_results(results, y, x):
    plt.figure(figsize=(18, 6))

    # Ground Truth
    plt.subplot(1, 3, 1)
    plt.imshow(x, cmap='gray')
    plt.title("Ground Truth")

    # Compressed measurement Truth
    plt.subplot(1, 3, 2)
    plt.imshow(y, cmap='gray')
    plt.title("Ground Truth")

    # score model
    plt.subplot(1, 3, 3)
    plt.imshow(results['score_model']['image'], cmap='gray')
    plt.title(f"NCSN Method\nNMSE: {results['score_model']['nmse']:.4f}")
    
    plt.tight_layout()
    plt.savefig('comparison_results.jpg', dpi=300)

    # Plot reconstruction process
    if 'process' in results['score_model']:
        plot_reconstruction_process(results['score_model']['process'])

def plot_reconstruction_process(process):
    plt.figure(figsize=(12, 10))
    # Calculate the amount of rows and columns needed for the subplots
    num_images = len(process)
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols  # Ceiling division

    for i, img in enumerate(process):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(img[0,0], cmap='gray')
        plt.title(f"Step {i*num_scales//num_images}")
        plt.axis('off')

    plt.suptitle("Reconstruction Process")
    plt.tight_layout()
    plt.savefig('reconstruction_process.jpg', dpi=300)
    plt.show()

def show_reconstructed_image(x, y, y_recon, title, wait=False):
    # let's upscale the image using new  width and height
    plt.figure(figsize=(12, 6))
    up_width = 600
    up_height = 600
    up_points = (up_width, up_height)
    resized_up_x = cv2.resize(x, up_points, interpolation= cv2.INTER_LINEAR)
    resized_up_y = cv2.resize(y, up_points, interpolation= cv2.INTER_LINEAR)
    resized_up_recon = cv2.resize(y_recon, up_points, interpolation= cv2.INTER_LINEAR)
    plt.subplot(1, 3, 1)
    plt.imshow(resized_up_x, cmap='gray', norm=None)
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(resized_up_y, cmap='gray', norm=None)
    plt.title("Compressed Image")

    plt.subplot(1, 3, 3)
    plt.imshow(resized_up_recon, cmap='gray', norm=None)
    plt.title("reconstruction in progress")
    plt.savefig('reconstruction.jpg', dpi=300)
    if wait:
        plt.show()


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
# Reconstruction
# =====================

if task == 'reconstruct':
    # Load weights
    contents = torch.load(os.path.join(config.log_path, 'mnist_ncsn_model.pt'))
    diffuser.load_state_dict(contents['model_state']) 
    diffuser.eval()
    
    # Load test data
    test_dataset = MNIST('.', train=False, transform=transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=recon_batch_size, shuffle=False)

    # Get the first batch of test data
    for i, (sample, y) in enumerate(test_dataloader):
        x = sample[0].unsqueeze(1).to(device)

if __name__ == "__main__":
    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)

    # Get image dimensions
    data_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    # Get the first batch of images
    x, y = next(iter(data_loader))
    print("Desired signal shape:", x.shape)
    print("Measurement shape:", y.shape)

    # Save the first 5 images to the dir
    for i in range(5):
        img = x[i][0].cpu().numpy()
        plt.imsave(f'images/mnist_mg_{i}.png', img, cmap='gray')

    # Train the model
    if task == 'train':
        print("Training score model...")
        score_model = train_score_model()
        torch.save(score_model.state_dict(), 'checkpoints/MNIST/mnist_model.pth')
        print("Model saved to checkpoints/mnist_model.pth")
        print("Training completed.")
        exit()

    if task == 'reconstruct': # Load pre-trained model
        print("Loading pre-trained score model...")

        # score_model = 
        score_model = score_model.to(device)
        score_model.load_state_dict(torch.load('checkpoints/MNIST/mnist_model.pth'))
        score_model.eval()

        # Generate measurements by downscaling the image in x to y, dummy TODO: Use compressed measuremnts
        factor = 2
        y_meas = cv2.resize(x[0,0].cpu().numpy(), (x.shape[2]//factor, x.shape[3]//factor), interpolation=cv2.INTER_LINEAR)
        y_meas = torch.tensor(y_meas, dtype=torch.float32).to(device)  # Convert to tensor and move to device
        y_meas = y_meas.flatten().to(device)  # Flatten and move to device

        # Create downsample matrices for the images
        # For 28×28 → 14×14
        D_height = create_downsample_matrix(28, factor)  # 32×128
        D_width = create_downsample_matrix(28, factor)   # 40×160

        y_down = D_height @ x[0,0].cpu().numpy() @ D_width.T
        y_down = torch.tensor(y_down, dtype=torch.float32).to(device)  # Convert to tensor and move to device
        D_height = torch.tensor(D_height, dtype=torch.float32).to(device)  # Convert to tensor and move to device
        D_width = torch.tensor(D_width, dtype=torch.float32).to(device)  # Convert to tensor and move to device

        # Reconstruct and evaluate compressed measurement
        results = reconstruct_and_evaluate(y_down, D_height, D_width, x[0,0], score_model, eps=eps)

        # Print metrics
        print("\nEvaluation Results:")
        print(f"NCSN + Annealed Langevin Dynamics Method - NMSE: {results['score_model']['nmse']:.4f}, Time: {results['score_model']['time']:.2f}s")
        
        # Visualize
        plot_results(results, y_down.cpu().numpy(), x[0,0].cpu().numpy())
        print("Reconstruction completed.")
        exit()
    