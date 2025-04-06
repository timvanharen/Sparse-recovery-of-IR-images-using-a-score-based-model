
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

  def __init__(self, marginal_prob_std, img_height, img_width, channels=[32, 64, 128, 256], embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()

    if img_height == 32 and img_width == 40:

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
      self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=(2,4), stride=2, padding=1, bias=False)
      self.dense3 = Dense(embed_dim, channels[2])
      self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
      self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=2, padding=0, bias=False)
      self.dense4 = Dense(embed_dim, channels[3])
      self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

      # Decoding layers where the resolution increases
      self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=3, stride=2, padding=0, bias=False)
      self.dense5 = Dense(embed_dim, channels[2])
      self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
      self.tconv3 = nn.ConvTranspose2d(channels[2]*2, channels[1], kernel_size=(2,4), stride=2, padding=1, bias=False)
      self.dense6 = Dense(embed_dim, channels[1])
      self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
      self.tconv2 = nn.ConvTranspose2d(channels[1]*2, channels[0], kernel_size=4, stride=2, padding=1, bias=False)
      self.dense7 = Dense(embed_dim, channels[0])
      self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
      self.tconv1 = nn.ConvTranspose2d(channels[0]*2, 1, kernel_size=4, stride=2, padding=1)
    
    elif img_height == 128 and img_width == 160:
      # Gaussian random feature embedding layer for time
      self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
          nn.Linear(embed_dim, embed_dim))
      # Encoding layers where the resolution decreases
      self.conv1 = nn.Conv2d(1, channels[0], kernel_size=4, stride=2, padding=1, bias=False)
      self.dense1 = Dense(embed_dim, channels[0])
      self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
      self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=(2,8), stride=2, padding=1, bias=False)
      self.dense2 = Dense(embed_dim, channels[1])
      self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
      self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=(3,8), stride=2, padding=1, bias=False)
      self.dense3 = Dense(embed_dim, channels[2])
      self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
      self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=2, padding=0, bias=False)
      self.dense4 = Dense(embed_dim, channels[3])
      self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

      # Decoding layers where the resolution increases
      self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=3, stride=2, padding=0, bias=False)
      self.dense5 = Dense(embed_dim, channels[2])
      self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
      self.tconv3 = nn.ConvTranspose2d(channels[2]*2, channels[1], kernel_size=(3,8), stride=2, padding=1, bias=False)
      self.dense6 = Dense(embed_dim, channels[1])
      self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
      self.tconv2 = nn.ConvTranspose2d(channels[1]*2, channels[0], kernel_size=(2,8), stride=2, padding=1, bias=False)
      self.dense7 = Dense(embed_dim, channels[0])
      self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
      self.tconv1 = nn.ConvTranspose2d(channels[0]*2, 1, kernel_size=4, stride=2, padding=1)
    
    elif img_height == 512 and img_width == 640:
      # Gaussian random feature embedding layer for time
      self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
          nn.Linear(embed_dim, embed_dim))
      # Encoding layers where the resolution decreases
      self.conv1 = nn.Conv2d(1, channels[0], kernel_size=4, stride=2, padding=1, bias=False)
      self.dense1 = Dense(embed_dim, channels[0])
      self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
      self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=(4,12), stride=4, padding=0, bias=False)
      self.dense2 = Dense(embed_dim, channels[1])
      self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
      self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=(2,16), stride=2, padding=1, bias=False)
      self.dense3 = Dense(embed_dim, channels[2])
      self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
      self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=2, padding=0, bias=False)
      self.dense4 = Dense(embed_dim, channels[3])
      self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

      # Decoding layers where the resolution increases
      self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=3, stride=2, padding=0, bias=False)
      self.dense5 = Dense(embed_dim, channels[2])
      self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
      self.tconv3 = nn.ConvTranspose2d(channels[2]*2, channels[1], kernel_size=(2,16), stride=2, padding=1, bias=False)
      self.dense6 = Dense(embed_dim, channels[1])
      self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
      self.tconv2 = nn.ConvTranspose2d(channels[1]*2, channels[0], kernel_size=(4,12), stride=4, padding=0, bias=False)
      self.dense7 = Dense(embed_dim, channels[0])
      self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
      self.tconv1 = nn.ConvTranspose2d(channels[0]*2, 1, kernel_size=4, stride=2, padding=1)
    else:
      raise ValueError("Unsupported image size. Supported sizes are (512, 640), (128, 160) and (32x40).")

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

    h = self.tconv4(h4)
    if PRINT_SIZE: print("h.shape: self.tconv4(h4)", h.shape)
    
    h += self.dense5(embed)
    if PRINT_SIZE:  print("h.shape self.dense5(embed):", h.shape)
    
    h = self.tgnorm4(h)
    if PRINT_SIZE:  print("h.shape: self.tgnorm4(h)", h.shape)
    
    h = self.act(h)
    if PRINT_SIZE:  print("h.shape: self.act(h)", h.shape)

    h = self.tconv3(self.padcat(h, h3))  # Modified
    if PRINT_SIZE:  print("h.shape: self.tconv3(self.padcat(h, h3))  # Modified", h.shape)

    h += self.dense6(embed)
    if PRINT_SIZE:  print("h.shape: self.dense6(embed)", h.shape)

    h = self.tgnorm3(h)
    if PRINT_SIZE:  print("h.shape: self.tgnorm3(h)", h.shape)

    h = self.act(h)
    if PRINT_SIZE:  print("h.shape: self.act(h)", h.shape)
    
    h = self.tconv2(self.padcat(h, h2))  # Modified
    if PRINT_SIZE:  print("h.shape: self.tconv2(self.padcat(h, h2))", h.shape)
    
    h += self.dense7(embed)
    if PRINT_SIZE:  print("h.shape: self.dense7(embed)", h.shape)
    h = self.tgnorm2(h)
    if PRINT_SIZE:  print("h.shape: self.tgnorm2(h)", h.shape)

    h = self.act(h)
    if PRINT_SIZE:  print("h.shape: self.act(h)", h.shape)

    h = self.tconv1(self.padcat(h, h1))  # Modified
    if PRINT_SIZE:  print("h.shape: self.tconv1(self.padcat(h, h1))  # Modified", h.shape)
    
    return h / self.marginal_prob_std(t)[:, None, None, None]

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

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

import functools
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

import torch
import functools
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
import time
import glob
#from tqdm import notebook
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import numpy as np
from scipy.fftpack import dctn, idctn
from scipy.sparse.linalg import LinearOperator

if not os.path.exists('/checkpoints'):
  os.makedirs('/checkpoints')

n_epochs =   50#@param {'type':'integer'}
## size of a mini-batch
batch_size =  32 #@param {'type':'integer'}
## learning rate
lr=1e-4 #@param {'type':'number'}
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

        return torch.tensor(lr_img), torch.tensor(hr_img) # Train with dense representation / image

USE_MNIST = False #@param {type:"boolean"}
if USE_MNIST:
    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
else:
    dataset = ImageDataset(high_res_dir='images/medium_res_train/MR_train', low_res_dir='images/low_res_train/LR_train',)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)#, num_workers=2)
lr_img, hr_img = dataset[0]
print('lr_img.shape:', lr_img.shape)

if not USE_MNIST:
    print('hr_img.shape:', hr_img.shape)
    # Dimensions
    hr_shape = hr_img.shape  # High resolution shape (x)
    lr_shape = lr_img.shape    # Low resolution shape (y)
    n = hr_img.shape[0] * hr_img.shape[1]  # 20.480 (x size)
    m = lr_img.shape[0] * lr_img.shape[1]  # 1,280 (y size)

def A_operator(x, hr_shape=hr_shape, lr_shape=lr_shape): 
    """Applies A to vector x: Downsampling after inverse DCT"""
    # Reshape to 2D DCT coefficients
    x_2d = x.reshape(hr_shape)
    factor = hr_shape[0] // lr_shape[0]  # Downsampling factor (4)
    
    # Inverse 2D DCT (to get real-space image)
    image = cv2.idct(x_2d)
    
    # Downsample (using simple slicing - modify if needed)
    downsampled = image[::factor, ::factor]
    
    return downsampled.ravel()

def A_transpose_operator(y, hr_shape=hr_shape, lr_shape=lr_shape, factor=4):
    """Applies A^T to vector y: DCT after upsampling"""
    # Reshape to 2D low-res image
    y_2d = y.reshape(lr_shape)
    factor = hr_shape[0] // lr_shape[0]  # Downsampling factor (4)
    
    # Upsample (with zero insertion)
    upsampled = np.zeros(hr_shape)
    upsampled[::factor, ::factor] = y_2d
    
    # Forward 2D DCT (to get coefficients)
    coeffs = cv2.dct(upsampled)
    
    return coeffs.ravel()

def construct_A_explicit():
    """Constructs explicit A matrix (memory intensive!)"""
    A = np.zeros((m, n))
    basis_vec = np.zeros(n)
    
    for i in range(n):
        basis_vec[i] = 1
        A[:, i] = A_operator(basis_vec)
        basis_vec[i] = 0
        
    return A

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn, img_height=lr_shape[0], img_width=lr_shape[1], channels=[32, 64, 128, 256], embed_dim=256))
#score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn, img_height=hr_shape[0], img_width=hr_shape[1], channels=[32, 64, 128, 256], embed_dim=256))
score_model = score_model.to(device)
print('device', device)

CAL_A_MATRIX = False #@param {type:"boolean"}
if CAL_A_MATRIX:
    # Create LinearOperator for A (memory efficient)
    A = LinearOperator(shape=(m, n), matvec=A_operator, rmatvec=A_transpose_operator)
    print("A.shape:", A.shape)
    print("A:", A)

    # COnstructing A matrix only if the size is small for experimentation
    if hr_shape[0] < 128 and hr_shape[1] < 160:
        A_exp = construct_A_explicit()  # Uncomment to construct explicit A (not recommended for large images)
        print("A_exp.shape:", A_exp.shape)
        print("A_exp:", A_exp)
    else:
        raise ValueError("Image size too large for explicit A matrix construction.")

optimizer = Adam(score_model.parameters(), lr=lr)
#tqdm_epoch = tqdm.notebook.trange(n_epochs)
for epoch in range(0, n_epochs): #tqdm_epoch:
  avg_loss = 0.
  num_items = 0
  start_epoch_time = time.time()
  for lr_img, hr_img in data_loader:
    
    if CAL_A_MATRIX:
        # Calculate the DCT of the Hr_img
        print('hr_img.shape:', hr_img[0].shape)
        hr_dct = cv2.dct(np.float32(hr_img[0]))
        print('hr_dct.shape:', hr_dct.shape)
        print('hr_dct.ravel().shape:', hr_dct.ravel().shape)
        
        lr_img_A = A_exp@hr_dct.flatten()
        # Reshape hr_img_a to the shape with
        lr_img_A = lr_img_A.reshape(lr_img[0].shape)
        print('hr_img_A.shape:', lr_img_A.shape)
        plt.subplot(1, 2, 1)
        plt.imshow(lr_img_A, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(hr_img[0], cmap='gray')
        plt.show()

    # Using our own set.
    lr_img = lr_img.to(device)
    if not USE_MNIST:
        lr_img = lr_img.view(-1, 1, lr_img.shape[1], lr_img.shape[2]) # Reshape to (batch_size, channels, height, width)

    loss = loss_fn(score_model, lr_img, marginal_prob_std_fn)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    avg_loss += loss.item() * lr_img.shape[0]
    num_items += lr_img.shape[0]
    # print('Batch Loss: {:5f}'.format(loss.item()))
  # Print the averaged training loss so far.
  #tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
  print('Epoch:', epoch, '/', n_epochs, '| Average Loss: {:5f}'.format(avg_loss / num_items), '| Time:', time.time() - start_epoch_time)
  # Update the checkpoint after each epoch of training.
  torch.save(score_model.state_dict(), '/checkpoints/ckpt.pth')

## The number of sampling steps.
num_steps =  500#@param {'type':'integer'}
def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           num_steps=num_steps,
                           device='cuda',
                           eps=1e-3):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

  Returns:
    Samples.
  """

  t = torch.ones(batch_size, device=device)
  # print('t.shape:', t.shape)
  # print("lr_img.shape:", lr_img.shape)
  init_x = torch.randn(batch_size, 1, lr_img.shape[2], lr_img.shape[3], device=device) \
  * marginal_prob_std(t)[:, None, None, None]
  # print('init_x.shape:', init_x.shape)
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  # print('step_size:', step_size)
  x = init_x
  with torch.no_grad():
    for time_step in range(len(time_steps)): #tqdm.notebook.tqdm(time_steps):
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)
      # print("g:", g)
      # print('g.shape:', g.shape)
      # print("(g**2)[:, None, None, None]:", (g**2)[:, None, None, None].shape)
      # print('x.shape:', x.shape)
      print("batch_time_step.shape:", batch_time_step.shape)
      print('score_model(x, batch_time_step).shape:', score_model(x, batch_time_step).shape)
      mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
      # print('mean_x.shape:', mean_x.shape)
      # x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
  # Do not include any noise in the last sampling step.
  return mean_x

signal_to_noise_ratio = 0.16 #@param {'type':'number'}

## The number of sampling steps.
num_steps =  500#@param {'type':'integer'}
def pc_sampler(score_model,
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64,
               num_steps=num_steps,
               snr=signal_to_noise_ratio,
               device='cuda',
               eps=1e-3):
  """Generate samples from score-based models with Predictor-Corrector method.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient
      of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

  Returns:
    Samples.
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, lr_img.shape[2], lr_img.shape[3], device=device)\
      * marginal_prob_std(t)[:, None, None, None]
  time_steps = np.linspace(1., eps, num_steps)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in range(len(time_steps)): #tqdm.notebook.tqdm(time_steps):
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      # Corrector step (Langevin MCMC)
      grad = score_model(x, batch_time_step)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = np.sqrt(np.prod(x.shape[1:]))
      langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
      x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

      # Predictor step (Euler-Maruyama)
      g = diffusion_coeff(batch_time_step)
      x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
      x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)

    # The last step does not include any noise
    return x_mean

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
    init_x = torch.randn(batch_size, 1, lr_img.shape[2], lr_img.shape[3], device=device) \
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
