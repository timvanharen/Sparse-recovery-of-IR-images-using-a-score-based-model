"""
Score-based sparse recovery of IR images using NCSNv2.

Replaces the custom ScoreNet with Yang Song's NCSNv2 architecture for
significantly better score estimation and image reconstruction quality.

Usage:
  Train:        python ir_ncsn_main.py --task train --config configs/ir_128x128.yml
  Reconstruct:  python ir_ncsn_main.py --task reconstruct --config configs/ir_128x128.yml
  Generate:     python ir_ncsn_main.py --task generate --config configs/ir_128x128.yml

Requirements:
  - HR image directories (configured in YAML: data.hr_train_dir, data.hr_test_dir)
  - Run process_data_set.py first if these directories don't exist
  - Run process_data_set.py first if these directories don't exist
"""

import os
import sys
import argparse
import time
import copy
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import glob
from pathlib import Path

# NCSNv2 imports
from ncsnv2.models import get_sigmas
from ncsnv2.models.ema import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deeper
from ncsnv2.losses import get_optimizer
from ncsnv2.losses.dsm import anneal_dsm_score_estimation

# Local imports
from utils import create_downsample_matrix, calculate_nmse, calculate_psnr


# =====================
# Configuration
# =====================
class DotDict(dict):
    """Dictionary that supports dot notation access."""
    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(key)
        if isinstance(val, dict):
            return DotDict(val)
        return val

    def __setattr__(self, key, val):
        self[key] = val


def load_config(config_path):
    """Load YAML config and return as nested DotDict."""
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    config = DotDict(raw)
    # Ensure nested dicts are also DotDicts
    for key, val in config.items():
        if isinstance(val, dict):
            config[key] = DotDict(val)

    # Set device
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return config


# =====================
# Dataset
# =====================
class IRImageDataset(Dataset):
    """Dataset for IR images. Loads high-resolution images for score model training."""

    def __init__(self, hr_dir, image_size=128):
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.jpg')))
        if not self.hr_files:
            self.hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.png')))
        if not self.hr_files:
            raise FileNotFoundError(f"No .jpg or .png images found in {hr_dir}")

        self.image_size = image_size
        print(f"Found {len(self.hr_files)} images in {hr_dir}")

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.hr_files[idx]).convert('L'))
        # Resize to target size if needed
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            from PIL import Image as PILImage
            img = np.array(PILImage.fromarray(img).resize(
                (self.image_size, self.image_size), PILImage.LANCZOS))
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # Add channel dimension: (1, H, W)
        img = torch.tensor(img).unsqueeze(0)
        return img


class IRImagePairDataset(Dataset):
    """Dataset that loads paired HR/LR images for evaluation."""

    def __init__(self, hr_dir, lr_dir, image_size=128):
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.jpg')))
        self.lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.jpg')))
        if not self.hr_files:
            self.hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.png')))
        if not self.lr_files:
            self.lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.png')))
        self.image_size = image_size
        print(f"Found {len(self.hr_files)} HR and {len(self.lr_files)} LR images")

    def __len__(self):
        return min(len(self.hr_files), len(self.lr_files))

    def __getitem__(self, idx):
        hr_img = np.array(Image.open(self.hr_files[idx]).convert('L'))
        lr_img = np.array(Image.open(self.lr_files[idx]).convert('L'))

        if hr_img.shape[0] != self.image_size or hr_img.shape[1] != self.image_size:
            hr_img = np.array(Image.fromarray(hr_img).resize(
                (self.image_size, self.image_size), Image.LANCZOS))

        hr_img = hr_img.astype(np.float32) / 255.0
        lr_img = lr_img.astype(np.float32) / 255.0

        hr_tensor = torch.tensor(hr_img).unsqueeze(0)
        lr_tensor = torch.tensor(lr_img)
        return hr_tensor, lr_tensor


# =====================
# Training
# =====================
def train(config, checkpoint_dir='checkpoints/ir_ncsn'):
    """Train the NCSNv2 score model on HR IR images."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = config.device

    # Dataset
    dataset = IRImageDataset(
        hr_dir=config.data.hr_train_dir,
        image_size=config.data.image_size
    )
    data_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    print(f"Training on {len(dataset)} images, "
          f"{len(data_loader)} batches/epoch, "
          f"batch_size={config.training.batch_size}")

    # Model
    model = NCSNv2Deeper(config).to(device)
    model = nn.DataParallel(model)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = get_optimizer(config, model.parameters())

    # EMA
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)

    # Sigma schedule
    sigmas = get_sigmas(config)
    print(f"Sigma schedule: {sigmas[0]:.2f} -> {sigmas[-1]:.4f} "
          f"({len(sigmas)} levels)")

    # Resume from checkpoint if available
    start_epoch = 0
    step = 0
    ckpt_path = os.path.join(checkpoint_dir, 'latest.pth')
    if os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        start_epoch = ckpt['epoch'] + 1
        step = ckpt['step']
        if config.model.ema and 'ema_state' in ckpt:
            ema_helper.load_state_dict(ckpt['ema_state'])
        print(f"Resumed at epoch {start_epoch}, step {step}")

    # Training loop
    train_losses = []
    total_epochs = config.training.n_epochs
    num_batches = len(data_loader)
    training_start = time.time()
    print(f"\nStarting training from epoch {start_epoch} to {total_epochs} "
          f"({num_batches} batches/epoch)...")

    for epoch in range(start_epoch, total_epochs):
        epoch_loss = 0.0
        num_items = 0
        t_start = time.time()

        for batch_idx, x in enumerate(data_loader):
            x = x.to(device)

            # Compute denoising score matching loss
            loss = anneal_dsm_score_estimation(
                model, x, sigmas, labels=None,
                anneal_power=config.training.anneal_power
            )

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # EMA update
            if config.model.ema:
                ema_helper.update(model)

            epoch_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            step += 1

            # Per-batch progress (every 10 batches or last batch)
            if (batch_idx + 1) % max(1, num_batches // 5) == 0 or batch_idx == num_batches - 1:
                batch_elapsed = time.time() - t_start
                batch_eta = batch_elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                print(f"  Epoch {epoch+1}/{total_epochs} | "
                      f"Batch {batch_idx+1}/{num_batches} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Batch ETA: {batch_eta:.0f}s", flush=True)

        avg_loss = epoch_loss / num_items
        train_losses.append(avg_loss)
        elapsed = time.time() - t_start

        # Estimate remaining time
        epochs_done = epoch - start_epoch + 1
        epochs_left = total_epochs - epoch - 1
        avg_epoch_time = (time.time() - training_start) / epochs_done
        eta_min = avg_epoch_time * epochs_left / 60

        print(f"Epoch {epoch+1}/{total_epochs} done | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Time: {elapsed:.1f}s | "
              f"Step: {step} | "
              f"ETA: {eta_min:.1f} min", flush=True)

        # Save checkpoint periodically
        if (epoch + 1) % config.training.snapshot_freq == 0 or epoch == config.training.n_epochs - 1:
            ckpt_data = {
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'epoch': epoch,
                'step': step,
                'config': dict(config),
                'train_losses': train_losses,
            }
            if config.model.ema:
                ckpt_data['ema_state'] = ema_helper.state_dict()

            torch.save(ckpt_data, os.path.join(checkpoint_dir, f'ckpt_epoch_{epoch}.pth'))
            torch.save(ckpt_data, ckpt_path)  # Always overwrite latest
            print(f"  -> Saved checkpoint at epoch {epoch}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('DSM Loss')
    plt.title('NCSNv2 Training Loss')
    plt.grid(True)
    plt.savefig(os.path.join(checkpoint_dir, 'training_loss.png'), dpi=150)
    plt.close()
    print(f"\nTraining complete. Loss curve saved to {checkpoint_dir}/training_loss.png")

    return model


# =====================
# Measurement-Guided Langevin Dynamics
# =====================
@torch.no_grad()
def measurement_guided_langevin_dynamics(
    y, D_height, D_width, image_size, model, sigmas, config,
    n_steps_each=5, step_lr=0.0000062, likelihood_weight=1.0
):
    """
    Reconstruct HR image from LR measurement y using annealed Langevin dynamics
    with the score model as prior and the measurement model as likelihood.

    The update has two parts applied at each step:
      1. Prior (Langevin): x += eps * score(x) + sqrt(2*eps) * z
         where eps = step_lr * (sigma_i / sigma_L)^2
      2. Likelihood (data consistency): x += lambda * D^T(y - Dx)
         with a FIXED weight lambda, independent of step_size.
         (Scaling by step_size causes divergence at high sigma.)

    Args:
        y: Low-resolution measurement tensor (lr_h, lr_w) on device
        D_height: Downsample matrix for height (lr_h, hr_h) on device
        D_width: Downsample matrix for width (lr_w, hr_w) on device
        image_size: HR image size (int)
        model: Trained NCSNv2 score model
        sigmas: Sigma schedule tensor
        config: Configuration dict
        n_steps_each: Langevin steps per noise level
        step_lr: Base learning rate for Langevin dynamics
        likelihood_weight: Weight for the likelihood gradient term
    """
    device = y.device

    # Initialize from noise
    x = torch.randn(1, 1, image_size, image_size, device=device)

    reconstruction_process = []
    num_sigmas = len(sigmas)

    t_start = time.time()
    for c, sigma in enumerate(sigmas):
        # Label for score model (integer index into sigma schedule)
        labels = torch.ones(1, device=device).long() * c

        # Step size scales with sigma: larger steps at higher noise
        step_size = step_lr * (sigma / sigmas[-1]) ** 2

        for s in range(n_steps_each):
            # 1. Score (prior gradient): grad_x log p(x | sigma)
            score = model(x, labels)

            # 2. Likelihood gradient: grad_x log p(y | x)
            # For y = D_h @ x @ D_w^T, the gradient is D_h^T @ (y - D_h @ x @ D_w^T) @ D_w
            x_squeezed = x.squeeze(0).squeeze(0)  # (H, W)
            predicted_y = D_height @ x_squeezed @ D_width.T
            residual = y - predicted_y
            grad_likelihood = D_height.T @ residual @ D_width  # (H, W)
            grad_likelihood = grad_likelihood.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            # 3. Noise
            noise = torch.randn_like(x)

            # 4. Langevin update: prior step + data consistency step
            # Prior: step_size scales with sigma (large steps at high noise)
            # Likelihood: FIXED weight (not scaled by step_size!)
            x = x + step_size * score \
                + likelihood_weight * grad_likelihood \
                + torch.sqrt(2 * step_size) * noise

            # Safety: clamp to prevent divergence
            x = x.clamp(-2, 2)

        # Progress logging (every 10% of sigma levels)
        if c % max(1, num_sigmas // 10) == 0 or c == num_sigmas - 1:
            elapsed = time.time() - t_start
            eta = elapsed / (c + 1) * (num_sigmas - c - 1)
            print(f"    Sigma {c+1}/{num_sigmas} (σ={sigma:.4f}) | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", flush=True)

        # Save snapshots for visualization (20 evenly-spaced steps)
        if num_sigmas >= 20 and c % (num_sigmas // 20) == 0:
            reconstruction_process.append(x.detach().cpu().numpy())

    # Final denoising step (Song & Ermon 2019, Sec 3.2)
    last_labels = torch.ones(1, device=device).long() * (num_sigmas - 1)
    x = x + sigmas[-1] ** 2 * model(x, last_labels)

    reconstruction_process.append(x.detach().cpu().numpy())

    return x.squeeze().cpu().numpy(), reconstruction_process


# =====================
# Reconstruction
# =====================
def reconstruct(config, checkpoint_dir='checkpoints/ir_ncsn', output_dir='results/ir_ncsn'):
    """Load trained model and reconstruct HR images from LR measurements."""
    os.makedirs(output_dir, exist_ok=True)
    device = config.device

    # Load model
    model = NCSNv2Deeper(config).to(device)
    model = nn.DataParallel(model)

    ckpt_path = os.path.join(checkpoint_dir, 'latest.pth')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint found at {ckpt_path}. Train first.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Use EMA weights for better quality
    if config.model.ema and 'ema_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
        ema_helper.load_state_dict(ckpt['ema_state'])
        ema_helper.ema(model)
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt['model_state'])
        print("Loaded model weights (no EMA)")

    model.eval()

    # Sigma schedule
    sigmas = get_sigmas(config)

    # Measurement resolution and downsample factor
    image_size = config.data.image_size
    measurement_size = config.reconstruction.measurement_size
    if image_size % measurement_size != 0:
        print(f"ERROR: image_size ({image_size}) must be divisible by measurement_size ({measurement_size})")
        sys.exit(1)
    factor = image_size // measurement_size

    # Create downsample matrices
    D_height_np = create_downsample_matrix(image_size, factor)
    D_width_np = create_downsample_matrix(image_size, factor)
    D_height = torch.tensor(D_height_np, dtype=torch.float32).to(device)
    D_width = torch.tensor(D_width_np, dtype=torch.float32).to(device)

    print(f"Reconstruction: {measurement_size}x{measurement_size} -> "
          f"{image_size}x{image_size} ({factor}x upsampling)")

    # Load test images
    hr_dir = config.data.hr_test_dir
    if not os.path.exists(hr_dir):
        hr_dir = config.data.hr_train_dir  # Fallback
        print(f"Warning: No test dir found, using train dir")

    dataset = IRImageDataset(hr_dir=hr_dir, image_size=image_size)

    # Reconstruction parameters
    step_lr = config.sampling.step_lr
    n_steps_each = config.sampling.n_steps_each
    likelihood_weight = config.reconstruction.likelihood_weight

    # Reconstruct a few test images
    num_test = min(5, len(dataset))
    all_psnr = []
    all_nmse = []

    for i in range(num_test):
        print(f"\n--- Reconstructing image {i+1}/{num_test} ---")
        hr_img = dataset[i].to(device)  # (1, H, W)
        hr_np = hr_img.squeeze().cpu().numpy()

        # Create LR measurement: y = D_h @ x @ D_w^T
        y = D_height @ hr_img.squeeze() @ D_width.T

        t_start = time.time()
        recon, process = measurement_guided_langevin_dynamics(
            y=y,
            D_height=D_height,
            D_width=D_width,
            image_size=image_size,
            model=model,
            sigmas=sigmas,
            config=config,
            n_steps_each=n_steps_each,
            step_lr=step_lr,
            likelihood_weight=likelihood_weight,
        )
        elapsed = time.time() - t_start

        # Clip to valid range
        recon = np.clip(recon, 0, 1)

        # Metrics
        psnr_val = calculate_psnr(hr_np, recon)
        nmse_val = calculate_nmse(hr_np, recon)
        all_psnr.append(psnr_val)
        all_nmse.append(nmse_val)

        print(f"  PSNR: {psnr_val:.2f} dB | NMSE: {nmse_val:.6f} | Time: {elapsed:.1f}s")

        # Plot comparison
        lr_display = y.cpu().numpy()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(hr_np, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Ground Truth (HR)')
        axes[0].axis('off')

        axes[1].imshow(lr_display, cmap='gray')
        axes[1].set_title(f'Measurement ({measurement_size}x{measurement_size})')
        axes[1].axis('off')

        axes[2].imshow(recon, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f'Reconstructed\nPSNR: {psnr_val:.2f} dB')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'reconstruction_{i}.png'), dpi=200)
        plt.close()

        # Plot reconstruction process
        if process:
            num_steps = len(process)
            cols = min(5, num_steps)
            rows = (num_steps + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
            if rows == 1:
                axes = [axes] if cols == 1 else axes
            axes_flat = np.array(axes).flatten()
            for j, snapshot in enumerate(process):
                if j < len(axes_flat):
                    axes_flat[j].imshow(np.clip(snapshot[0, 0], 0, 1), cmap='gray')
                    axes_flat[j].set_title(f'Step {j}')
                    axes_flat[j].axis('off')
            for j in range(len(process), len(axes_flat)):
                axes_flat[j].axis('off')
            plt.suptitle(f'Reconstruction Process (Image {i})')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'process_{i}.png'), dpi=150)
            plt.close()

    # Summary
    print(f"\n{'='*50}")
    print(f"Reconstruction Summary ({num_test} images)")
    print(f"{'='*50}")
    print(f"  Avg PSNR: {np.mean(all_psnr):.2f} dB")
    print(f"  Avg NMSE: {np.mean(all_nmse):.6f}")
    print(f"  Results saved to {output_dir}/")


# =====================
# Generation (Unconditional)
# =====================
@torch.no_grad()
def generate(config, checkpoint_dir='checkpoints/ir_ncsn', output_dir='results/ir_ncsn'):
    """Generate new IR images unconditionally using annealed Langevin dynamics."""
    os.makedirs(output_dir, exist_ok=True)
    device = config.device

    # Load model
    model = NCSNv2Deeper(config).to(device)
    model = nn.DataParallel(model)

    ckpt_path = os.path.join(checkpoint_dir, 'latest.pth')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if config.model.ema and 'ema_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
        ema_helper.load_state_dict(ckpt['ema_state'])
        ema_helper.ema(model)
    else:
        model.load_state_dict(ckpt['model_state'])

    model.eval()

    sigmas = get_sigmas(config)
    image_size = config.data.image_size
    n_samples = 4
    step_lr = config.sampling.step_lr
    n_steps_each = config.sampling.n_steps_each

    # Initialize from noise
    x = torch.randn(n_samples, 1, image_size, image_size, device=device)

    print(f"Generating {n_samples} images with annealed Langevin dynamics...")
    t_start = time.time()

    for c, sigma in enumerate(sigmas):
        labels = torch.ones(n_samples, device=device).long() * c
        step_size = step_lr * (sigma / sigmas[-1]) ** 2

        for s in range(n_steps_each):
            score = model(x, labels)
            noise = torch.randn_like(x)
            x = x + step_size * score + torch.sqrt(2 * step_size) * noise

        if c % 50 == 0:
            print(f"  Sigma level {c}/{len(sigmas)}, sigma={sigma:.4f}")

    # Final denoising
    if config.sampling.denoise:
        last_labels = torch.ones(n_samples, device=device).long() * (len(sigmas) - 1)
        x = x + sigmas[-1] ** 2 * model(x, last_labels)

    elapsed = time.time() - t_start
    print(f"Generation took {elapsed:.1f}s")

    # Visualize
    x = x.clamp(0.0, 1.0)
    grid = make_grid(x, nrow=2, padding=2)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title('Generated IR Images (NCSNv2)')
    plt.savefig(os.path.join(output_dir, 'generated_samples.png'), dpi=200)
    plt.close()
    print(f"Saved generated samples to {output_dir}/generated_samples.png")


# =====================
# Main Entry Point
# =====================
def main():
    parser = argparse.ArgumentParser(description='NCSNv2-based IR image super-resolution')
    parser.add_argument('--task', type=str, default='train',
                        choices=['train', 'reconstruct', 'generate'],
                        help='Task to perform')
    parser.add_argument('--config', type=str, default='configs/ir_128x128.yml',
                        help='Path to YAML config file')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory for model checkpoints (default: auto from config name)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for output results (default: auto from config name)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    args = parser.parse_args()

    # Auto-derive directories from config filename if not specified
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join('checkpoints', config_name)
    if args.output_dir is None:
        args.output_dir = os.path.join('results', config_name)

    # GPU selection
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Disable TF32 for precision
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    # Load config
    config = load_config(args.config)
    print(f"Task: {args.task}")
    print(f"Config: {args.config}")
    print(f"Device: {config.device}")
    print(f"Image size: {config.data.image_size}x{config.data.image_size}")
    print(f"Model: NCSNv2Deeper (ngf={config.model.ngf})")
    print()

    if args.task == 'train':
        train(config, checkpoint_dir=args.checkpoint_dir)
    elif args.task == 'reconstruct':
        reconstruct(config, checkpoint_dir=args.checkpoint_dir, output_dir=args.output_dir)
    elif args.task == 'generate':
        generate(config, checkpoint_dir=args.checkpoint_dir, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
