import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F


def build_ill_config(ill_level):
    """
    Map ill_level in [0, 1] to data-generation parameters.
    Larger ill_level => stronger coefficient contrast and convection dominance.
    """
    level = float(np.clip(ill_level, 0.0, 1.0))
    return {
        'kappa_grf_scale': 0.2 + 0.8 * level,
        'kappa_floor': 0.5 - 0.45 * level,
        'psi_scale': 0.01 + 0.08 * level,
        'num_sources_range': (1, 2) if level < 0.4 else (2, 4),
        'amplitude_range': (10.0 + 10.0 * level, 20.0 + 40.0 * level),
        'std_range': (
            max(0.005, 0.02 - 0.015 * level),
            max(0.015, 0.10 - 0.06 * level)
        ),
    }

def finite_difference_derivative(field, dx, axis):
    """
    Computes a first derivative of a 2D field using second-order central differences
    for the interior and first-order differences at the boundaries.

    Args:
        field (torch.Tensor): The 2D input tensor (H, W).
        dx (float): The grid spacing.
        axis (int): The axis for the derivative (0 for y, 1 for x).
    
    Returns:
        torch.Tensor: The derivative of the field with the same shape.
    """
    if field.ndim != 2:
        raise ValueError("Input 'field' must be a 2D tensor.")

    # Create an output tensor to store the derivatives
    deriv = torch.zeros_like(field)

    if axis == 0:  # Derivative with respect to y (rows)
        # Central difference for the interior
        deriv[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dx)
        # Forward difference for the top boundary
        deriv[0, :] = (field[1, :] - field[0, :]) / dx
        # Backward difference for the bottom boundary
        deriv[-1, :] = (field[-1, :] - field[-2, :]) / dx
        
    elif axis == 1:  # Derivative with respect to x (columns)
        # Central difference for the interior
        deriv[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dx)
        # Forward difference for the left boundary
        deriv[:, 0] = (field[:, 1] - field[:, 0]) / dx
        # Backward difference for the right boundary
        deriv[:, -1] = (field[:, -1] - field[:, -2]) / dx
        
    else:
        raise ValueError("'axis' must be 0 or 1.")

    return deriv

def generate_grf(low_res_shape, high_res_shape, scale=1.0, device='cpu'):
    """
    Generates a single Gaussian Random Field with a mean of 0 and std of 1,
    then scales it.
    """
    noise = torch.randn(1, 1, *low_res_shape, device=device)
    # Use bicubic interpolation for smoothness
    field = F.interpolate(noise, size=high_res_shape, mode='bicubic', align_corners=False)
    field = field.squeeze(0).squeeze(0)
    # Normalize
    field = (field - field.mean()) / field.std()
    return scale * field

def generate_blob_sources(shape, num_sources_range=(1, 4), amplitude_range=(100, 500), std_range=(0.01, 0.05), device='cpu'):
    """
    Generates a field with a few localized Gaussian "blob" sources.
    """
    grid_y = torch.linspace(0, 1, shape[0], device=device)
    grid_x = torch.linspace(0, 1, shape[1], device=device)
    Y, X = torch.meshgrid(grid_y, grid_x, indexing='ij')
    
    field = torch.zeros(shape, device=device)
    num_sources = np.random.randint(*num_sources_range)
    
    for _ in range(num_sources):
        # Randomize parameters for each blob
        cx, cy = np.random.rand(), np.random.rand()
        amp = amplitude_range[0] + (amplitude_range[1] - amplitude_range[0]) * np.random.rand()
        std = std_range[0] + (std_range[1] - std_range[0]) * np.random.rand()
        
        # Add the Gaussian blob to the field
        g = amp * torch.exp(-((X - cx)**2 + (Y - cy)**2) / (std**2)) # Note: removed 2* from denominator for wider blobs
        field += g
        
    return field

def generate_dataset(num_samples, shape=(128, 128), low_res_shape=(16, 16), save_path='data/train', device='cpu', ill_level=0.0):
    """
    Generates and saves the convection-diffusion dataset.
    - kappa: from GRF, positive
    - v: from stream function (GRF), incompressible
    - f: from localized blob sources
    - sigma: constant 1.0
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    dx = 1.0 / shape[0]
        
    print(f"Generating {num_samples} samples and saving to {save_path} (ill_level={ill_level:.2f})...")
    for i in tqdm(range(num_samples)):
        sample_level = float(np.clip(np.random.uniform(0.8, 1.2) * ill_level, 0.0, 1.0))
        cfg = build_ill_config(sample_level)

        # 1. Generate kappa (diffusion coefficient)
        # Exponentiation ensures kappa is strictly positive
        kappa = cfg['kappa_floor'] * torch.exp(
            generate_grf(low_res_shape, shape, scale=cfg['kappa_grf_scale'], device=device)
        ) + cfg['kappa_floor']

        # 2. Generate v (incompressible velocity field)
        # First, generate a stream function psi as a GRF
        psi = generate_grf(low_res_shape, shape, scale=cfg['psi_scale'], device=device)
        # Then, compute v = (d(psi)/dy, -d(psi)/dx)
        # Note: axis 0 is y, axis 1 is x for a (H, W) tensor
        vx = finite_difference_derivative(psi, dx, axis=0)
        vy = -finite_difference_derivative(psi, dx, axis=1)
        v = torch.stack([vx, vy], dim=-1) # Shape (H, W, 2)

        # 3. Generate f (source term)
        f = generate_blob_sources(
            shape,
            num_sources_range=cfg['num_sources_range'],
            amplitude_range=cfg['amplitude_range'],
            std_range=cfg['std_range'],
            device=device
        )
        
        # 4. Define sigma (reaction term) as a constant
        # sigma = torch.ones(shape, device=device)

        # Save the sample as a dictionary of float32 tensors on CPU
        sample = {
            'kappa': kappa.cpu().to(torch.float32),
            'v': v.cpu().to(torch.float32),
            # 'sigma': sigma.cpu().to(torch.float32),
            'f': f.cpu().to(torch.float32)
        }
        
        torch.save(sample, os.path.join(save_path, f"sample_{i:04d}.pt"))

    print(f"Dataset successfully generated.")

if __name__ == '__main__':
    # Use GPU for generation if available, as it's much faster
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for data generation.")

    # --- Parameters ---
    N_train = 5000
    N_test = 500
    resolution = (128, 128)
    low_res = (16, 16) # Controls the "smoothness" of the random fields
    ill_level_train = 0.4
    ill_level_test = 0.4

    # --- Generate Training Data ---
    train_path = 'data_ill/train'
    generate_dataset(
        num_samples=N_train,
        shape=resolution,
        low_res_shape=low_res,
        save_path=train_path,
        device=device,
        ill_level=ill_level_train
    )

    # --- Generate Testing Data ---
    test_path = 'data_ill/test'
    generate_dataset(
        num_samples=N_test,
        shape=resolution,
        low_res_shape=low_res,
        save_path=test_path,
        device=device,
        ill_level=ill_level_test
    )
