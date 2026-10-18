"""Logging, timing, and result visualization for Helmholtz runners."""

import os
import sys
import time
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt


class Timer:
    def __init__(self, device='cpu'):
        """
        Args:
            device: str, 'cpu' or 'cuda'
        """
        self.device = device
        if 'cuda' in self.device and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

    def record(self):
        """
        time marker
        
        Returns:
            float or torch.cuda.Event
        """
        if self.device == 'cpu':
            return time.perf_counter()
        elif 'cuda' in self.device: 
            with torch.cuda.device(self.device):
                evt = torch.cuda.Event(enable_timing=True)
                evt.record()
            return evt
        else:
            raise NotImplementedError("no such device! ")

    def elapsed_time(self, start, end):
        """
        calculate the time between start and end in second
        
        """
        if 'cpu' in self.device:
            return end - start
        elif 'cuda' in self.device: 
            torch.cuda.synchronize(self.device)
            ms = start.elapsed_time(end)
            return ms * 1e-3
        else:
            raise NotImplementedError("no such device! ")


class Logger:
    def __init__(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(save_path),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger()

    def __call__(self, msg):
        self.logger.info(msg)


def plot_result_one_method(u, sos, iter_history, src_position, id, 
                           boundary_width,plot_pml,solve_time,
                           save_dir='./', save_name='solve_result'):
    if plot_pml == True:
        u_real = torch.view_as_real(u)[0,0,:,:,0].cpu().numpy()
    else:
        u_real = torch.view_as_real(u)[0,0,:,:,0].cpu().numpy()[boundary_width:-boundary_width, boundary_width:-boundary_width]
    sos    = sos[0,0,:,:].cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    img0 = axes[0].imshow(sos, interpolation='none', cmap='jet')
    # img0 = axes[0].imshow(sos, interpolation='none', cmap='gray')
    axes[0].set_title('Speed of Sound', fontsize=14)
    fig.colorbar(img0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].scatter(src_position[1], src_position[0], s=100, c='w', marker='x')
    # axes[0].scatter(src_position[1]//2, src_position[0], s=100, c='w', marker='x')
    # axes[0].scatter(src_position[1]*3//2, src_position[0], s=100, c='w', marker='x')

    vmax = max(float(np.abs(u_real).max()) * 0.6, np.finfo(float).eps)
    img1 = axes[1].imshow(u_real, interpolation='none', cmap='seismic', vmin=-vmax, vmax=vmax)
    # img1 = axes[1].imshow(u_real, interpolation='none', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Wavefield Real', fontsize=14)
    fig.colorbar(img1, ax=axes[1], fraction=0.046, pad=0.04)

    img3 = axes[2].semilogy(np.arange(len(iter_history))+1,iter_history, 'o-')
    axes[2].set_title(f'Iteration History: {solve_time:.2f} s', fontsize=14)
    axes[2].grid(True)
    axes[2].set_xlabel('Iteration', fontsize=12)
    axes[2].set_ylabel("Rel residual", fontsize=12)

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.tight_layout()
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    plt.savefig(f'{save_dir}/{save_name}_{id}.png', dpi=400)
    plt.close()


def plot_result_one_method_3d(u, sos, iter_history, id,
                              boundary_width, plot_pml, solve_time,
                              save_dir='.', save_name='solve_result_3d'):
    """Save central orthogonal slices using only Matplotlib."""
    wave = u[0, 0].detach().cpu().real.numpy()
    speed = sos[0, 0].detach().cpu().numpy()
    if not plot_pml and boundary_width:
        crop = (slice(boundary_width, -boundary_width),) * 3
        wave = wave[crop]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    limit = max(float(np.abs(wave).max()), np.finfo(float).eps)
    for axis in range(3):
        velocity_slice = np.take(speed, speed.shape[axis] // 2, axis=axis)
        wave_slice = np.take(wave, wave.shape[axis] // 2, axis=axis)
        first = axes[0, axis].imshow(velocity_slice, cmap='viridis')
        second = axes[1, axis].imshow(wave_slice, cmap='seismic', vmin=-limit, vmax=limit)
        axes[0, axis].set_title(f'Velocity, axis {axis}')
        axes[1, axis].set_title(f'Real wavefield, axis {axis}')
        fig.colorbar(first, ax=axes[0, axis], shrink=0.8)
        fig.colorbar(second, ax=axes[1, axis], shrink=0.8)
    axes[0, 3].semilogy(np.arange(len(iter_history)) + 1, iter_history)
    axes[0, 3].set_title(f'Residual history ({solve_time:.3f} s)')
    axes[0, 3].set_xlabel('Iteration')
    axes[0, 3].set_ylabel('True relative residual')
    axes[0, 3].grid(True)
    axes[1, 3].axis('off')
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'{save_name}_{id}.png'), dpi=150)
    plt.close(fig)


def trainable_params_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)*4/(1024**2)


def save_loss_history(dir_path, loss_list, filename, 
                      title='Training Loss History',x_label='Epoch',
                      log_scale=True, fig_size=(8, 6), dpi=300):

    txt_path = os.path.join(dir_path, f"{filename}.txt")
    with open(txt_path, 'w') as f:
        for loss in loss_list:
            f.write(f"{loss:.5e}\n")

    fig, ax = plt.subplots(figsize=fig_size)
    if log_scale:
        ax.semilogy(loss_list, linewidth=1.5)
    else:
        ax.plot(loss_list, linewidth=1.5)
    
    ax.set_title(title, fontsize=12, pad=20)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    img_path = os.path.join(dir_path, f"{filename}.png")
    plt.savefig(img_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
