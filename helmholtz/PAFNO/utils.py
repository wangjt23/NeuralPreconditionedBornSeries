import os
import sys
import time
import random
import torch
import scipy.io
import logging
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Circle
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
            torch.cuda.synchronize()
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

    vmax = np.max(u_real)*0.6
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

def plot_result_one_method_error(u, sos, iter_history, src_position, id, 
                           boundary_width,plot_pml,solve_time,
                           save_dir='./', save_name='solve_result'):
    if plot_pml == True:
        u_real = torch.view_as_real(u)[0,0,:,:,0].cpu().numpy()
    else:
        u_real = torch.view_as_real(u)[0,0,:,:,0].cpu().numpy()[boundary_width:-boundary_width, boundary_width:-boundary_width]
    sos    = sos[0,0,:,:].cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    img0 = axes[0].imshow(sos, interpolation='none', cmap='RdBu_r')
    axes[0].set_title('Speed of Sound', fontsize=14)
    fig.colorbar(img0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].scatter(src_position[1], src_position[0], s=100, c='w', marker='x')
    # axes[0].scatter(src_position[1]//2, src_position[0], s=100, c='w', marker='x')
    # axes[0].scatter(src_position[1]*3//2, src_position[0], s=100, c='w', marker='x')

    vmax = np.max(u_real)*0.6
    img1 = axes[1].imshow(u_real, interpolation='none', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Wavefield Real', fontsize=14)
    fig.colorbar(img1, ax=axes[1], fraction=0.046, pad=0.04)

    img3 = axes[2].semilogy(np.arange(len(iter_history))+1,iter_history, 'o-')
    axes[2].set_title(f'Iteration History: {solve_time:.2f} s', fontsize=14)
    axes[2].grid(True)
    axes[2].set_xlabel('Iteration', fontsize=12)
    axes[2].set_ylabel("Rel error", fontsize=12)

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.tight_layout()
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    plt.savefig(f'{save_dir}/{save_name}_{id}_error.png', dpi=400)
    plt.close()


def plot_result_one_method_3d(u, sos, iter_history, id,
                              boundary_width, plot_pml, solve_time,
                              save_dir='./', save_name='solve_result_3d'):
    import pyvista as pv
    if plot_pml:
        u_real = u[0, 0, :, :, :].cpu().numpy().real
        sos_np = sos[0,0, :, :, :].cpu().numpy()
    else:
        # Crop PML boundaries
        u_real = u[0, 0, boundary_width:-boundary_width, boundary_width:-boundary_width, boundary_width:-boundary_width].cpu().numpy()
        sos_np = sos[0, 0, boundary_width:-boundary_width, boundary_width:-boundary_width, boundary_width:-boundary_width].cpu().numpy()

    # Create the main figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # =============================================================================
    # --- Plot 1: 3D SOS Model ---
    # =============================================================================
    grid_sos = pv.ImageData(dimensions=sos_np.shape)
    grid_sos.point_data["values"] = sos_np.flatten(order="F") 

    plotter_sos = pv.Plotter(off_screen=True, window_size=[800, 800])
    plotter_sos.hide_axes()

    common_style = {'cmap': 'jet', 'show_edges': False, 'show_scalar_bar': False}
    plotter_sos.add_mesh(grid_sos.slice(normal='x'), **common_style)
    plotter_sos.add_mesh(grid_sos.slice(normal='y'), **common_style)
    plotter_sos.add_mesh(grid_sos.slice(normal='z'), **common_style)


    plotter_sos.add_scalar_bar(vertical=True, position_x=0.8, position_y=0.1, height=0.3)
    plotter_sos.camera_position = 'iso'

    img_sos = plotter_sos.screenshot(return_img=True)
    axes[0].imshow(img_sos)
    axes[0].set_title('3D Speed of Sound', fontsize=14)
    axes[0].axis('off')

    # =============================================================================
    # --- Plot 2: 3D Wavefield ---
    # =============================================================================
    grid_wave = pv.ImageData(dimensions=u_real.shape)
    grid_wave.point_data["amplitude"] = u_real.flatten(order="F")

    vmean = np.mean(np.abs(u_real)) * 5  # Adjust sensitivity
    clim = [-vmean, vmean]

    plotter_wave = pv.Plotter(off_screen=True, window_size=[800, 800])
    plotter_wave.hide_axes()

    common_style_wave = {"cmap": 'seismic', "clim": clim, "show_edges": False, 'show_scalar_bar': False}
    plotter_wave.add_mesh(grid_wave.slice(normal='x'), **common_style_wave)
    plotter_wave.add_mesh(grid_wave.slice(normal='y'), **common_style_wave)
    plotter_wave.add_mesh(grid_wave.slice(normal='z'), **common_style_wave)

    plotter_wave.add_scalar_bar(vertical=True, position_x=0.8, position_y=0.1, height=0.3)
    plotter_wave.camera_position = 'iso'

    img_wave = plotter_wave.screenshot(return_img=True)
    axes[1].imshow(img_wave)
    axes[1].set_title('3D Wavefield', fontsize=14)
    axes[1].axis('off')

    # =============================================================================
    # --- Plot 3: Iteration History ---
    # =============================================================================
    axes[2].semilogy(np.arange(len(iter_history)) + 1, iter_history, 'o-')
    axes[2].set_title(f'Iteration History: {solve_time:.2f} s', fontsize=14)
    axes[2].grid(True)
    axes[2].set_xlabel('Iteration', fontsize=12)
    axes[2].set_ylabel("Rel residual", fontsize=12)

    plt.tight_layout()

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    plt.savefig(f'{save_dir}/{save_name}_{id}.png', dpi=300, bbox_inches='tight')
    plt.close()

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

def read_marmousi_vp(fn):
    '''
    url: https://www.geoazur.fr/WIND/bin/view/Main/Data/Marmousi
    '''
    data = np.fromfile(fn,dtype=np.float32)
    data = data.reshape(751, 2301, order='F')
    return data


def plot_coarse_level(self,rhs, a_list, inv_a_list):
    num_levels = len(self.num_iteration)
    check_levels = range(1,len(self.num_iteration)) 
    fig = plt.figure(figsize=(18, 10))
    widths = [0.5] + [1]*(len(check_levels)) 
    gs = gridspec.GridSpec(3, len(check_levels)+1, width_ratios=widths, figure=fig)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(len(check_levels)+1)] for i in range(3)]
    # fig, axes = plt.subplots(3, len(check_levels)+1, figsize=(19, 9))
    u = None
    rhs_ori = rhs.clone()
    out_list = [0] * len(self.num_iteration)
    for l in range(len(self.num_iteration)):
        u, a, _, rhs, res = self.Down_sweep_layers[l]((u, a_list[l], inv_a_list[l], rhs))
        out_list[l] = (u, rhs)
        if l < len(self.num_iteration) - 1:
            u, rhs = self.R_list[l]((u, a, rhs, res))
    for j in range(len(self.num_iteration) - 2, -1, -1):
        u, rhs = out_list[j][0], out_list[j][1]
        if not isinstance(out_list[j+1][0], int):
            u = u + self.RT_list[j](out_list[j+1][0])
        out_list[j] = (u, rhs)
    axes[0][0].axis("off")
    axes[0][1].axis("off")
    axes[0][2].axis("off")
    axes[0][4].axis("off")
    axes[0][5].axis("off")
    # axes[0][6].axis("off")
    ax = axes[0][3]
    d = out_list[0][0][0,10].cpu().detach()
    im = ax.imshow(d,vmax=d.max()*1, vmin=-d.max()*1)
    ax.set_title(r"$u$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)   
    for idx, level_check in enumerate(check_levels):
        u = None
        out_list = [0] * len(self.num_iteration)
        rhs = rhs_ori
        for l in range(len(self.num_iteration)):
            u, a, _, rhs, res = self.Down_sweep_layers[l]((u, a_list[l], inv_a_list[l], rhs))
            out_list[l] = (u, rhs)
            if l < len(self.num_iteration) - 1:
                u, rhs = self.R_list[l]((u, a, rhs, res))
        for i in range(num_levels):
            if i != level_check:
                out_list[i] = (0,0)
        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, rhs = out_list[j][0], out_list[j][1]
            if not isinstance(out_list[j+1][0], int):
                u = u + self.RT_list[j](out_list[j+1][0])
            
            out_list[j] = (u, rhs)    

    
        ax = axes[1][idx+1]
        d1 = out_list[level_check][0][0,10].cpu().detach()
        im = ax.imshow(d1,vmax=d1.max()*1, vmin=-d1.max()*1)
        ax.set_title(f"Level {level_check}")
        # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04) 

        ax = axes[2][idx+1]
        d2 = out_list[0][0][0,10].cpu().detach()
        im = ax.imshow(d2,vmax=d2.max()*1, vmin=-d2.max()*1)
        ax.set_title(f"Level {level_check}")
        # ax.set_title(f"u reconstruct ")
        # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[1][0].axis("off")
    axes[1][0].text(0.5, 0.5, r"$u$ $coarse$",
                    ha="center", va="center", fontsize=16)
    axes[2][0].axis("off")
    axes[2][0].text(0.5, 0.5, r"$Contribution$",
                    ha="center", va="center", fontsize=16)
    plt.tight_layout()
    plt.savefig("coarse_level.pdf")
    exit()

def plot_basis(self, rhs, a_list, inv_a_list):
    u = None
    out_list = [0] * len(self.num_iteration)
    for l in range(len(self.num_iteration)):
        u, a, _, rhs, res = self.Down_sweep_layers[l]((u, a_list[l], inv_a_list[l], rhs))
        out_list[l] = (u, rhs)
        if l < len(self.num_iteration) - 1:
            u, rhs = self.R_list[l]((u, a, rhs, res))
    
    idxs = 5
    x1 = torch.tensor([1, 34, 18, 1, 34])
    x2 = torch.tensor([1, 1, 18, 34, 34])
    fig = plt.figure(figsize=(20, 8))
    widths = [0.5] + [1]*(idxs) 
    gs = gridspec.GridSpec(2, idxs+1, width_ratios=widths, figure=fig)

    axes = [[fig.add_subplot(gs[i, j]) for j in range(idxs+1)] for i in range(2)]
    axins_config = {'xx':[[0, 45],[0, 45],[576//2-18, 576//2+27], [576-47, 576-2], [576-47, 576-2]],
                    'yy':[[0, 45],[576-46, 576-1],[576//2-18, 576//2+27], [0, 45], [576-47, 576-2]],
                    "loc":['center left','center left','upper center','center right','center right'],
                    'loc1':[1, 3, 3, 1, 3],
                    'loc2':[2, 4, 4, 2, 4],}
    for idx in range(idxs):
        ax1 = axes[0][idx+1]
        ax2 = axes[1][idx+1]
        num_levels = len(self.num_iteration)
        for i in range(num_levels):
            if i != num_levels-1:
                out_list[i] = (0,out_list[i][1])
        basis = torch.zeros_like(out_list[num_levels-1][0])
        basis[:,:,x1[idx],x2[idx]] = 1.
        # basis = out_list[num_levels-1][0]

        out_list[num_levels-1] = (basis, 0)
        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, rhs = out_list[j][0], out_list[j][1]
            # rhs = torch.rand_like(rhs)
            if not isinstance(out_list[j+1][0], int):
                u = u + self.RT_list[j](out_list[j+1][0])
                # u, _, _, rhs, _ = self.Up_sweep_layers[j]((u, a_list[j], inv_a_list[j], rhs, None))
            out_list[j] = (u, rhs)
        ax1.imshow(out_list[num_levels-1][0][0,1].cpu().detach())
        ax1.set_title(f"{idx}")
        uu = out_list[0][0][0,1].cpu().detach()
        ax2.imshow(uu)
        ax2.set_title(f"{idx}")
        xx1, xx2 = axins_config['xx'][idx]
        yy1, yy2 = axins_config['yy'][idx]
        axins = inset_axes(ax2, width="30%", height="30%", loc=axins_config['loc'][idx])

        axins.imshow(uu)
        axins.set_xlim(xx1, xx2)
        axins.set_ylim(yy2, yy1)
        axins.set_xticks([])
        axins.set_yticks([])
        mark_inset(ax2, axins, loc1=axins_config['loc1'][idx], loc2=axins_config['loc2'][idx], 
                    fc="none", ec="0.15", linewidth=0.5,linestyle="-" )

    axes[0][0].axis("off")
    axes[0][0].text(0.5, 0.5, r"$Level$ $4$ $basis$",
                    ha="center", va="center", fontsize=16)
    axes[1][0].axis("off")
    axes[1][0].text(0.5, 0.5, r"$Contribution$",
                    ha="center", va="center", fontsize=16)
    plt.tight_layout()
    plt.savefig('level4_basis.png')
    exit()


def plot_frequency_error_step(u_current, u_truth, iter_idx, save_dir, save_name, sample_idx):
    error_spatial = (u_current - u_truth).squeeze() # Shape: (H, W)
    fft_error = torch.fft.fft2(error_spatial, dim=(-2, -1))
    fft_error_shifted = torch.fft.fftshift(fft_error, dim=(-2, -1))

    magnitude_spectrum = torch.abs(fft_error_shifted)
    log_magnitude = torch.log10(magnitude_spectrum + 1e-16)
    img_data = log_magnitude.cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150) # Square figsize ensures circle looks like a circle

    h = 0.0005
    k_max_val = np.pi / h 
    extent = [-k_max_val, k_max_val, -k_max_val, k_max_val]
    im = ax.imshow(img_data, cmap='inferno', origin='lower', interpolation='nearest', extent=extent)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=10)
    # cbar.set_label(r'$\log_{10}(|\mathcal{F}(u - u_{truth})|)$', fontsize=8)
    
    ax.set_title(f"Spectral Error ($\\log_{{10}}(|\\mathcal{{F}}(u - u_{{truth}})|)$) - Iter {iter_idx}", fontsize=12)
    ax.set_xlabel(r"$k_x$ (rad/m)", fontsize=10)
    ax.set_ylabel(r"$k_y$ (rad/m)", fontsize=10)
    circle_list = []
    target_k = 2100 / 10
    circle_list.append(Circle((0, 0), target_k, fill=False, color='blue', linestyle='--', linewidth=1.0, alpha=0.8, label='k/10'))
    target_k = 2100 / 2
    circle_list.append(Circle((0, 0), target_k, fill=False, color='green', linestyle='--', linewidth=1.0, alpha=0.8, label='k/2'))
    target_k = 2100
    circle_list.append(Circle((0, 0), target_k, fill=False, color='red', linestyle='--', linewidth=1.0, alpha=0.8, label='k=2100'))
    target_k = 2100 * 2
    circle_list.append(Circle((0, 0), target_k, fill=False, color='black', linestyle='--', linewidth=1.0, alpha=0.8, label='2k'))
    for i in range(len(circle_list)):
        ax.add_patch(circle_list[i])
    
    ax.legend(loc='upper right', framealpha=0.6, fontsize=8)

    ticks = np.arange(-k_max_val, k_max_val, 2000)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    plt.tight_layout()
    filename = f"fft_error_{save_name}_{sample_idx}_iter_{iter_idx:03d}.pdf"
    save_path = os.path.join(save_dir, "fft_analysis")
    os.makedirs(save_path, exist_ok=True)
    
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)

def compute_band_error_norm(u_current, u_truth, R_grid, target_k_list, bandwidth=200.0):
    error_spatial = (u_current - u_truth).squeeze()
    fft_error = torch.fft.fft2(error_spatial, dim=(-2, -1))
    fft_error_shifted = torch.fft.fftshift(fft_error, dim=(-2, -1))

    magnitude_spectrum = torch.abs(fft_error_shifted)
    
    band_errors = []
    for k_val in target_k_list:
        mask = (R_grid >= (k_val - bandwidth)) & (R_grid < (k_val + bandwidth))
        masked_error = magnitude_spectrum[mask]
        if masked_error.numel() > 0:
            error_norm = torch.linalg.norm(masked_error).item()
        else:
            error_norm = 1e-16
            
        band_errors.append(error_norm)
        
    return band_errors

def plot_spectral_convergence(history_dict, save_dir, save_name, iter_history, sos, src_point_loc, u, sample_idx):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    
    iters = np.arange(1, len(next(iter(history_dict.values()))) + 1)
    
    colors = {'k/10': 'blue', 'k/2': 'green', 'k': 'red', '2k': 'black'}
    styles = {'k/10': '--', 'k/2': '-.', 'k': '-', '2k': ':'}
    
    for label, values in history_dict.items():
        log_values = values#np.log10(np.array(values) + 1e-16)
        ax.plot(iters, log_values, label=label, color=colors[label], linestyle=styles[label], linewidth=1.5)
    ax.set_yscale('log')
    ax.set_xlabel("Iteration",fontsize=12)
    ax.set_ylabel("Spectral Component Error",fontsize=12)
    ax.set_title(f"Convergence of Spectral Component Error",fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    
    filename = f"spectral_conv_{save_name}_{sample_idx}.pdf"
    save_path = os.path.join(save_dir, "convergence_analysis")
    os.makedirs(save_path, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    # plot reltive error
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    iters = np.arange(1, len(iter_history) + 1)
    ax.plot(iters, iter_history, 'o-', linewidth=1.5)
    ax.set_yscale('log')
    ax.set_xlabel("Iteration",fontsize=12)
    ax.set_ylabel("Reltive Error",fontsize=12)
    ax.set_title(f"Iteration History",fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)    
    filename = f"conv_{save_name}_{sample_idx}.pdf"
    save_path = os.path.join(save_dir, "convergence_analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    # plot sos
    sos    = sos[0,0,:,:].cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    
    img = ax.imshow(sos, interpolation='none', cmap='RdBu_r')
    ax.set_title('Speed of Sound', fontsize=14)
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    ax.scatter(src_point_loc[1], src_point_loc[0], s=100, c='w', marker='x')
    # axes[0].scatter(src_position[1]//2, src_position[0], s=100, c='w', marker='x')
    # axes[0].scatter(src_position[1]*3//2, src_position[0], s=100, c='w', marker='x')
    plt.tight_layout()
    filename = f"sos_{save_name}_{sample_idx}.pdf"
    save_path = os.path.join(save_dir, "convergence_analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    # plot u
    fig, ax = plt.subplots(figsize=(5, 5))
    u_real = torch.view_as_real(u)[0,0,:,:,0].cpu().numpy()
    vmax = np.max(u_real)*0.6
    img = ax.imshow(u_real, interpolation='none', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title('Wavefield Real', fontsize=14)
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    filename = f"u_real_{save_name}_{sample_idx}.pdf"
    save_path = os.path.join(save_dir, "convergence_analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
