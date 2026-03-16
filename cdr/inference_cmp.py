import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

from dataloader import ConvectionDiffusionDataset
from baseline import FNO_solver

def run_inference():
    # --- Configuration ---
    # Ensure these match the parameters used during training!
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float64
    TEST_PATH = 'data/test'
    MODEL_SAVE_PATH = 'models/fno_solver_precond.pt'
    CBS_PATH = '/home/wangjt/work/precond_loss/convection_diffusion/iter_his_cbs.pt'
    fno__noprecond_path = '/home/wangjt/work/precond_loss/convection_diffusion/iter_his_fno_noprecond.pt'
    # Model architecture params
    RESOLUTION = (128, 128)
    DX = 1.0 / RESOLUTION[0]
    FEATURES = 32
    WAVENUMBER = [16, 16, 32] 
    PADDING = 8
    
    # Inference params
    ITER_NUM = 2000  # Max iterations for the solver
    TOLERANCE = 1e-4# Convergence tolerance

    print(f"Using device: {DEVICE}")


    test_dataset = ConvectionDiffusionDataset(TEST_PATH, data_num=200, device=DEVICE)
    # No need for a full DataLoader, just get the first sample
    coeffs_sample = test_dataset[0].unsqueeze(0) # Add batch dimension

    # --- Load Model ---
    print(f"Loading model from {MODEL_SAVE_PATH}...")

    # Instantiate the model with the same architecture
    model = FNO_solver(N=RESOLUTION, dx=DX, num_features=FEATURES, 
                       wavenumber=WAVENUMBER, padding=PADDING, device=DEVICE,dtype=DTYPE).to(DEVICE)
    
    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval() # Set the model to evaluation mode

    print("Model loaded successfully.")

    # --- Run Iterative Solver ---
    print("\nStarting iterative solver...")
    u_solved, final_residual, loss_his = model.solve(
        coeffs_sample, 
        iter_num=ITER_NUM, 
        tol=TOLERANCE
    )

    print("\n--- Inference Results ---")
    print(f"Solver finished in {len(loss_his)} iterations.")
    print(f"Final Relative Residual: {final_residual:.4e}")

    # --- Visualization ---
    coeffs_cpu = coeffs_sample.squeeze(0).cpu().numpy()
    u_solved_cpu = u_solved.squeeze(0).squeeze(0).cpu().numpy()

    kappa = coeffs_cpu[0]
    vx = coeffs_cpu[1]
    vy = coeffs_cpu[2]
    f = coeffs_cpu[3]
    v_mag = np.sqrt(vx**2 + vy**2)

    # Two rows, three columns (last one for loss history)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = axes.flatten()
    # fig.suptitle("FNO Solver Inference Results", fontsize=16)

    im0 = axes[0].imshow(kappa, cmap='viridis')
    axes[0].set_title("Kappa (κ)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(v_mag, cmap='viridis')
    axes[1].set_title("Velocity Magnitude |v|")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(f, cmap='Reds')
    axes[2].set_title("Source (f)")
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("inference_cmp_result_0.png")
    print("\nSaved visualization to 'inference_result.png'")

    # Two rows, three columns (last one for loss history)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes = axes.flatten()

    im3 = axes[0].imshow(u_solved_cpu, cmap='jet')
    axes[0].set_title(f"Solution (u)")
    fig.colorbar(im3, ax=axes[0], shrink=0.8)
    classical_iter_his = torch.load(CBS_PATH)
    fno_iter_his = torch.load(fno__noprecond_path)
    axes[1].plot(classical_iter_his,'-^',label='classical')
    axes[1].plot(fno_iter_his[:50],'-*',label='FNO')
    axes[1].plot(loss_his,'-o', label='LS-FNO')


    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].set_title("Convergence History")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("inference_cmp_result_1.png")
    print("\nSaved visualization to 'inference_result.png'")


if __name__ == '__main__':
    run_inference()