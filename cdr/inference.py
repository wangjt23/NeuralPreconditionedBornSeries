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
    TEST_PATH = 'data_ill/test'
    
    # Path to the experiment directory (e.g., 'models/fno_solver_precond_02_10_...')
    # EXP_DIR = 'models/fno_solver_02_10_01_16_25' 
    # EXP_DIR = 'models/fno_solver_precond_02_19_18_32_38'
    EXP_DIR = 'models/fno_solver_precond_02_19_19_43_50'
    MODEL_NAME = "fno_solver_precond" if "precond" in EXP_DIR else "fno_solver"
    USE_PRECOND_RES = True if "precond" in MODEL_NAME else False
    
    MODEL_SAVE_PATH = os.path.join(EXP_DIR, f"{MODEL_NAME}.pt")
    SAVE_DIR = os.path.join(EXP_DIR, "infer_results")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Model architecture params
    RESOLUTION = (128, 128)
    DX = 1.0 / RESOLUTION[0]
    FEATURES = 32
    WAVENUMBER = [16, 16, 32] 
    PADDING = 8
    
    # Inference params
    INFER_SAMPLE_NUM = 10 # Number of samples to infer
    ITER_NUM = 200  # Max iterations for the solver
    TOLERANCE = 1e-6   # Convergence tolerance

    print(f"Using device: {DEVICE} | Mode: {'Precond' if USE_PRECOND_RES else 'Standard'}")

    # --- Load Data ---
    try:
        test_dataset = ConvectionDiffusionDataset(TEST_PATH, data_num=INFER_SAMPLE_NUM, device=DEVICE)
    except (FileNotFoundError, IndexError) as e:
        print(f"Error: Could not load samples: {e}")
        return

    # --- Load Model ---
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model file not found at '{MODEL_SAVE_PATH}'.")
        return

    model = FNO_solver(
        N=RESOLUTION, 
        dx=DX, 
        num_features=FEATURES, 
        wavenumber=WAVENUMBER, 
        padding=PADDING, 
        device=DEVICE,
        dtype=DTYPE,
        use_precond_res=USE_PRECOND_RES
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # --- Loop through Samples ---
    total_iterations = 0
    total_residual   = 0
    cal_num = 0
    for idx in range(INFER_SAMPLE_NUM):
        print(f"\n--- Inferring Sample {idx} ---")
        coeffs_sample = test_dataset[idx].unsqueeze(0) # Add batch dimension

        # --- Run Iterative Solver ---
        u_solved, final_residual, loss_his = model.solve(
            coeffs_sample, 
            iter_num=ITER_NUM, 
            tol=TOLERANCE
        )

        iters = len(loss_his) - 1
        if final_residual<TOLERANCE:
            cal_num +=1
            total_iterations += iters
            total_residual += final_residual
        print(f"Sample {idx} finished in {iters} iterations.")
        print(f"Final Relative Residual: {final_residual:.4e}")

        # --- Visualization & Saving ---
        coeffs_cpu = coeffs_sample.squeeze(0).cpu().numpy()
        u_solved_cpu = u_solved.squeeze(0).squeeze(0).cpu().detach().numpy()

        kappa = coeffs_cpu[0]
        vx, vy = coeffs_cpu[1], coeffs_cpu[2]
        f = coeffs_cpu[3]
        v_mag = np.sqrt(vx**2 + vy**2)

        # 1. Physics Parameters Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        im0 = axes[0].imshow(kappa, cmap='viridis', origin='lower')
        axes[0].set_title(f"Sample {idx}: Kappa")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(v_mag, cmap='viridis', origin='lower')
        axes[1].set_title("Velocity Mag")
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(f, cmap='Reds', origin='lower')
        axes[2].set_title("Source (f)")
        fig.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"sample_{idx}_params.png"))
        plt.close()

        # 2. Results Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im3 = axes[0].imshow(u_solved_cpu, cmap='jet', origin='lower')
        axes[0].set_title(f"Solution u (Sample {idx})")
        fig.colorbar(im3, ax=axes[0])

        axes[1].plot(loss_his, '-o', markersize=3)
        axes[1].set_yscale('log')
        axes[1].set_title("Convergence History")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Relative Residual")

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"sample_{idx}_results.png"))
        plt.close()

        # Save data
        torch.save({
            'u': u_solved.cpu(),
            'loss_his': loss_his,
            'final_residual': final_residual
        }, os.path.join(SAVE_DIR, f"sample_{idx}_data.pt"))

    avg_iterations = total_iterations / cal_num
    avg_residual   = total_residual /cal_num
    print(f"\nAverage Iterations for {cal_num} samples: {avg_iterations:.2f}")
    print(f"\nAverage Residual for {cal_num} samples: {avg_residual:.4e}")
    print(f"\nAll inference results saved to: {SAVE_DIR}")

if __name__ == '__main__':
    run_inference()