import argparse
import json
import os

import torch
import matplotlib.pyplot as plt
import numpy as np

from dataloader import ConvectionDiffusionDataset
from baseline import FNO_solver


def run_inference(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float64

    config_path = os.path.join(args.exp_dir, 'args.json')
    if not os.path.exists(config_path):
        print(f"Error: args.json not found at '{config_path}'.")
        return
    with open(config_path) as f:
        cfg = json.load(f)

    MODEL_NAME       = cfg['model_name']
    USE_PRECOND_RES  = cfg['use_precond_res']
    USE_PRECOND_LOSS = cfg['use_precond_loss']
    RESOLUTION       = tuple(cfg['resolution'])
    DX               = cfg['dx']
    FEATURES         = cfg['features']
    WAVENUMBER       = cfg['wavenumber']
    PADDING          = cfg['padding']

    MODEL_SAVE_PATH = os.path.join(args.exp_dir, f"{MODEL_NAME}.pt")
    SAVE_DIR = os.path.join(args.exp_dir, "infer_results")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Using device: {DEVICE} | Precond Res: {USE_PRECOND_RES} | Precond Loss: {USE_PRECOND_LOSS}")

    try:
        test_dataset = ConvectionDiffusionDataset(args.test_path, data_num=args.infer_sample_num, device=DEVICE)
    except (FileNotFoundError, IndexError) as e:
        print(f"Error: Could not load samples: {e}")
        return

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
        use_precond_res=USE_PRECOND_RES,
        use_precond_loss=USE_PRECOND_LOSS,
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    total_iterations = 0
    total_residual   = 0
    cal_num = 0
    for idx in range(args.infer_sample_num):
        print(f"\n--- Inferring Sample {idx} ---")
        coeffs_sample = test_dataset[idx].unsqueeze(0)

        u_solved, final_residual, loss_his = model.solve(
            coeffs_sample,
            iter_num=args.iter_num,
            tol=args.tolerance,
        )

        iters = len(loss_his) - 1
        if final_residual < args.tolerance:
            cal_num += 1
            total_iterations += iters
            total_residual += final_residual
        print(f"Sample {idx} finished in {iters} iterations.")
        print(f"Final Relative Residual: {final_residual:.4e}")

        coeffs_cpu = coeffs_sample.squeeze(0).cpu().numpy()
        u_solved_cpu = u_solved.squeeze(0).squeeze(0).cpu().detach().numpy()

        kappa = coeffs_cpu[0]
        vx, vy = coeffs_cpu[1], coeffs_cpu[2]
        f = coeffs_cpu[3]
        v_mag = np.sqrt(vx**2 + vy**2)

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

        torch.save({
            'u': u_solved.cpu(),
            'loss_his': loss_his,
            'final_residual': final_residual,
        }, os.path.join(SAVE_DIR, f"sample_{idx}_data.pt"))

    if cal_num > 0:
        print(f"\nAverage Iterations for {cal_num} samples: {total_iterations / cal_num:.2f}")
        print(f"\nAverage Residual for {cal_num} samples: {total_residual / cal_num:.4e}")
    print(f"\nAll inference results saved to: {SAVE_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='data_ill/test')
    parser.add_argument('--exp_dir', default='models/fno_solver_04_19_23_18_02')
    parser.add_argument('--infer_sample_num', type=int, default=10)
    parser.add_argument('--iter_num', type=int, default=600)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    args = parser.parse_args()

    run_inference(args)
