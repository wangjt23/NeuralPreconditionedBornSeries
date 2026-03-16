import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import argparse
import logging
import os
import random
import matplotlib.pyplot as plt

# Import the necessary classes from the provided source code.
# This assumes the classes (FNO_solver, Newton_FDM_op, etc.) are in a file named 'newton.py'
# in the same directory or in the Python path.

from newton import FNO_solver, Newton_FDM_op, Newton_FDM_precond_op

# --- 日志记录设置 (Logging Setup) ---
def setup_logging(log_path):
    """Sets up logging to file and console."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_path,
                        filemode='w')
    # Add a handler to also print log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(console_handler)

def solve_pde_with_newton(model, u_initial, max_newton_iter, newton_tol,
                          linear_solver_max_iter, linear_solver_tol, save_dir=None):
    u_k = u_initial.clone()
    pde_op = model.A_op
    newton_history = []

    def save_iter_image(u, iter_idx):
        if save_dir is None:
            return
        save_path = os.path.join(save_dir, f"iter_{iter_idx}.png")
        plt.figure(figsize=(4, 4))
        plt.imshow((u[0, 0].detach().cpu().numpy()))
        plt.title(f"iter {iter_idx}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    for i in range(max_newton_iter):
        # 1. Calculate the PDE residual (negative of the gradient)
        # R(u) = laplace(u) - u^2 - f
        # We want to solve J(u) * du = -R(u)
        # model.A_op.getGrad(u) computes -R(u), so it's our 'b' for the linear system.
        pde_residual = -pde_op.getGrad(u_k)
        pde_res_norm = torch.norm(pde_residual).item() / torch.norm(pde_op.rhs).item()
        newton_history.append(pde_res_norm)
        
        logging.info(f"Newton Iter {i+1}/{max_newton_iter} | PDE Residual Norm: {pde_res_norm:.3e}")

        # 2. Check for convergence of the outer Newton loop
        if pde_res_norm < newton_tol:
            logging.info(f"Newton method converged after {i} iterations.")
            return u_k, pde_res_norm, i, newton_history

        # 3. Solve the linear system J(u_k) * du = -R(u_k) using the neural solver
        # The 'solve' method is an iterative linear solver (e.g., Richardson)
        # that uses the FNO as a preconditioner.
        du, final_linear_res_norm, linear_iter_hist = model.solve(
            u_k,
            iter_num=linear_solver_max_iter,
            tol=linear_solver_tol
        )
        
        logging.info(f"  -> Linear solver finished in {len(linear_iter_hist)-1} iterations. Final relative residual: {final_linear_res_norm:.3e}")

        # 4. Update the solution
        u_k = u_k + du
        save_iter_image(u_k, i + 1)

    # If the loop finishes, max iterations were reached
    final_pde_res_norm = torch.norm(-pde_op.getGrad(u_k)).item()/ torch.norm(pde_op.rhs).item()
    newton_history.append(final_pde_res_norm)
    logging.warning(f"Newton method did not converge within {max_newton_iter} iterations. Final residual: {final_pde_res_norm:.3e}")
    
    return u_k, final_pde_res_norm, max_newton_iter, newton_history


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Logging Setup ---
    log_dir = os.path.dirname(args.model_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, 'inference_newton_log.txt')
    setup_logging(log_path)
    logging.info(f"Starting Newton inference with arguments: {args}")
    logging.info(f"Using device: {device}")

    # --- Load Model ---
    logging.info(f"Loading model from {args.model_path}")
    try:
        model = torch.load(args.model_path, map_location=device, weights_only=False)
        model.use_precond_res = False
        if hasattr(model, "use_precond_res") and "precond" in os.path.basename(args.model_path).lower():
            model.use_precond_res = True
        model.eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return


    # --- Load Data ---
    # The data file should contain the initial guess `u_initial` and the true solution.
    logging.info(f"Loading data from {args.data_path}")
    # u_list_2 is u_initial, solution is the ground truth final solution
    u_initial_0, delta_u_list = torch.load('./data/newton_method_iterations_5.pt', weights_only=False)
    # print(u_list.shape)
    # import matplotlib.pyplot as plt
    # plt.imshow(u_list[10000][0])
    # plt.colorbar()
    # plt.savefig('test.png')
    # exit()
    u_initial_1, _, solution_all = torch.load(args.data_path, weights_only=False)
    # print(u_initial_all.shape)
    # print(solution_all.shape)
    # exit()
    u_initial_all = torch.cat([u_initial_0, u_initial_1],dim=0)
    u_initial_all = u_initial_0
    # For this script, we solve one instance. Let's pick the first one.
    u_initial = u_initial_all[args.sample_index:args.sample_index+1].to(device).float()
    # u_true_solution = torch.from_numpy(solution_all).unsqueeze(0).unsqueeze(0).to(device).float()
    def resize_to_64(tensor):
        return torch.nn.functional.interpolate(tensor, size=(64, 64), mode='bilinear', align_corners=False)
    u_initial  = resize_to_64(u_initial)
    # u_true_solution  = resize_to_64(u_true_solution)
    logging.info(f"Solving for sample index: {args.sample_index}")
    logging.info(f"Initial guess shape: {u_initial.shape}")

    # --- Solve PDE with Newton's Method ---
    save_dir = os.path.join(log_dir, f"newton_sample_{args.sample_index}")
    os.makedirs(save_dir, exist_ok=True)
    u_final, final_pde_res, newton_iters, newton_history = solve_pde_with_newton(
        model, u_initial, 
        args.max_newton_iter, args.newton_tol,
        args.linear_solver_max_iter, args.linear_solver_tol,
        save_dir=save_dir
    )

    # --- Results and Visualization ---
    # l2_error_vs_true = torch.norm(u_final - u_true_solution) / torch.norm(u_true_solution)

    logging.info("---------- Newton Inference Finished ----------")
    logging.info(f"Newton iterations taken: {newton_iters}")
    logging.info(f"Final PDE residual norm: {final_pde_res:.4e}")
    # logging.info(f"Relative L2 error vs. true solution: {l2_error_vs_true.item():.4e}")
    logging.info("---------------------------------------------")

    # --- Plotting ---
    # Plot Newton's method convergence history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.semilogy(range( len(newton_history) ), newton_history, 'o-')
    plt.title("Newton's Method Convergence")
    plt.xlabel("Newton Iteration")
    plt.ylabel("PDE Residual Norm (log scale)")
    plt.grid(True)
    
    # Plot final solution vs. true solution
    # vmin = min(u_true_solution.min(), u_final.min())
    # vmax = max(u_true_solution.max(), u_final.max())
    # plt.subplot(1, 3, 1)
    # plt.imshow(u_initial[0, 0].cpu().numpy())
    # plt.title("U_initial")
    # plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow((u_final)[0, 0].cpu().numpy())
    plt.title(r"$u_{final}$")
    plt.colorbar()


    plt.subplot(1, 3, 3)
    plt.imshow((u_final-u_initial)[0, 0].cpu().numpy())
    plt.title(r"$u_{final}$ - $u_{init}$")
    plt.colorbar()
    plt.tight_layout()
    save_path = os.path.join(log_dir, f"newton_result_sample_{args.sample_index}.png")
    plt.savefig(save_path)
    logging.info(f"Saved result plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for solving a non-linear PDE using Newton's method with an FNO-based solver.")
    
    # --- Paths ---
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained FNO_solver model (.pt file).")
    parser.add_argument("--data_path", type=str, default='./data/newton_method_iterations_7_solution.pt', help="Path to the data file.")
    
    # --- Data Selection ---
    parser.add_argument("--sample_index", type=int, default=0, help="Index of the sample from the dataset to solve.")

    # --- Newton Method Parameters (Outer Loop) ---
    parser.add_argument("--max_newton_iter", type=int, default=20, help="Maximum number of Newton iterations.")
    parser.add_argument("--newton_tol", type=float, default=1e-5, help="Convergence tolerance for the PDE residual in Newton's method.")
    
    # --- Linear Solver Parameters (Inner Loop) ---
    parser.add_argument("--linear_solver_max_iter", type=int, default=50, help="Maximum number of iterations for the inner linear solver.")
    parser.add_argument("--linear_solver_tol", type=float, default=1e-4, help="Relative tolerance for the inner linear solver.")
    
    args = parser.parse_args()
    main(args)