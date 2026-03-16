import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from utilities3 import LpLoss
import argparse
import logging
import os

# Import the class you want to test from your script
# Ensure newton.py is in the same directory or in the python path
from newton import Newton_FDM_precond_op

def setup_logging(log_path):
    """Sets up logging to file and console."""
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler())

def solve_with_preconditioner(precond_op, u_current, max_iter=50, tol=1e-6):
    """
    Solves the Newton system using preconditioned Richardson iteration.
    The system is J(u)du = -R(u).
    Iteration: du_{k+1} = du_k + P * (-R(u) - J(u)du_k)
    """
    device = u_current.device
    du = torch.zeros_like(u_current)
    
    # Target for the linear system: b = -R(u)
    b_newton = precond_op.getGrad(u_current)
    
    iterations_taken = 0
    final_residual_norm = float('inf')

    for i in range(max_iter):
        # Calculate the residual of the linear system: r_k = b - A*du_k
        residual = b_newton - precond_op.laplace_minus_u(u_current, du)
        
        # --- CHANGE 1: Correct norm calculation for batches ---
        # Calculate the norm for each sample in the batch, then average.
        residual_norm = torch.norm(residual)
        final_residual_norm = (residual_norm/torch.norm(b_newton)).item()
        
        # Optional: print progress for debugging
        print(f"Iter: {i+1}, residual norm: {final_residual_norm:.4e}")
        
        if final_residual_norm < tol:
            break
            
        # --- CHANGE 2: Apply preconditioner to a new variable ---
        # Apply the preconditioner: P*r_k
        preconditioned_residual = precond_op.apply_preconditioner(residual, u_current=u_current)
        
        # --- CHANGE 3: Use a full step, remove the 0.1 damping factor ---
        # Update the solution: du_{k+1} = du_k + P*r_k
        du = du + preconditioned_residual
        
        iterations_taken = i + 1
        
    return du, final_residual_norm, iterations_taken


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log_path = os.path.join(args.log_dir, 'inference_dst_log.txt')
    setup_logging(log_path)
    logging.info(f"Starting DST Preconditioner Inference with arguments: {args}")
    logging.info(f"Using device: {device}")

    precond_op = Newton_FDM_precond_op(N=63, s=1600).to(device)

    logging.info("Loading test dataset...")
    try:
        u_list, du_list, _ = torch.load(args.data_path, weights_only=False)
        x_all = u_list.to(device).float()
        y_all = du_list.to(device).float()

        test_indices = range(5490, 5500) # Using a larger, more representative test set
        x_test = x_all[test_indices, ...]
        y_test_true_du = y_all[test_indices, ...]

        test_loader = DataLoader(TensorDataset(x_test.contiguous(), y_test_true_du.contiguous()), 
                                 batch_size=args.batch_size, shuffle=False)
        logging.info(f"Test data loaded. Number of samples: {len(x_test)}")
    except FileNotFoundError:
        logging.error(f"Data file could not be found at {args.data_path}")
        return
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        return

    l2loss_rel = LpLoss(size_average=True, relative=True)
    total_rel_l2_error = 0.0
    total_iterations = 0.0
    solved_count = 0
    
    with torch.no_grad():
        # pbar = tqdm(test_loader, desc="Solving with DST Preconditioner")
        for u_initial, du_true in test_loader:
            du_final, final_res_norm, iters = solve_with_preconditioner(
                precond_op,
                u_initial, 
                max_iter=args.max_iter, 
                tol=args.tol
            )
            
            if final_res_norm < args.tol:
                solved_count += u_initial.size(0)
                rel_l2_error = l2loss_rel(du_final, du_true)
                total_rel_l2_error += rel_l2_error.item() * u_initial.size(0)
            
            total_iterations += iters * u_initial.size(0)

    num_test_samples = len(x_test)
    avg_rel_l2_error = total_rel_l2_error / solved_count if solved_count > 0 else float('inf')
    avg_iterations = total_iterations / num_test_samples if num_test_samples > 0 else 0
    convergence_rate = (solved_count / num_test_samples) * 100 if num_test_samples > 0 else 0

    logging.info("---------- Inference Finished ----------")
    logging.info(f"Number of test samples: {num_test_samples}")
    logging.info(f"Average relative L2 error vs true du (for converged cases): {avg_rel_l2_error:.4e}")
    logging.info(f"Average number of iterations per sample: {avg_iterations:.2f}")
    logging.info(f"Convergence rate (residual < {args.tol:.1e}): {convergence_rate:.2f}%")
    logging.info("------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for DST-based iterative solver.")
    
    parser.add_argument("--data_path", type=str, 
        default='/home/wangjt/work/Newton-Informed-Neural-Operator/data/newton_method_iterations_7_solution.pt',
        help="Path to the test data file.")
    
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save log files.")
    
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for inference.")
    
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations for the solver.")
    
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for the residual to determine convergence.")
    
    args = parser.parse_args()
    main(args)