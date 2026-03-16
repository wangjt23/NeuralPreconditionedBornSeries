import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from utilities3 import LpLoss, getSavePath
import argparse
import logging
import os
import random
# 从 newton.py 导入所需的类
# 确保 newton.py 在您的 Python 路径中
from newton import FNO_solver, Newton_FDM_op, Newton_FDM_precond_op,MG_solver

# --- 日志记录设置 ---
def setup_logging(log_path):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler())

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 设置日志 ---
    log_dir = os.path.dirname(args.model_path)
    log_path = os.path.join(log_dir, 'inference_log.txt')
    setup_logging(log_path)
    logging.info(f"Starting inference with arguments: {args}")
    logging.info(f"Using device: {device}")

    # --- 加载模型 ---
    logging.info(f"Loading model from {args.model_path}")
    try:
        model = torch.load(args.model_path, map_location=device, weights_only=False)
        model.eval()
        logging.info("Model loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Model file not found at {args.model_path}")
        return
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")
        return

    # --- 加载测试数据 ---
    # 根据 newton.py 中的数据加载逻辑
    logging.info("Loading test dataset...")

    # 这个数据集包含初始猜测 u_current (x) 和对应的真实解 (solution)
    u_list_2, du_list_2, solution = torch.load(args.data_path, weights_only=False)
    
    # 我们需要 u_list 作为迭代的初始点，solution 作为最终比较的基准
    x_all = u_list_2.to(device).float()
    y_all = du_list_2.to(device).float()

    # 使用与训练时相同的测试集索引
    test_indices = range(5024, 5025)
    # random.seed(42)
    # test_indices = random.sample(list(test_indices), 10)
    x_test = x_all[test_indices, ...]
    y_test_solution = y_all[test_indices, ...] # 真实的最终解
    def resize_to_64(tensor):
        return torch.nn.functional.interpolate(tensor, size=(64, 64), mode='bilinear', align_corners=False)

    # x_train = resize_to_64(x_train)
    # y_train = resize_to_64(y_train.unsqueeze(1))
    x_test  = resize_to_64(x_test)
    # print(y_test_solution.shape)
    y_test_solution  = resize_to_64(y_test_solution)
    test_loader = DataLoader(TensorDataset(x_test.contiguous(), y_test_solution.contiguous()), 
                                batch_size=args.batch_size, shuffle=False)
    logging.info(f"Test data loaded. Number of samples: {len(x_test)}")


    # --- 推理和求解 ---
    l2loss_rel = LpLoss(size_average=False, relative=True)
    total_rel_l2_error = 0.0
    total_rel_pde_error = 0.0
    total_iterations = 0
    solved_count = 0
    
    # 获取模型内部的 PDE 操作符，用于计算最终残差
    # pde_op = Newton_FDM_op(N=model.A_op.N, s=model.A_op.s)

    with torch.no_grad():
        # 使用tqdm显示进度条
        # pbar = tqdm(test_loader, desc="Solving PDEs for test set")
        for i, (u_initial, du_true_solution) in enumerate(test_loader):
            # u_initial 是牛顿法的初始猜测 u_0
            # u_true_solution 是PDE的真实解
            
            # 使用模型内置的 solve 方法进行迭代求解
            # 该方法返回最终的 u_k = u_0 + du_final
            print(f"---------------Solving {i} data---------------------")
            du_final, final_res_norm, iter_his = model.solve(
                u_initial, 
                iter_num=args.max_iter, 
                tol=args.tol
            )
            
            # 检查是否收敛
            if final_res_norm < args.tol:
                solved_count += 1
                rel_l2_error = (torch.norm(du_final-du_true_solution)/torch.norm(du_true_solution)).item()
                print(f"solve done! pde error: {final_res_norm:.3e}, L2 error: {rel_l2_error:.3e}")
                # 计算最终解 u_final 与真实解 u_true_solution 之间的相对L2误差

                total_rel_l2_error += rel_l2_error
                total_rel_pde_error+=final_res_norm
                total_iterations += len(iter_his)-1

            # 更新进度条信息
            # pbar.set_postfix({
            #     'batch_avg_error': f"{rel_l2_error.item()/u_initial.size(0):.4e}",
            #     'batch_avg_iters': f"{iters/u_initial.size(0):.1f}"
            # })

    # # --- 结果统计和报告 ---
    num_test_samples = len(x_test)
    avg_rel_l2_error = total_rel_l2_error / solved_count
    avg_rel_pde_error = total_rel_pde_error / solved_count
    avg_iterations = total_iterations / solved_count
    convergence_rate = (solved_count / num_test_samples) * 100
    import matplotlib.pyplot as plt
    # 假设只处理第一个样本 (i=0)，因为 test_loader 只有一个样本
    u_init_plot = u_initial[0, 0].cpu().numpy()      # shape: (64, 64)
    du_final_plot = du_final[0, 0].cpu().numpy()     # shape: (64, 64)

    # 获取迭代历史中的残差范数列表（假设 iter_his 是一个列表，如 [res_0, res_1, ..., res_k]）
    # 注意：在 model.solve 中，iter_his 应该是每步的残差值列表（相对范数）

    iter_steps = list(range(len(iter_his)))
    res_norms = np.array(iter_his)

    # 创建子图：一行三列
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 第一列：初始猜测 u_initial
    im1 = axes[0].imshow(u_init_plot, cmap='jet', extent=[0, 1, 0, 1], origin='lower')
    axes[0].set_title(r'$u_{0}$ (Initial Guess)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # 第二列：最终预测的 du_final
    im2 = axes[1].imshow(du_final_plot, cmap='jet', extent=[0, 1, 0, 1], origin='lower')
    axes[1].set_title(r'$\delta u$ (Predicted Correction)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # 第三列：迭代残差下降曲线（对数坐标）
    iter_his_fno = torch.load('/home/wangjt/work/precond_loss/Newton/iter_his_fno.pt')
    axes[2].plot(res_norms, 'o-', linewidth=1.5, markersize=4,label='LS-FNO')
    axes[2].plot(iter_his_fno, '^-', linewidth=1.5, markersize=4, label='FNO')
    axes[2].set_yscale('log')  # 对数尺度，便于观察收敛行为
    axes[2].set_title(r'Convergence History')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Relative Residual Norm')
    axes[2].grid(True, which="both", linestyle='--', alpha=0.6)
    # axes[2].axhline(y=args.tol, color='blue', linestyle='--', label=f'Tolerance ({args.tol:.1e})')
    axes[2].legend()

    # 调整布局，避免重叠
    plt.tight_layout()
    plt.savefig("inference_results.png", dpi=300, bbox_inches='tight')
    logging.info("---------- Inference Finished ----------")
    logging.info(f"Number of test samples: {num_test_samples}")
    logging.info(f"Average relative PDE error: {avg_rel_pde_error:.4e}")
    logging.info(f"Average relative L2 error vs true du: {avg_rel_l2_error:.4e}")
    logging.info(f"Average number of iterations to converge: {avg_iterations:.2f}")
    logging.info(f"Convergence rate (residual < {args.tol:.1e}): {convergence_rate:.2f}%")
    logging.info("------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for FNO_solver model.")
    parser.add_argument(
        "--model_path", type=str, required=True, 
        help="Path to the trained FNO_solver model (.pt file)."
    )
    parser.add_argument(
        "--data_path", type=str, 
        default='/home/wangjt/work/Newton-Informed-Neural-Operator/data/newton_method_iterations_7_solution.pt',
        help="Path to the test data file containing initial states and true solutions."
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, 
        help="Batch size for inference."
    )
    parser.add_argument(
        "--max_iter", type=int, default=50, 
        help="Maximum number of iterations for the solver."
    )
    parser.add_argument(
        "--tol", type=float, default=1e-5, 
        help="Tolerance for the residual to determine convergence."
    )
    
    args = parser.parse_args()
    
    # 在 FNO_solver 类中添加 solve 方法
    def fno_solver_solve(self, u_current, iter_num=10, tol=1e-6):
        with torch.no_grad():
            du = torch.zeros_like(u_current)
            b = self.A_op.getGrad(u_current)
            norm_b = torch.norm(b)
            iterations_taken = 0
            final_res_norm = float('inf')

            for i in range(iter_num):
                # self.forward 更新 du: du_{k+1} = du_k + P*(b - A*du_k)
                # 并返回新的残差 res = b - A*du_{k+1}
                du, res = self.forward(u_current, du)
                
                # 计算残差的范数
                # .mean() 用于处理 batch, 如果 batch_size=1 则无影响
                res_norm = torch.norm(res)/norm_b
                final_res_norm = res_norm.item()
                iterations_taken = i + 1

                # 打印每一次迭代信息 (可选)
                print(f"Iter: {i+1}, error: {final_res_norm}")

                if final_res_norm < tol:
                    break
            
            # 返回最终解，最终残差范数，以及所用迭代次数
            return du, final_res_norm, iterations_taken

    # 动态地将 solve 方法添加到 FNO_solver 类中
    MG_solver.solve = fno_solver_solve

    main(args)