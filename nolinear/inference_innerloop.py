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
from newton import FNO_solver, Newton_FDM_op, Newton_FDM_precond_op

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
        model.use_precond_res = False
        model.use_l2_loss = False
        if "precond" in os.path.basename(args.model_path).lower():
            model.use_precond_res = True
        if "l_2" in os.path.basename(args.model_path).lower():
            model.use_l2_loss = True
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
    u_list, du_list =  torch.load('./data/newton_method_iterations_5.pt', weights_only=False)
    # u_list, du_list, solution = torch.load('./data/newton_method_iterations_7_solution.pt', weights_only=False)
    
    # 我们需要 u_list 作为迭代的初始点，solution 作为最终比较的基准
    
    # x_all = torch.cat([u_list_1, u_list_2], dim=0).to(device).float()
    # y_all = torch.cat([du_list_1, du_list_2], dim=0).to(device).float()
    x_all = u_list.to(device).float()
    y_all = du_list.to(device).float()
    # 使用与训练时相同的测试集索引
    test_indices = range(5000, 5100)
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
            total_iterations += len(iter_his) - 1
            if final_res_norm < args.tol:
                solved_count += 1
                print(f"solve done! relative residual: {final_res_norm:.3e} ")

                total_rel_pde_error += final_res_norm
                # import matplotlib.pyplot as plt
                # plt.imshow(u_initial[0][0].cpu())
                # plt.colorbar()
                # plt.savefig('test.png')
                # exit()               
            # else:
            #     import matplotlib.pyplot as plt
            #     plt.subplot(121)
            #     plt.imshow(u_initial[0][0].cpu())
            #     plt.colorbar()
            #     plt.subplot(122)
            #     plt.imshow(du_true_solution[0][0].cpu())
            #     plt.colorbar()
            #     plt.savefig('test.png')
            #     exit()
            # 更新进度条信息
            # pbar.set_postfix({
            #     'batch_avg_error': f"{rel_l2_error.item()/u_initial.size(0):.4e}",
            #     'batch_avg_iters': f"{iters/u_initial.size(0):.1f}"
            # })

    # # --- 结果统计和报告 ---
    
    num_test_samples = len(x_test)
    if solved_count != 0:
        avg_rel_pde_error = total_rel_pde_error / solved_count
    else:
        avg_rel_pde_error = 0
    if num_test_samples != 0:
        avg_iterations = total_iterations / num_test_samples
    else:
        avg_iterations = 0
    convergence_rate = (solved_count / num_test_samples) * 100
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(du_true_solution[0][0].cpu())
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(du_final[0][0].cpu())
    plt.colorbar()
    plt.savefig("test.png")
    if "precond" in os.path.basename(args.model_path).lower():
        torch.save(iter_his, "iter_his_fno_precond_l2.pt")
        torch.save(du_final, "du_fno_precond_l2.pt")
        torch.save(u_initial, "u_initial_fno_precond_l2.pt")
    else:
        torch.save(iter_his, "iter_his_fno.pt")
        torch.save(du_final, "du_fno.pt")
        torch.save(u_initial, "u_initial_fno.pt")
    logging.info("---------- Inference Finished ----------")
    logging.info(f"Number of test samples: {num_test_samples}")
    logging.info(f"Average relative residual: {avg_rel_pde_error:.4e}")
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
        default='./data/newton_method_iterations_5.pt',
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
    

    main(args)