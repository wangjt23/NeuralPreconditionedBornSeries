import numpy as np
import torch
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

# ----------------------------
# 1) Dirichlet 拉普拉斯 (矩阵版)
# ----------------------------
def create_laplacian_matrix_dirichlet(N):
    """
    为内部点构建5点差分离散拉普拉斯(未除 h^2) + Dirichlet (u=0) 边界。
    矩阵大小为 (N*N) x (N*N)。
    """
    main_diag = -4.0 * np.ones(N * N)
    off_diag1 = np.ones(N * N - 1)
    off_diagN = np.ones(N * N - N)

    # 在行末断开对角线，避免 wrap-around
    for i in range(1, N):
        off_diag1[i * N - 1] = 0

    diagonals = [main_diag, off_diag1, off_diag1, off_diagN, off_diagN]
    offsets = [0, -1, 1, -N, N]
    L = diags(diagonals, offsets, shape=(N**2, N**2), format="csr")
    return L

def build_neg_laplacian_operator_dirichlet(N, h):
    """返回 (-Δ) 的稀疏矩阵（已除以 h^2）。"""
    L = create_laplacian_matrix_dirichlet(N)  # h^2 * Δ
    negDelta = -(L) / (h**2)                  # => -Δ
    return negDelta.tocsr()

# ----------------------------
# 2) 牛顿法，求解 -Δu - u² = f(x,y)
# ----------------------------
def newton_method_single_field(N, h, u0, f_source_vec, max_iter=15, tol=1e-7):
    """
    求解 F(u) = (-Δ)u - u² - f = 0
    返回: u_final, u_traj, du_traj
    """
    # 预组装 (-Δ) 算子
    negDelta = build_neg_laplacian_operator_dirichlet(N, h)
    u = u0.copy()

    u_list, du_list = [], []

    for k in range(max_iter):
        u_vec = u.reshape(-1)

        # 1. 计算残差 F(u) = (-Δ)u - u² - f
        # 注意：这里的 f 对应原方程的 -s*sin*sin，所以是减法
        Fu = negDelta @ u_vec - (u_vec**2) - f_source_vec

        # 2. 组装雅可比矩阵 J(u) = (-Δ) - 2*diag(u)
        J = negDelta + diags(-2.0 * u_vec, 0, shape=(N*N, N*N), format="csr")

        # 3. 求解线性系统 J δu = -F
        delta = spsolve(J, -Fu)
        du = delta.reshape(N, N)

        # 记录轨迹
        u_list.append(u.copy())
        du_list.append(du.copy())

        # 更新解
        u += du

        # 检查收敛
        norm_du = np.linalg.norm(du, ord=np.inf)
        print(f"Iter {k+1}/{max_iter}: ||δu||_inf = {norm_du:.2e}")
        if norm_du < tol:
            print(f"Converged after {k+1} iterations.")
            break
    
    if k + 1 == max_iter and norm_du >= tol:
        print("Warning: Newton's method did not converge within max_iter.")

    return u, u_list, du_list

# ----------------------------
# 3) 随机正弦级数扰动 (满足Dirichlet边界)
# ----------------------------
def generate_initial_guess_dirichlet(s, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # 使用 (0, 1) 区间内部的点
    x = np.linspace(0, 1, s + 2)[1:-1]
    y = np.linspace(0, 1, s + 2)[1:-1]
    X, Y = np.meshgrid(x, y, indexing='ij')

    u = np.zeros((s, s), dtype=np.float64)
    # 随机系数
    coeffs = np.random.randn(5, 5) * 0.1 
    
    # 叠加几个低频正弦模式
    for i in range(1, 5):
        for j in range(1, 5):
            u += coeffs[i, j] * np.sin(np.pi * i * X) * np.sin(np.pi * j * Y)
    return u

# ----------------------------
# 4) 主执行函数
# ----------------------------
# --- 配置 ---
N = 63             # 内部网格点数
h = 1.0 / (N + 1)  # 网格尺寸
s = 100.0          # 源项系数
noise_scale = 4.0  # 初始扰动大小
u_ref_list, du_ref_list,solution = torch.load('/home/wangjt/work/Newton-Informed-Neural-Operator/data/newton_method_iterations_7_solution.pt', weights_only=False)
# --- 构建源项 f(x,y) ---
# f(x,y) = -s * sin(πx) * sin(πy)
# 注意：我们的 F(u) 定义为 ... - f，所以这里 f 是 PDE 的右侧项
x = np.linspace(0, 1, N + 2)[1:-1]
y = np.linspace(0, 1, N + 2)[1:-1]
X, Y = np.meshgrid(x, y, indexing='ij')
f_source = -s * np.sin(np.pi * X) * np.sin(np.pi * Y)
f_source_vec = f_source.reshape(-1)

# --- 运行牛顿法 ---
print(f"--- Solving for s={s} ---")
# 初始猜测 u0 = 0，也可以加扰动
# u0 = np.zeros((N, N), dtype=np.float64)
u0 = solution + noise_scale * generate_initial_guess_dirichlet(N, seed=42)

u_final, u_traj, du_traj = newton_method_single_field(
    N, h, u0, f_source_vec, max_iter=7, tol=1e-7
)

