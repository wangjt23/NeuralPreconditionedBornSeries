import torch
import numpy as np

def _sorted_real_eig(block, tol=1e-8):
    vals, vecs = torch.linalg.eig(block.to(torch.float64))
    max_imag = torch.max(torch.abs(torch.imag(vals))).item()
    if max_imag > tol:
        raise ValueError(f'Complex eigenvalues above tolerance: {max_imag:.3e}')
    vals = torch.real(vals)
    vecs = torch.real(vecs)
    order = torch.argsort(vals, dim=-1)
    vals = torch.gather(vals, -1, order)
    vecs = torch.gather(vecs, -1, order.unsqueeze(-2).expand(*vecs.shape[:-1], order.shape[-1]))
    max_vals = torch.clamp(torch.max(torch.abs(vecs), dim=-2, keepdim=True)[0], min=1e-30)
    vecs = vecs / max_vals
    first_nonzero = torch.argmax((torch.abs(vecs) > tol).int(), dim=-2).unsqueeze(-2)
    signs = torch.gather(torch.sign(vecs), -2, first_nonzero)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return vals, vecs / signs


def VDXY(Sigma_T, Sigma_a, Varepsilon, ct, st, omega, Kappa, M, device='cpu'):
    """
    参数:
        Sigma_T, Sigma_a, Varepsilon: [I0,J0]张量
         4M: 离散速度方向个数
        ct, st, omega: DOM得到的二维速度方向以及权重，[4M]张量
        Kappa：离散化的核函数
        I0, J0: 网格数
        device: 计算设备
        
    返回:
        VX, DX: 形状为 [I0,J0,4M,4M], [I0,J0,4M]
        VY, DY: 形状为 [I0,J0,4M,4M], [I0,J0,4M]
    """
    # 预计算公共部分
    ratio = (1 - Varepsilon**2 * Sigma_a / Sigma_T)  # [I0,J0]
    KW = Kappa @ torch.diag(omega)  # [4M,4M]
    eye = torch.eye(4*M, device=device)
    inv_ct = torch.diag(1/ct.squeeze(-1))  # [4M,4M]
    inv_st = torch.diag(1/st.squeeze(-1))  # [4M,4M]
    
    # 向量化计算 blockx [I0,J0,4M,4M]
    base = ratio.unsqueeze(-1).unsqueeze(-1) * KW - eye
    blockx = inv_ct @ base
    blocky = inv_st @ base
    
    # 批量特征分解 (需转为float64)
    vx_vals, vex = _sorted_real_eig(blockx)
    vy_vals, vey = _sorted_real_eig(blocky)
    
    VX = vex.permute(2,3,0,1).to(torch.float32)
    DX = vx_vals.permute(2,0,1).to(torch.float32)
    VY = vey.permute(2,3,0,1).to(torch.float32)
    DY = vy_vals.permute(2,0,1).to(torch.float32)
    return VX, DX, VY, DY



def FSM(VX, DX, VY, DY, xl, xr, yl, yr, Sigma_T, Varepsilon, M, I0, J0, device='cpu'):
    """
    参数:
        VX, DX, VY, DY: VDXY函数的输出, 形状分别为 [4M,4M,I0,J0], [4M,I0,J0], [4M,4M,I0,J0], [4M,I0,J0]
        xl, xr, yl, yr: 计算域边界
        Sigma_T: [I0,J0] 张量
        Varepsilon: [I0,J0] 张量
        4M: 离散速度方向个数
        I0, J0: 网格数
        device: 计算设备
        
    返回:
        fsml_full, fsmr_full, fsmb_full, fsmt_full, fsmc_full: 形状均为 [4M,8M,I0,J0]
    """
    # 预计算网格步长和坐标
    hx, hy = (xr - xl) / I0, (yr - yl) / J0
    VX_permuted, DX_permuted, VY_permuted, DY_permuted = VX.permute(2,3,0,1), DX.permute(1,2,0), VY.permute(2,3,0,1), DY.permute(1,2,0)
    # 生成所有网格点坐标 [I0,J0]
    i_indices = torch.arange(I0, device=device)
    j_indices = torch.arange(J0, device=device)
    
    # 计算所有左/右/中心点坐标 [I0,J0]
    x_l = (i_indices * hx + xl).view(I0, 1, 1)
    x_r = ((i_indices + 1) * hx + xl).view(I0, 1, 1)
    x_c = ((i_indices + 0.5) * hx + xl).view(I0, 1, 1)
    
    y_l = (j_indices * hy + yl).view(J0, 1)
    y_r = ((j_indices + 1) * hy + yl).view(J0, 1)
    y_c = ((j_indices + 0.5) * hy + yl).view(J0, 1)
    
    # 扩展维度用于广播计算 [I0,J0,4M]
    Sigma_T_expand = Sigma_T.unsqueeze(-1)  # [I0,J0,1]
    Varepsilon_expand = Varepsilon.unsqueeze(-1)  # [I0,J0,1]
    
    # 初始化输出 [4M,8M,I0,J0]
    fsml_full = torch.zeros(I0, J0, 4*M, 8*M, device=device)
    fsmr_full = torch.zeros(I0, J0, 4*M, 8*M, device=device)
    fsmb_full = torch.zeros(I0, J0, 4*M, 8*M, device=device)
    fsmt_full = torch.zeros(I0, J0, 4*M, 8*M, device=device)
    fsmc_full = torch.zeros(I0, J0, 4*M, 8*M, device=device)
    
    # 批量填充矩阵
    # left
    exp_x_left = torch.exp(DX_permuted[:, :, :2*M] * (x_l - x_l) * Sigma_T_expand / Varepsilon_expand)  # 左边界x=x_l
    exp_x_right = torch.exp(DX_permuted[:, :, 2*M:] * (x_l - x_r) * Sigma_T_expand / Varepsilon_expand)  # 右边界x=x_r
    exp_y_left = torch.exp(DY_permuted[:, :, :2*M] * (y_c - y_l) * Sigma_T_expand / Varepsilon_expand)  # 下边界y=y_l
    exp_y_right = torch.exp(DY_permuted[:, :, 2*M:] * (y_c - y_r) * Sigma_T_expand / Varepsilon_expand)  # 上边界y=y_r
    
    fsml_full[:, :, :, :2*M] = torch.einsum('ijkl,ijl->ijkl', VX_permuted[:, :, :, :2*M], exp_x_left)  # 左边界
    fsml_full[:, :, :, 2*M:4*M] = torch.einsum('ijkl,ijl->ijkl', VX_permuted[:, :, :, 2*M:], exp_x_right)
    fsml_full[:, :, :, 4*M:6*M] = torch.einsum('ijkl,ijl->ijkl', VY_permuted[:, :, :, :2*M], exp_y_left)
    fsml_full[:, :, :, 6*M:] = torch.einsum('ijkl,ijl->ijkl', VY_permuted[:, :, :, 2*M:], exp_y_right)
    
    # right
    exp_x_left = torch.exp(DX_permuted[:, :, :2*M] * (x_r - x_l) * Sigma_T_expand / Varepsilon_expand)  # 左边界x=x_l
    exp_x_right = torch.exp(DX_permuted[:, :, 2*M:] * (x_r - x_r) * Sigma_T_expand / Varepsilon_expand)  # 右边界x=x_r
    exp_y_left = torch.exp(DY_permuted[:, :, :2*M] * (y_c - y_l) * Sigma_T_expand / Varepsilon_expand)  # 下边界y=y_l
    exp_y_right = torch.exp(DY_permuted[:, :, 2*M:] * (y_c - y_r) * Sigma_T_expand / Varepsilon_expand)  # 上边界y=y_r
    
    fsmr_full[:, :, :, :2*M] = torch.einsum('ijkl,ijl->ijkl', VX_permuted[:, :, :, :2*M], exp_x_left)  # 左边界
    fsmr_full[:, :, :, 2*M:4*M] = torch.einsum('ijkl,ijl->ijkl', VX_permuted[:, :, :, 2*M:], exp_x_right)
    fsmr_full[:, :, :, 4*M:6*M] = torch.einsum('ijkl,ijl->ijkl', VY_permuted[:, :, :, :2*M], exp_y_left)
    fsmr_full[:, :, :, 6*M:] = torch.einsum('ijkl,ijl->ijkl', VY_permuted[:, :, :, 2*M:], exp_y_right)
    
    # bottom
    exp_x_left = torch.exp(DX_permuted[:, :, :2*M] * (x_c - x_l) * Sigma_T_expand / Varepsilon_expand)  # 左边界x=x_l
    exp_x_right = torch.exp(DX_permuted[:, :, 2*M:] * (x_c - x_r) * Sigma_T_expand / Varepsilon_expand)  # 右边界x=x_r
    exp_y_left = torch.exp(DY_permuted[:, :, :2*M] * (y_l - y_l) * Sigma_T_expand / Varepsilon_expand)  # 下边界y=y_l
    exp_y_right = torch.exp(DY_permuted[:, :, 2*M:] * (y_l - y_r) * Sigma_T_expand / Varepsilon_expand)  # 上边界y=y_r
    
    fsmb_full[:, :, :, :2*M] = torch.einsum('ijkl,ijl->ijkl', VX_permuted[:, :, :, :2*M], exp_x_left)  # 左边界
    fsmb_full[:, :, :, 2*M:4*M] = torch.einsum('ijkl,ijl->ijkl', VX_permuted[:, :, :, 2*M:], exp_x_right)
    fsmb_full[:, :, :, 4*M:6*M] = torch.einsum('ijkl,ijl->ijkl', VY_permuted[:, :, :, :2*M], exp_y_left)
    fsmb_full[:, :, :, 6*M:] = torch.einsum('ijkl,ijl->ijkl', VY_permuted[:, :, :, 2*M:], exp_y_right)
    
    # top
    exp_x_left = torch.exp(DX_permuted[:, :, :2*M] * (x_c - x_l) * Sigma_T_expand / Varepsilon_expand)  # 左边界x=x_l
    exp_x_right = torch.exp(DX_permuted[:, :, 2*M:] * (x_c - x_r) * Sigma_T_expand / Varepsilon_expand)  # 右边界x=x_r
    exp_y_left = torch.exp(DY_permuted[:, :, :2*M] * (y_r - y_l) * Sigma_T_expand / Varepsilon_expand)  # 下边界y=y_l
    exp_y_right = torch.exp(DY_permuted[:, :, 2*M:] * (y_r - y_r) * Sigma_T_expand / Varepsilon_expand)  # 上边界y=y_r
    
    fsmt_full[:, :, :, :2*M] = torch.einsum('ijkl,ijl->ijkl', VX_permuted[:, :, :, :2*M], exp_x_left)  # 左边界
    fsmt_full[:, :, :, 2*M:4*M] = torch.einsum('ijkl,ijl->ijkl', VX_permuted[:, :, :, 2*M:], exp_x_right)
    fsmt_full[:, :, :, 4*M:6*M] = torch.einsum('ijkl,ijl->ijkl', VY_permuted[:, :, :, :2*M], exp_y_left)
    fsmt_full[:, :, :, 6*M:] = torch.einsum('ijkl,ijl->ijkl', VY_permuted[:, :, :, 2*M:], exp_y_right)
    
    # center
    exp_x_left = torch.exp(DX_permuted[:, :, :2*M] * (x_c - x_l) * Sigma_T_expand / Varepsilon_expand)  # 左边界x=x_l
    exp_x_right = torch.exp(DX_permuted[:, :, 2*M:] * (x_c - x_r) * Sigma_T_expand / Varepsilon_expand)  # 右边界x=x_r
    exp_y_left = torch.exp(DY_permuted[:, :, :2*M] * (y_c - y_l) * Sigma_T_expand / Varepsilon_expand)  # 下边界y=y_l
    exp_y_right = torch.exp(DY_permuted[:, :, 2*M:] * (y_c - y_r) * Sigma_T_expand / Varepsilon_expand)  # 上边界y=y_r
    
    fsmc_full[:, :, :, :2*M] = torch.einsum('ijkl,ijl->ijkl', VX_permuted[:, :, :, :2*M], exp_x_left)  # 左边界
    fsmc_full[:, :, :, 2*M:4*M] = torch.einsum('ijkl,ijl->ijkl', VX_permuted[:, :, :, 2*M:], exp_x_right)
    fsmc_full[:, :, :, 4*M:6*M] = torch.einsum('ijkl,ijl->ijkl', VY_permuted[:, :, :, :2*M], exp_y_left)
    fsmc_full[:, :, :, 6*M:] = torch.einsum('ijkl,ijl->ijkl', VY_permuted[:, :, :, 2*M:], exp_y_right)
    
    return fsml_full.permute(2,3,0,1), fsmr_full.permute(2,3,0,1), fsmb_full.permute(2,3,0,1), fsmt_full.permute(2,3,0,1), fsmc_full.permute(2,3,0,1)


# # Inside Projections to the coefficients of all basis functions
# def Projection2alpha(fsml_full, fsmr_full, fsmb_full, fsmt_full, Ml, Mr, Mb, Mt, M, I0, J0, device='cpu'):
#     P2A = torch.zeros([8*M, 8*M, I0, J0]).to(device)
#     for i in range(I0):
#         for j in range(J0):
#             block = torch.cat([Ml[:, :, i, j] @ fsml_full[:, :, i, j], Mr[:, :, i, j] @ fsmr_full[:, :, i, j], Mb[:, :, i, j] @ fsmb_full[:, :, i, j], Mt[:, :, i, j] @ fsmt_full[:, :, i, j]], dim=0)
#             block_old = torch.cat([fsml_full[3*M:4*M, :, i, j], fsml_full[0:M, :, i, j], fsmr_full[M:3*M, :, i, j], fsmb_full[0:2*M, :, i, j], fsmt_full[2*M:4*M, :, i, j]], dim=0)
#             P2A[:, :, i, j] = torch.linalg.inv(block.to(torch.float64)).to(torch.float32)
#     return P2A

def Inflow2alpha(fsml_full, fsmr_full, fsmb_full, fsmt_full, M):
    """
    参数:
        fsml_full, fsmr_full, fsmb_full, fsmt_full: 形状均为 [4M,8M,I0,J0]
        device: 计算设备
        
    返回:
        I2A: 形状 [8M,8M,I0,J0]
    """
    # 批量拼接矩阵 [8M,8M,I0,J0]
    fsml_full_permuted, fsmr_full_permuted, fsmb_full_permuted, fsmt_full_permuted = fsml_full.permute(2,3,0,1), fsmr_full.permute(2,3,0,1), fsmb_full.permute(2,3,0,1), fsmt_full.permute(2,3,0,1)
    block = torch.cat([
        fsml_full_permuted[:, :, 3*M: 4*M, :],  # [I0,J0,M,8M]
        fsml_full_permuted[:, :, 0: M, :],  # [I0,J0,M,8M] 
        fsmr_full_permuted[:, :, M: 3*M, :],    # [I0,J0,2M,8M]
        fsmb_full_permuted[:, :, 0: 2*M, :],  # [I0,J0,2M,8M] 
        fsmt_full_permuted[:, :, 2*M: 4*M, :]   # [I0,J0,2M,8M] 
    ], dim=2)
    
    # 批量计算逆矩阵 (需转为float64增强数值稳定性)
    I2A = torch.linalg.inv(block.to(torch.float64)).permute(2,3,0,1).to(torch.float32)
    return I2A


def MLRO_Dirichlet(VX, DX, Sigma_T, Varepsilon, hx, tol, M, I0, J0, device='cpu'):
    # Ml.size = Mr.size = [2M, 4M, I0, J0], VecSizel.size = VecSizer.size = [I0,J0]
    VX_permuted, DX_permuted = VX.permute(2,3,0,1), DX.permute(1,2,0)
    # 预分配输出
    Ml = torch.zeros(I0, J0, 2*M, 4*M, device=device)
    Mr = torch.zeros_like(Ml)
    VecSizel = torch.zeros(I0, J0, dtype=torch.int32, device=device)
    VecSizer = torch.zeros_like(VecSizel)

    # 预计算公共项 [I0,J0]
    Dx_ratio = DX_permuted * (0.5 * hx * Sigma_T / Varepsilon).unsqueeze(-1)  # [I0,J0,4*M]
    
    # 基函数个数
    idx_l = (torch.abs(Dx_ratio[0, :, :2*M]) < tol) # [J0,2M]
    idx_r = (torch.abs(Dx_ratio[-1, :, 2*M:]) < tol) # [J0,2M]
    idx_lin = (torch.abs(Dx_ratio[1:, :, :2*M]) < tol) # [I0-1,J0,2M]
    idx_rin = (torch.abs(Dx_ratio[:-1, :, 2*M:]) < tol) # [I0-1,J0,2M]
    VecSizel[0, :] = torch.sum(idx_l, dim=-1)
    VecSizel[1:, :] = torch.sum(idx_lin, dim=-1)
    VecSizer[-1, :] = torch.sum(idx_r, dim=-1)
    VecSizer[:-1, :] = torch.sum(idx_rin, dim=-1)
    
    # 批量处理左边界(i=0)
    Matrix_l = torch.cat([VX_permuted[0, :, 3*M:4*M, :2*M],VX_permuted[0, :, 0:M, :2*M]], dim=1)  # [J0,2M,2M]
    Vx_QR = torch.zeros(J0, 2*M, 2*M, device=device)
    for j in range(J0):
        Q1, _ = torch.linalg.qr(Matrix_l[j, :, idx_l[j]], mode='reduced')
        Vx_QR[j] = torch.cat([Matrix_l[j, :, ~idx_l[j]], Q1], dim=-1)
    Vxi = torch.linalg.inv(Vx_QR) # [J0,2M,2M]
    Ml[0, :, :, :M] = Vxi[:, :, M:2*M]
    Ml[0, :, :, 3*M:4*M] = Vxi[:, :, 0:M]
    
    # 批量处理右边界
    Matrix_r = VX_permuted[-1, :, M:3*M, 2*M:]  # [J0,2M,2M]
    Vx_QR = torch.zeros(J0, 2*M, 2*M, device=device)
    for j in range(J0):
        Q1, _ = torch.linalg.qr(Matrix_r[j, :, idx_r[j]], mode='reduced')
        Vx_QR[j] = torch.cat([Q1, Matrix_r[j, :, ~idx_r[j]]], dim=-1)
    Vxi = torch.linalg.inv(Vx_QR) # [J0,2M,2M]
    Mr[-1, :, :, M:3*M] = Vxi
    
    # 批量处理内部
    Matrix_rin = VX_permuted[:-1, :, :, 2*M:]  # [I0-1,J0,4M,2M]
    Matrix_lin = VX_permuted[1:, :, :, :2*M] # [I0-1,J0,4M,2M]
    Vx_QR = torch.zeros(I0-1, J0, 4*M, 4*M, device=device)
    for i in range(I0-1):
        for j in range(J0):
            Q1, _ = torch.linalg.qr(torch.cat([Matrix_rin[i, j, :, idx_rin[i,j]], Matrix_lin[i, j, :, idx_lin[i,j]]], dim=1), mode='reduced')
            Vx_QR[i,j] = torch.cat([Q1[:,:VecSizer[i,j]],Matrix_rin[i, j, :, ~idx_rin[i,j]],Matrix_lin[i, j, :, ~idx_lin[i,j]],Q1[:,VecSizer[i,j]:]], dim=-1)
    Vxi = torch.linalg.inv(Vx_QR) # [I0-1,J0,4M,4M]
    Mr[:-1, :, :, :] = Vxi[:, :, :2*M, :]
    Ml[1:, :, :, :] = Vxi[:, :, 2*M:, :]
    return Ml.permute(2,3,0,1), Mr.permute(2,3,0,1), VecSizel, VecSizer

def MBTO_Dirichlet(VY, DY, Sigma_T, Varepsilon, hy, tol, M, I0, J0, device='cpu'):
    # Ml.size = Mt.size = [2M,4M,I0,J0], VecSizeb.size = VecSizet.size = [I0,J0]
    VY_permuted, DY_permuted = VY.permute(2,3,0,1), DY.permute(1,2,0)
    # 预分配输出
    Mb = torch.zeros(I0, J0, 2*M, 4*M, device=device)
    Mt = torch.zeros_like(Mb)
    VecSizeb = torch.zeros(I0, J0, dtype=torch.int32, device=device)
    VecSizet = torch.zeros_like(VecSizeb)

    # 预计算公共项 [I0,J0]
    Dy_ratio = DY_permuted * (0.5 * hy * Sigma_T / Varepsilon).unsqueeze(-1)  # [I0,J0,4*M]
    
    # 基函数个数
    idy_b = (torch.abs(Dy_ratio[:, 0, :2*M]) < tol) # [I0,2M]
    idy_t = (torch.abs(Dy_ratio[:, -1, 2*M:]) < tol) # [I0,2M]
    idy_bin = (torch.abs(Dy_ratio[:, 1:, :2*M]) < tol) # [I0,J0-1,2M]
    idy_tin = (torch.abs(Dy_ratio[:, :-1, 2*M:]) < tol) # [I0,J0-1,2M]
    VecSizeb[:, 0] = torch.sum(idy_b, dim=-1)
    VecSizeb[:, 1:] = torch.sum(idy_bin, dim=-1)
    VecSizet[:, -1] = torch.sum(idy_t, dim=-1)
    VecSizet[:, :-1] = torch.sum(idy_tin, dim=-1)
    
    # 批量处理下边界(j=0)
    Matrix_b = VY_permuted[:, 0, :2*M, :2*M]  # [I0,2M,2M]
    Vy_QR = torch.zeros(I0, 2*M, 2*M, device=device)
    for i in range(I0):
        Q1, _ = torch.linalg.qr(Matrix_b[i, :, idy_b[i]], mode='reduced')
        Vy_QR[i] = torch.cat([Matrix_b[i, :, ~idy_b[i]], Q1], dim=-1)
    Vxi = torch.linalg.inv(Vy_QR) # [I0,2M,2M]
    Mb[:, 0, :, :2*M] = Vxi
    
    # 批量处理上边界
    Matrix_t = VY_permuted[:, -1, 2*M:, 2*M:]  # [I0,2M,2M]
    Vy_QR = torch.zeros(I0, 2*M, 2*M, device=device)
    for i in range(I0):
        Q1, _ = torch.linalg.qr(Matrix_t[i, :, idy_t[i]], mode='reduced')
        Vy_QR[i] = torch.cat([Q1, Matrix_t[i, :, ~idy_t[i]]], dim=-1)
    Vyi = torch.linalg.inv(Vy_QR) # [J0,2M,2M]
    Mt[:, -1, :, 2*M:] = Vyi
    
    # 批量处理内部
    Matrix_tin = VY_permuted[:, :-1, :, 2*M:]  # [I0,J0-1,4M,2M]
    Matrix_bin = VY_permuted[:, 1:, :, :2*M] # [I0,J0-1,4M,2M]
    Vy_QR = torch.zeros(I0, J0-1, 4*M, 4*M, device=device)
    for i in range(I0):
        for j in range(J0-1):
            Q1, _ = torch.linalg.qr(torch.cat([Matrix_tin[i, j, :, idy_tin[i,j]], Matrix_bin[i, j, :, idy_bin[i,j]]], dim=1), mode='reduced')
            Vy_QR[i,j] = torch.cat([Q1[:,:VecSizet[i,j]],Matrix_tin[i, j, :, ~idy_tin[i,j]],Matrix_bin[i, j, :, ~idy_bin[i,j]],Q1[:,VecSizet[i,j]:]], dim=-1)
            # Vy_QR[i,j] = torch.cat([Matrix_tin[i, j, :, ~idy_tin[i,j]],Q1, Matrix_bin[i, j, :, ~idy_bin[i,j]]], dim=-1)
    Vyi = torch.linalg.inv(Vy_QR) # [I0,J0-1,4M,4M]
    Mt[:, :-1, :, :] = Vyi[:, :, :2*M, :]
    Mb[:, 1:, :, :] = Vyi[:, :, 2*M:, :]
    
    return Mb.permute(2,3,0,1), Mt.permute(2,3,0,1), VecSizeb, VecSizet

def basis_mask(VecSize, M, device=None):
    if device is None:
        device = VecSize.device
    return torch.cat([
        torch.arange(2*M-1, -1, -1, device=device).reshape([1, 2*M, 1, 1]) < VecSize[:, 0, :, :].unsqueeze(1),
        torch.arange(2*M, device=device).reshape([1, 2*M, 1, 1]) < VecSize[:, 1, :, :].unsqueeze(1),
        torch.arange(2*M-1, -1, -1, device=device).reshape([1, 2*M, 1, 1]) < VecSize[:, 2, :, :].unsqueeze(1),
        torch.arange(2*M, device=device).reshape([1, 2*M, 1, 1]) < VecSize[:, 3, :, :].unsqueeze(1),
    ], dim=1)


def block_basis_mask(VecSize, M, device=None):
    mask = basis_mask(VecSize, M, device).unsqueeze(2)
    return mask.expand(-1, -1, 8*M, -1, -1)


# Here VecSize includes VecSizel, VecSizer, VecSizeb, VecSizet, they belongs to 4 channels
# get the selected and unselected basis function/velocity modes
# for batch
def restrict_basis(basis, VecSize, M, I0, J0, device='cpu'):
    # rhs.shape = [num_data, 8*M, I0, J0]
    # VecSize.shape = [num_data, 4, I0, J0]
    squeeze_output = False
    if len(basis.shape) == 3:
        basis = basis.unsqueeze(0)
        squeeze_output = True
    if len(VecSize.shape) == 3:
        VecSize = VecSize.unsqueeze(0)
    device = basis.device
    VecSize = VecSize.to(device)
    basis_selected = torch.zeros([basis.shape[0], 8*M, I0, J0], device=device, dtype=basis.dtype)
    basis_unselected = torch.zeros([basis.shape[0], 8*M, I0, J0], device=device, dtype=basis.dtype)
    mask = basis_mask(VecSize, M, device)
    basis_selected[mask] = basis[mask]
    basis_unselected[~mask] = basis[~mask]
    if squeeze_output:
        basis_selected = basis_selected.squeeze(0)
        basis_unselected = basis_unselected.squeeze(0)
    return basis_selected, basis_unselected

def squeeze_basis(basis, VecSize):
    # basis.shape = [num_data, 8*M, I0, J0]
    # VecSize.shape = [num_data, 4, I0, J0]
    squeeze_output = False
    if len(basis.shape) == 3:
        basis = basis.unsqueeze(0)
        squeeze_output = True
    if len(VecSize.shape) == 3:
        VecSize = VecSize.unsqueeze(0)
    VecSize = VecSize.to(basis.device)
    num_data, M, I0, J0, device = basis.size(0), int(basis.size(1)/8), basis.size(2), basis.size(3), basis.device
    mask = basis_mask(VecSize, M, device)

    basis_selected = basis[mask]
    basis_unselected = basis[~mask]
    
    if squeeze_output:
        basis_selected = basis_selected.squeeze(0)
        basis_unselected = basis_unselected.squeeze(0)
    return basis_selected, basis_unselected

def func_inflow2alpha(inflow, I2A, M, I0, J0, device='cpu'):
    # inflow.shape = [batch, 8*M, I0, J0]
    # I2A.shape = [batch, 8*M, 8*M, I0, J0]
    mark = 0
    if len(inflow.shape) == 3:
        inflow = inflow.unsqueeze(0)
        mark = 1
    if len(I2A.shape) == 4:
        I2A = I2A.unsqueeze(0)
    inflow_permuted, I2A_permuted = inflow.permute(0,2,3,1), I2A.permute(0,3,4,1,2)
    alpha_permuted = torch.einsum('bijom,bijm->bijo', I2A_permuted, inflow_permuted)
    alpha  = alpha_permuted.permute(0,3,1,2)
    if mark == 1:
        alpha = alpha.squeeze(0)
    return alpha

# for batch
def func_alpha_Dirichlet(alpha, fsmLRBTC, MLRBT, VecSize, M, I0, J0, device='cpu'):
    # alpha.shape = [batch, 8*M, I0, J0]
    # fsmLRBTC.shape = [batch, 5, 4*M, 8*M, I0, J0]
    # MLRBT.shape = [batch, 4, 2*M, 4*M, I0, J0]
    [alpha_selected, alpha_unselected] = restrict_basis(alpha, VecSize, M, I0, J0, device)
    mark = 0
    if len(alpha.shape) == 3:
        alpha = alpha.unsqueeze(0)
        mark = 1
    if len(fsmLRBTC.shape)==5:
        fsmLRBTC = fsmLRBTC.unsqueeze(0)
        mark = 1
    if len(MLRBT.shape)==5:
        MLRBT = MLRBT.unsqueeze(0)
        mark = 1
    # 分解输入张量
    # [batch, 4*M, 8*M, I0, J0]
    fsml_full, fsmr_full, fsmb_full, fsmt_full = fsmLRBTC[:, 0], fsmLRBTC[:, 1], fsmLRBTC[:, 2], fsmLRBTC[:, 3]
    # [batch, 2*M, 4*M, I0, J0]
    Ml, Mr, Mb, Mt = MLRBT[:, 0], MLRBT[:, 1], MLRBT[:, 2],MLRBT[:, 3]
    sol_in = torch.zeros(alpha.shape[0], 8*M, I0, J0, device=device)
    sol_out = torch.zeros(alpha.shape[0], 8*M, I0, J0, device=device)
    # left
    sol_in[:, :2*M, :, :] = torch.einsum('bklij,bloij,boij->bkij', Ml, fsml_full, alpha)
    sol_out[:, :2*M, 1:, :] = torch.einsum('bklij,bloij,boij->bkij', Ml[:, :, :, 1:, :], fsmr_full[:, :, :, :-1, :], alpha[:, :, :-1, :])
    # right
    sol_in[:, 2*M:4*M, :, :] = torch.einsum('bklij,bloij,boij->bkij', Mr, fsmr_full, alpha)
    sol_out[:, 2*M:4*M, :-1, :] = torch.einsum('bklij,bloij,boij->bkij', Mr[:, :, :, :-1, :], fsml_full[:, :, :, 1:, :], alpha[:, :, 1:, :])
    # bottom
    sol_in[:, 4*M:6*M, :, :] = torch.einsum('bklij,bloij,boij->bkij', Mb, fsmb_full, alpha)
    sol_out[:, 4*M:6*M, :, 1:] = torch.einsum('bklij,bloij,boij->bkij', Mb[:, :, :, :, 1:], fsmt_full[:, :, :, :, :-1], alpha[:, :, :, :-1])
    # top
    sol_in[:, 6*M:, :, :] = torch.einsum('bklij,bloij,boij->bkij', Mt, fsmt_full, alpha)
    sol_out[:, 6*M:, :, :-1] = torch.einsum('bklij,bloij,boij->bkij', Mt[:, :, :, :, :-1], fsmb_full[:, :, :, :, 1:], alpha[:, :, :, 1:])
    
    sol = sol_in - sol_out
    if mark == 1:
        sol = sol.squeeze(0)
    [sol_selected, sol_unselected] = restrict_basis(sol, VecSize, M, I0, J0, device)
    result = sol_selected + alpha_unselected
    return result
    
# for batch
# for batch
def func_alpha_Dirichlet_4correction(alpha, fsmLRBTC, MLRBT, VecSize, M, I0, J0, device='cpu'):
    # alpha.shape = [batch, 8*M, I0, J0]
    # fsmLRBTC.shape = [batch, 5, 4*M, 8*M, I0, J0]
    # MLRBT.shape = [batch, 4, 2*M, 4*M, I0, J0]
    [alpha_selected, alpha_unselected] = restrict_basis(alpha, VecSize, M, I0, J0, device)
    mark = 0
    if len(alpha.shape) == 3:
        alpha = alpha.unsqueeze(0)
        mark = 1
    if len(fsmLRBTC.shape)==5:
        fsmLRBTC = fsmLRBTC.unsqueeze(0)
        mark = 1
    if len(MLRBT.shape)==5:
        MLRBT = MLRBT.unsqueeze(0)
        mark = 1
    # 分解输入张量
    # [batch, 4*M, 8*M, I0, J0]
    fsml_full, fsmr_full, fsmb_full, fsmt_full = fsmLRBTC[:, 0], fsmLRBTC[:, 1], fsmLRBTC[:, 2], fsmLRBTC[:, 3]
    # [batch, 2*M, 4*M, I0, J0]
    Ml, Mr, Mb, Mt = MLRBT[:, 0], MLRBT[:, 1], MLRBT[:, 2],MLRBT[:, 3]
    sol_in = torch.zeros(alpha.shape[0], 8*M, I0, J0, device=device)
    sol_out = torch.zeros(alpha.shape[0], 8*M, I0, J0, device=device)
    # left
    sol_in[:, :2*M, :, :] = torch.einsum('bklij,bloij,boij->bkij', Ml, fsml_full, alpha)
    sol_out[:, :2*M, 1:, :] = torch.einsum('bklij,bloij,boij->bkij', Ml[:, :, :, 1:, :], fsmr_full[:, :, :, :-1, :], alpha[:, :, :-1, :])
    # right
    sol_in[:, 2*M:4*M, :, :] = torch.einsum('bklij,bloij,boij->bkij', Mr, fsmr_full, alpha)
    sol_out[:, 2*M:4*M, :-1, :] = torch.einsum('bklij,bloij,boij->bkij', Mr[:, :, :, :-1, :], fsml_full[:, :, :, 1:, :], alpha[:, :, 1:, :])
    # bottom
    sol_in[:, 4*M:6*M, :, :] = torch.einsum('bklij,bloij,boij->bkij', Mb, fsmb_full, alpha)
    sol_out[:, 4*M:6*M, :, 1:] = torch.einsum('bklij,bloij,boij->bkij', Mb[:, :, :, :, 1:], fsmt_full[:, :, :, :, :-1], alpha[:, :, :, :-1])
    # top
    sol_in[:, 6*M:, :, :] = torch.einsum('bklij,bloij,boij->bkij', Mt, fsmt_full, alpha)
    sol_out[:, 6*M:, :, :-1] = torch.einsum('bklij,bloij,boij->bkij', Mt[:, :, :, :, :-1], fsmb_full[:, :, :, :, 1:], alpha[:, :, :, 1:])
    
    sol = sol_in - sol_out
    if mark == 1:
        sol = sol.squeeze(0)
    [sol_selected, sol_unselected] = restrict_basis(sol, VecSize, M, I0, J0, device)
    result = sol_unselected + alpha_selected
    return result


# for batch
def func_inflow_Dirichlet(inflow, I2A, fsmLRBTC, MLRBT, VecSize, M, I0, J0, device='cpu'):
    # inflow.shape = [batch, 8*M, I0, J0]
    # I2A.shape = [batch, 8*M, 8*M, I0, J0]
    # fsmLRBTC.shape = [batch, 5, 4*M, 8*M, I0, J0]
    # MLRBT.shape = [batch, 4, 2*M, 4*M, I0, J0]
    alpha = func_inflow2alpha(inflow, I2A, M, I0, J0, device)
    result = func_alpha_Dirichlet(alpha, fsmLRBTC, MLRBT, VecSize, M, I0, J0, device)
    return result

# for batch
def func_inflow_Dirichlet_4correction(inflow, I2A, fsmLRBTC, MLRBT, VecSize, M, I0, J0, device='cpu'):
    # inflow.shape = [batch, 8*M, I0, J0]
    # I2A.shape = [batch, 8*M, 8*M, I0, J0]
    # fsmLRBTC.shape = [batch, 5, 4*M, 8*M, I0, J0]
    # MLRBT.shape = [batch, 4, 2*M, 4*M, I0, J0]
    alpha = func_inflow2alpha(inflow, I2A, M, I0, J0, device)
    result = func_alpha_Dirichlet_4correction(alpha, fsmLRBTC, MLRBT, VecSize, M, I0, J0, device)
    return result


def funcDc(inflow, vec, I2A, fsmLRBTC, M, I0, J0, device = 'cpu'):
    # inflow.shape = [batch, 8*M, I0, J0]
    # vec.shape = [batch, 4*M, I0, J0]
    # I2A.shape = [batch, 8*M, 8*M, I0, J0]
    # fsmLRBTC.shape = [batch, 5, 4*M, 8*M, I0, J0]
    mark = 0
    if len(inflow.shape) == 3:
        inflow = inflow.unsqueeze(0)
        mark = 1
    if len(vec.shape) == 3:
        vec = vec.unsqueeze(0)
        mark = 1
    if len(I2A.shape) == 4 and len(fsmLRBTC.shape) == 5:
        I2A, fsmLRBTC = I2A.unsqueeze(0), fsmLRBTC.unsqueeze(0)
        mark = 1
    fsmc_full = fsmLRBTC[:, 4, :, :, :, :]
    sol = torch.einsum('bloij,bomij,bmij->blij', fsmc_full, I2A, inflow) + vec
    if mark == 1:
        sol = sol.squeeze(0)
    return sol

    

# block diagonal of func_inflow_Dirichlet
# for batch
def funcD(inflow, I2A, fsmLRBTC, MLRBT, VecSize, M, I0, J0, device='cpu'):
    # inflow.shape = [batch, 8*M, I0, J0]
    # I2A.shape = [batch, 8*M, 8*M, I0, J0]
    # fsmLRBTC.shape = [batch, 5, 4*M, 8*M, I0, J0]
    # MLRBT.shape = [batch, 4, 2*M, 4*M, I0, J0]
    # VecSize.shape = [batch, 4, I0, J0]
    mark = 0
    if len(inflow.shape) == 3:
        inflow = inflow.unsqueeze(0)
        mark = 1
    if len(I2A.shape) == 4 and len(fsmLRBTC.shape)==5 and len(MLRBT.shape)==5 and len(VecSize.shape)==3:
        I2A, fsmLRBTC, MLRBT, VecSize = I2A.unsqueeze(0), fsmLRBTC.unsqueeze(0), MLRBT.unsqueeze(0), VecSize.unsqueeze(0)
        mark = 1
    device = inflow.device
    fsml_full, fsmr_full, fsmb_full, fsmt_full = fsmLRBTC[:, 0, :, :, :, :], fsmLRBTC[:, 1, :, :, :, :], fsmLRBTC[:, 2, :, :, :, :], fsmLRBTC[:, 3, :, :, :, :]
    Ml, Mr, Mb, Mt = MLRBT[:, 0, :, :, :, :], MLRBT[:, 1, :, :, :, :], MLRBT[:, 2, :, :, :, :], MLRBT[:, 3, :, :, :, :]
    # block_sol.shape = [batch, 8*M, 8*M, I0, J0]
    block_sol = torch.cat([torch.einsum('bklij,bloij,bomij->bkmij', Ml, fsml_full, I2A),
                           torch.einsum('bklij,bloij,bomij->bkmij', Mr, fsmr_full, I2A),
                           torch.einsum('bklij,bloij,bomij->bkmij', Mb, fsmb_full, I2A),
                           torch.einsum('bklij,bloij,bomij->bkmij', Mt, fsmt_full, I2A)], dim=1)
    VecSize = VecSize.to(inflow.device)
    mask_selected = block_basis_mask(VecSize, M, inflow.device)
    block = torch.zeros(inflow.shape[0], 8*M, 8*M, I0, J0, device=device)
    block[mask_selected] = block_sol[mask_selected]
    block[~mask_selected] = I2A[~mask_selected]
    sol = torch.einsum('bkmij, bmij->bkij', block, inflow)
    if mark == 1:
        sol = sol.squeeze(0)
    return sol


# inverse of funcD
# for batch
def funcDinv(inflow, I2A, fsmLRBTC, MLRBT, VecSize, M, I0, J0, device='cpu'):
    # inflow.shape = [batch, 8*M, I0, J0]
    # I2A.shape = [batch, 8*M, 8*M, I0, J0]
    # fsmLRBTC.shape = [batch, 5, 4*M, 8*M, I0, J0]
    # MLRBT.shape = [batch, 4, 2*M, 4*M, I0, J0]
    # VecSize.shape = [batch, 4, I0, J0]
    mark = 0
    if len(inflow.shape) == 3:
        inflow = inflow.unsqueeze(0)
        mark = 1
    if len(I2A.shape) == 4 and len(fsmLRBTC.shape)==5 and len(MLRBT.shape)==5 and len(VecSize.shape)==3:
        I2A, fsmLRBTC, MLRBT, VecSize = I2A.unsqueeze(0), fsmLRBTC.unsqueeze(0), MLRBT.unsqueeze(0), VecSize.unsqueeze(0)
        mark = 1
    device = inflow.device
    fsml_full, fsmr_full, fsmb_full, fsmt_full = fsmLRBTC[:, 0, :, :, :, :], fsmLRBTC[:, 1, :, :, :, :], fsmLRBTC[:, 2, :, :, :, :], fsmLRBTC[:, 3, :, :, :, :]
    Ml, Mr, Mb, Mt = MLRBT[:, 0, :, :, :, :], MLRBT[:, 1, :, :, :, :], MLRBT[:, 2, :, :, :, :], MLRBT[:, 3, :, :, :, :]
    # block_sol.shape = [batch, 8*M, 8*M, I0, J0]
    block_sol = torch.cat([torch.einsum('bklij,bloij,bomij->bkmij', Ml, fsml_full, I2A),
                           torch.einsum('bklij,bloij,bomij->bkmij', Mr, fsmr_full, I2A),
                           torch.einsum('bklij,bloij,bomij->bkmij', Mb, fsmb_full, I2A),
                           torch.einsum('bklij,bloij,bomij->bkmij', Mt, fsmt_full, I2A)], dim=1)
    VecSize = VecSize.to(inflow.device)
    mask_selected = block_basis_mask(VecSize, M, inflow.device)
    block = torch.zeros(inflow.shape[0], 8*M, 8*M, I0, J0, device=device)
    block[mask_selected] = block_sol[mask_selected]
    block[~mask_selected] = I2A[~mask_selected]
    sol = torch.linalg.solve(block.permute(0,3,4,1,2), inflow.permute(0,2,3,1)).permute(0,3,1,2)
    if mark == 1:
        sol = sol.squeeze(0)
    return sol



# generate special solution from external source term
# for batch
def special_sol_aniso(Q, Sigma_T, Sigma_a, Varepsilon, Kappa, omega, M, I0, J0, device = 'cpu'):
    # Q.shape = [batch, 4*M, I0, J0]
    # Sigma_T.shape, Sigma_a.shape, Varepsilon.shape = [batch, I0, J0]
    mark = 0
    if len(Q.shape) == 3:
        Q = Q.unsqueeze(0)
        mark = 1
    if len(Sigma_T.shape) == 2 and len(Sigma_a.shape) == 2 and len(Varepsilon.shape) == 2:
        Sigma_T, Sigma_a, Varepsilon = Sigma_T.unsqueeze(0), Sigma_a.unsqueeze(0), Varepsilon.unsqueeze(0)
        mark = 1
    # block.shape = [batch, I0, J0, 4*M, 4*M]
    block = (Sigma_T/Varepsilon**2).unsqueeze(-1).unsqueeze(-1)*torch.eye(4*M, 4*M, device=device)-(Sigma_T/Varepsilon**2-Sigma_a).unsqueeze(-1).unsqueeze(-1)*Kappa@torch.diag(omega)
    vec = torch.linalg.solve(block.to(torch.float64) ,Q.permute(0,2,3,1).to(torch.float64)).permute(0,3,1,2).to(torch.float32)
    if mark == 1:
        vec = vec.squeeze(0)
    return vec


# assume here Q isotropic
def special_sol_iso(Q, Sigma_T, Sigma_a, Varepsilon, Kappa, omega, M, I0, J0, device = 'cpu'):
    # Q.shape = [batch_size, 4*M, I0, J0]
    # Sigma_a.shape = [batch_size, I0, J0]
    mark = 0
    if len(Q.shape) == 3:
        Q = Q.unsqueeze(0)
        mark = 1
    if len(Sigma_a.shape) == 2:
        Sigma_a = Sigma_a.unsqueeze(0)
        mark = 1
    vec = Q/Sigma_a.unsqueeze(1)
    if mark == 1:
        vec = vec.squeeze(0)
    return vec


def generate_TFPS_rhs_Dirichlet_iso(Q, psiL, psiR, psiB, psiT, Coef,  Kappa, omega, MLRBT, VecSize, M, I0, J0, device = 'cpu'):
    # Q.shape = [num_data,4*M,I0,J0]
    # psiL.shape, psiR.shape = [num_data,2*M, J0], [num_data,2*M, J0]
    # psiB.shape, psiT.shape = [num_data,2*M, I0], [num_data,2*M, I0]
    # Coef.shape = [num_data, 3, I0, J0]
    # MLRBT.shape = [num_data, 4, 2*M, 4*M, I0, J0]
    mark = 0
    if len(Q.shape) == 3:
        Q = Q.unsqueeze(0)
        mark = 1
    if len(psiL.shape)==2 and len(psiR.shape)==2 and len(psiB.shape)==2 and len(psiT.shape)==2:
        psiL, psiR, psiB, psiT = psiL.unsqueeze(0), psiR.unsqueeze(0), psiB.unsqueeze(0), psiT.unsqueeze(0) 
        mark = 1 
    if len(Coef.shape) == 3:
        Coef = Coef.unsqueeze(0)
        mark = 1
    if len(MLRBT.shape)==5:
        MLRBT = MLRBT.unsqueeze(0)
        mark = 1
    Sigma_T, Sigma_a, Varepsilon = Coef[:, 0, :, :], Coef[:, 1, :, :], torch.exp(Coef[:, 2, :, :])
    Ml, Mr, Mb, Mt = MLRBT[:, 0, :, :, :, :], MLRBT[:, 1, :, :, :, :], MLRBT[:, 2, :, :, :, :], MLRBT[:, 3, :, :, :, :]
    vec = special_sol_iso(Q, Sigma_T, Sigma_a, Varepsilon, Kappa, omega, M, I0, J0, device)
    rhs_in = torch.zeros(Q.shape[0], 8*M, I0, J0, device=device)
    rhs_out = torch.zeros(Q.shape[0], 8*M, I0, J0, device=device)
    # handle boundary condition
    rhs_out[:, :2*M, 0, :] = torch.einsum('bkmj,bmj->bkj', Ml[:, :, :, 0, :], torch.hstack([psiL[:, M:2*M, :], torch.zeros(Q.shape[0], 2*M, J0, device=device), psiL[:, 0:M, :]]))
    rhs_out[:, 2*M:4*M, -1, :] = torch.einsum('bkmj,bmj->bkj', Mr[:, :, :, -1, :], torch.hstack([torch.zeros(Q.shape[0], M, J0, device=device), psiR, torch.zeros(Q.shape[0], M, J0, device=device)]))
    rhs_out[:, 4*M:6*M, :, 0] = torch.einsum('bkmi,bmi->bki', Mb[:, :, :, :, 0], torch.hstack([psiB, torch.zeros(Q.shape[0], 2*M, I0, device=device)]))
    rhs_out[:, 6*M:, :, -1] = torch.einsum('bkmi,bmi->bki', Mt[:, :, :, :, -1], torch.hstack([torch.zeros(Q.shape[0], 2*M, I0, device=device), psiT]))
    # left
    rhs_in[:, :2*M, :, :] = torch.einsum('bkmij,bmij->bkij', Ml, vec)
    rhs_out[:, :2*M, 1:, :] = torch.einsum('bkmij,bmij->bkij', Ml[:, :, :, 1:, :], vec[:, :, :-1, :])
    # right
    rhs_in[:, 2*M:4*M, :, :] = torch.einsum('bkmij,bmij->bkij', Mr, vec)
    rhs_out[:, 2*M:4*M, :-1, :] = torch.einsum('bkmij,bmij->bkij', Mr[:, :, :, :-1, :], vec[:, :, 1:, :])
    # bottom
    rhs_in[:, 4*M:6*M, :, :] = torch.einsum('bkmij,bmij->bkij', Mb, vec)
    rhs_out[:, 4*M:6*M, :, 1:] = torch.einsum('bkmij,bmij->bkij', Mb[:, :, :, :, 1:], vec[:, :, :, :-1])
    # top
    rhs_in[:, 6*M:, :, :] = torch.einsum('bkmij,bmij->bkij', Mt, vec)
    rhs_out[:, 6*M:, :, :-1] = torch.einsum('bkmij,bmij->bkij', Mt[:, :, :, :, :-1], vec[:, :, :, 1:])
    
    rhs = rhs_out - rhs_in
    if mark == 1:
        rhs = rhs.squeeze(0)
        vec = vec.squeeze(0)
    [rhs_selected, rhs_unselected] = restrict_basis(rhs, VecSize, M, I0, J0, device)
    return rhs_selected, rhs_unselected, vec

def generate_TFPS_rhs_Dirichlet_aniso(Q, psiL, psiR, psiB, psiT, Coef,  Kappa, omega, MLRBT, VecSize, M, I0, J0, device = 'cpu'):
    # Q.shape = [num_data,4*M,I0,J0]
    # psiL.shape, psiR.shape = [num_data,2*M, J0], [num_data,2*M, J0]
    # psiB.shape, psiT.shape = [num_data,2*M, I0], [num_data,2*M, I0]
    # Coef.shape = [num_data, 3, I0, J0]
    # Ml.shape = Mr.shape = Mb.shape = Mt.shape = [num_data, 2*M, 4*M, I0, J0]
    mark = 0
    if len(Q.shape) == 3:
        Q = Q.unsqueeze(0)
        mark = 1
    if len(psiL.shape)==2 and len(psiR.shape)==2 and len(psiB.shape)==2 and len(psiT.shape)==2:
        psiL, psiR, psiB, psiT = psiL.unsqueeze(0), psiR.unsqueeze(0), psiB.unsqueeze(0), psiT.unsqueeze(0) 
        mark = 1 
    if len(Coef.shape) == 3:
        Coef = Coef.unsqueeze(0)
        mark = 1
    if len(MLRBT.shape)==5:
        MLRBT = MLRBT.unsqueeze(0)
        mark = 1
    Sigma_T, Sigma_a, Varepsilon = Coef[:, 0, :, :], Coef[:, 1, :, :], torch.exp(Coef[:, 2, :, :])
    Ml, Mr, Mb, Mt = MLRBT[:, 0, :, :, :, :], MLRBT[:, 1, :, :, :, :], MLRBT[:, 2, :, :, :, :], MLRBT[:, 3, :, :, :, :]
    vec = special_sol_aniso(Q, Sigma_T, Sigma_a, Varepsilon, Kappa, omega, M, I0, J0, device)
    rhs_in = torch.zeros(Q.shape[0], 8*M, I0, J0, device=device)
    rhs_out = torch.zeros(Q.shape[0], 8*M, I0, J0, device=device)
    # handle boundary condition
    rhs_out[:, :2*M, 0, :] = torch.einsum('bkmj,bmj->bkj', Ml[:, :, :, 0, :], torch.hstack([psiL[:, M:2*M, :], torch.zeros(Q.shape[0], 2*M, J0, device=device), psiL[:, 0:M, :]]))
    rhs_out[:, 2*M:4*M, -1, :] = torch.einsum('bkmj,bmj->bkj', Mr[:, :, :, -1, :], torch.hstack([torch.zeros(Q.shape[0], M, J0, device=device), psiR, torch.zeros(Q.shape[0], M, J0, device=device)]))
    rhs_out[:, 4*M:6*M, :, 0] = torch.einsum('bkmi,bmi->bki', Mb[:, :, :, :, 0], torch.hstack([psiB, torch.zeros(Q.shape[0], 2*M, I0, device=device)]))
    rhs_out[:, 6*M:, :, -1] = torch.einsum('bkmi,bmi->bki', Mt[:, :, :, :, -1], torch.hstack([torch.zeros(Q.shape[0], 2*M, I0, device=device), psiT]))
    # left
    rhs_in[:, :2*M, :, :] = torch.einsum('bkmij,bmij->bkij', Ml, vec)
    rhs_out[:, :2*M, 1:, :] = torch.einsum('bkmij,bmij->bkij', Ml[:, :, :, 1:, :], vec[:, :, :-1, :])
    # right
    rhs_in[:, 2*M:4*M, :, :] = torch.einsum('bkmij,bmij->bkij', Mr, vec)
    rhs_out[:, 2*M:4*M, :-1, :] = torch.einsum('bkmij,bmij->bkij', Mr[:, :, :, :-1, :], vec[:, :, 1:, :])
    # bottom
    rhs_in[:, 4*M:6*M, :, :] = torch.einsum('bkmij,bmij->bkij', Mb, vec)
    rhs_out[:, 4*M:6*M, :, 1:] = torch.einsum('bkmij,bmij->bkij', Mb[:, :, :, :, 1:], vec[:, :, :, :-1])
    # top
    rhs_in[:, 6*M:, :, :] = torch.einsum('bkmij,bmij->bkij', Mt, vec)
    rhs_out[:, 6*M:, :, :-1] = torch.einsum('bkmij,bmij->bkij', Mt[:, :, :, :, :-1], vec[:, :, :, 1:])
    
    rhs = rhs_out - rhs_in
    if mark == 1:
        rhs = rhs.squeeze(0)
        vec = vec.squeeze(0)
    [rhs_selected, rhs_unselected] = restrict_basis(rhs, VecSize, M, I0, J0, device)
    return rhs_selected, rhs_unselected, vec


# for batch
def correction(inflow, rhs_unselected, I2A, fsmLRBTC, MLRBT, VecSize, M, I0, J0, device='cpu'):
    # inflow.shape = [batch, 8*M, I0, J0]
    # rhs_unselected.shape = [batch, 8*M, I0, J0]
    # fsmLRBTC.shape = [batch, 5, 4*M, 8*M, I0, J0]
    # MLRBT.shape = [batch, 4, 2*M, 4*M, I0, J0]
    # VecSize.shape = [batch, 4, I0, J0]
    mark = 0
    if len(inflow.shape) == 3 and len(rhs_unselected.shape) == 3:
        inflow, rhs_unselected = inflow.unsqueeze(0), rhs_unselected.unsqueeze(0)
        mark = 1
    if len(I2A.shape) == 4 and len(fsmLRBTC.shape) == 5 and len(MLRBT.shape) == 5 and len(VecSize.shape) == 3:
        I2A, fsmLRBTC, MLRBT, VecSize = I2A.unsqueeze(0), fsmLRBTC.unsqueeze(0), MLRBT.unsqueeze(0), VecSize.unsqueeze(0)
        mark = 1
    result = func_inflow_Dirichlet_4correction(inflow, I2A, fsmLRBTC, MLRBT, VecSize, M, I0, J0, device)
    alpha_selected, sol_unselected = restrict_basis(result, VecSize, M, I0, J0, device)
    alpha_correction = rhs_unselected - sol_unselected
    alpha_corrected = alpha_selected + alpha_correction
    fsml_full, fsmr_full, fsmb_full, fsmt_full, fsmc_full = fsmLRBTC[:, 0, :, :, :, :], fsmLRBTC[:, 1, :, :, :, :], fsmLRBTC[:, 2, :, :, :, :], fsmLRBTC[:, 3, :, :, :, :], fsmLRBTC[:, 4, :, :, :, :]
    # block.shape = [batch,8*M,8*M,I0,J0]
    block = torch.cat([fsml_full[:, 3*M:4*M], fsml_full[:, 0:M], fsmr_full[:, M:3*M], fsmb_full[:, 0:2*M], fsmt_full[:, 2*M:4*M]], dim = 1)
    inflow_corrected = torch.einsum('bkmij,bmij->bkij', block, alpha_corrected)
    valuec_corrected = torch.einsum('bkmij,bmij->bkij', fsmc_full, alpha_corrected)
    if mark == 1:
        alpha_corrected, inflow_corrected, valuec_corrected = alpha_corrected.squeeze(0), inflow_corrected.squeeze(0), valuec_corrected.squeeze(0)
    return alpha_corrected, inflow_corrected, valuec_corrected
