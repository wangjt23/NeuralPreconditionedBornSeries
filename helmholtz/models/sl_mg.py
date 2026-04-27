'''
shift laplacian preconditioner

date  : 2025.7.29
author: Juntao Wang
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .gmres import gmres

def complex_mul(a, b):
    """
    torch complex multiply: (a_real + i * a_imag) * (b_real + i * b_imag)

    input:
        a, b: shape = (batch, 2, N, N)
    return:
        out: shape = (batch, 2, N, N)
    """
    a_real, a_imag = a[:, 0:1], a[:, 1:2] # batch, 1, N, N
    b_real, b_imag = b[:, 0:1], b[:, 1:2] # batch, 1, N, N

    real = a_real * b_real - a_imag * b_imag
    imag = a_real * b_imag + a_imag * b_real

    return torch.cat([real, imag], dim=1)# batch, 2, N, N

def set_abc_layer(domain_size, width, value):
    # not just set value, set value linearly increase to value within boundary layer
    abc_coord = torch.arange(width)
    # Sigma function (quadratic)
    layer_outer = value * (1 - abc_coord / width) ** 2
    layer_1d = torch.zeros((domain_size,))
    layer_1d[:width] = layer_outer
    layer_1d[-width:] = torch.flip(layer_outer, dims=[0])
    layer_x_2d, layer_y_2d = torch.meshgrid(layer_1d, layer_1d, indexing='xy')
    return layer_x_2d + layer_y_2d

class shift_lap(nn.Module):
    def __init__(self, float_type=torch.float64):
        super().__init__()
        self.complex_type = torch.complex128 if float_type==torch.float64 else torch.complex64
        self.float_type = float_type


        self.lap_op = nn.Conv2d(2,2,kernel_size=3,stride=1,padding=1,groups=2,bias=False)
        kernel = torch.tensor([[[[0.,  1., 0.],
                                 [1., -4., 1.],
                                 [0.,  1., 0.]]]],dtype=float_type).repeat(2,1,1,1)
        self.lap_op.weight = nn.Parameter(kernel)
        self.lap_op.weight.requires_grad = False

        self.beta1 = 1.
        self.beta2 = 1.

    def forward(self, u, k, absorb, dx):
        '''
        input:
            u: float tensor, batch,2,N,N  real/imag
            k: float tensor, batch,1,N,N
        return:
            Au: complex tensor, batch,1,N,N
        '''
        if isinstance(u, int):
            return 0
        
        # u = torch.cat([u.real, u.imag], dim=1)#.to(torch.float32)
        lap_u = self.lap_op(u)
        kh2 = torch.pow(k*dx, 2)
        # shift
        kh2_shift_absorb = torch.cat([-kh2 * self.beta1 + absorb * kh2 * self.beta2,
                                       kh2 * self.beta2 + absorb * kh2 * self.beta1], dim=1)
        Au = -lap_u + complex_mul(kh2_shift_absorb, u)

        return Au
class helm_op(nn.Module):
    def __init__(self, float_type=torch.float64):
        super().__init__()
        self.complex_type = torch.complex128 if float_type==torch.float64 else torch.complex64
        self.float_type = float_type


        self.lap_op = nn.Conv2d(2,2,kernel_size=3,stride=1,padding=1,groups=2,bias=False)
        kernel = torch.tensor([[[[0.,  1., 0.],
                                 [1., -4., 1.],
                                 [0.,  1., 0.]]]],dtype=float_type).repeat(2,1,1,1)
        self.lap_op.weight = nn.Parameter(kernel)
        self.lap_op.weight.requires_grad = False

        self.beta1 = 1.
        self.beta2 = 0.

    def forward(self, u, k, absorb, dx):
        '''
        input:
            u: float tensor, batch,2,N,N  real/imag
            k: float tensor, batch,1,N,N
        return:
            Au: complex tensor, batch,1,N,N
        '''
        if isinstance(u, int):
            return 0
        
        # u = torch.cat([u.real, u.imag], dim=1)#.to(torch.float32)
        lap_u = self.lap_op(u)
        kh2 = torch.pow(k*dx, 2)
        # shift
        kh2_shift_absorb = torch.cat([-kh2 * self.beta1 ,
                                       absorb * kh2 * self.beta1], dim=1)
        Au = -lap_u + complex_mul(kh2_shift_absorb, u)

        return Au
class mg_iteration(nn.Module):
    '''
    Performs a single multigrid iteration:
      u = u + B(b - Au)
    where B(b-Au) = linear_conv(nonlinear_conv(a)*linear_conv(b-Au))
    '''
    def __init__(self, A: shift_lap, weight: float):
        super().__init__()
        self.weight = weight
        self.A = A

    def D_inv(self, k, absorb, dx, rhs):
        """
        Applies the inverse of the diagonal of operator A to the residual rhs.
        The diagonal operator is approximated as a simple complex multiplication
        for simplicity and computational efficiency.
        """
        rhs_complex = torch.view_as_complex(rhs.permute(0, 2, 3, 1).contiguous())
        kh2 = torch.pow(k * dx, 2)
        
        d_real = 4.0 - kh2 * self.A.beta1 + absorb * kh2 * self.A.beta2
        d_imag = kh2 * self.A.beta2 + absorb * kh2 * self.A.beta1
        
        
        D = torch.complex(d_real, d_imag).squeeze(1)

        # 3. Apply inverse diagonal by complex division
        out = rhs_complex / D

        out = torch.view_as_real(out)
        
        return out.permute(0, 3, 1, 2)

    def forward(self, out):
        (u, k, absorb, dx, rhs, res) = out
        res = rhs - self.A(u, k, absorb, dx)
        # weighted jacobi
        u = u + self.weight * self.D_inv(k=k, absorb=absorb, dx=dx, rhs=res)
        return u, k, absorb, dx, rhs, res
    
class mg_restriction(nn.Module):
    '''
    Restricts rhs and initializes u to zeros at the coarser level.
    '''
    def __init__(self, ):
        super().__init__()
        self.R_rhs = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, 
                               stride=2, padding=0, groups=2, bias=False)
        kernel = torch.tensor([[1., 2., 1.],
                               [2., 4., 2.],
                               [1., 2., 1.]]) / 4.0
        kernel = kernel.view(1, 1, 3, 3).repeat(2, 1, 1, 1)
        self.R_rhs.weight.data.copy_(kernel)
        self.R_rhs.weight.requires_grad = False
    def forward(self, res):
        rhs = self.R_rhs(res)     
        return rhs
    
class mg_prolongation(nn.Module):
    '''
    Prolongates u from the coarser level.
    '''
    def __init__(self,):
        super().__init__()
        self.P_u = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=3, 
                                      stride=2, padding=0, groups=2, bias=False)
        kernel = torch.tensor([[1., 2., 1.],
                               [2., 4., 2.],
                               [1., 2., 1.]]) / 4.0
        kernel = kernel.view(1, 1, 3, 3).repeat(2, 1, 1, 1)
        self.P_u.weight.data.copy_(kernel)
        self.P_u.weight.requires_grad = False

    def forward(self, u):
        u = self.P_u(u)
        return u
    

class MG(nn.Module):
    '''
    Multigrid operator.
    Input: rhs, a
    Returns: mg(rhs,a) approximates inv_A * rhs,
    using different u_channels and a_channels for each level.
    '''
    def __init__(self, num_iteration=[1, 1, 1, 1, 1], fdtype=torch.float64):
        super().__init__()
        torch.set_default_dtype(fdtype)
        self.fdtype = fdtype
        self.cdtype = torch.complex128 if fdtype==torch.float64 else torch.complex64
        self.num_iteration = num_iteration
        jacobi_weight = 0.4
        self.k_list      = [None] * len(num_iteration)
        self.absorb_list = [None] * len(num_iteration)
        self.dx_list     = [None] * len(num_iteration)
        # restrict coeff
        self.R_coeff = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0, bias=False)
        kernel = torch.tensor([[1., 2., 1.],
                               [2., 4., 2.],
                               [1., 2., 1.]]) / 16.0
        kernel = kernel.view(1, 1, 3, 3)
        self.R_coeff.weight.data.copy_(kernel)
        self.R_coeff.weight.requires_grad = False
        self.A_op = shift_lap(float_type=self.fdtype)

        self.RT_list = nn.ModuleList([])
        self.R_list  = nn.ModuleList([])

        for j in range(len(num_iteration) - 1):
            self.R_list.append(mg_restriction())
            self.RT_list.append(mg_prolongation())

        # Setup cycle layers for down and up sweeps
        self.Down_sweep_layers = nn.ModuleList([])
        self.Up_sweep_layers   = nn.ModuleList([])
        for l, num_iteration_l in enumerate(num_iteration):
            down_layer_i = []
            up_layer_i   = []
            # for down layers
            for li in range(num_iteration_l):
                down_layer_i.append(mg_iteration(A=self.A_op, weight=jacobi_weight))
                    
            self.Down_sweep_layers.append(nn.Sequential(*down_layer_i))

            # for up layers 
            if l < len(num_iteration) - 1:
                for li in range(num_iteration_l):
                    up_layer_i.append(mg_iteration(A=self.A_op, weight=jacobi_weight))
                self.Up_sweep_layers.append(nn.Sequential(*up_layer_i))

    def setup(self, k, absorb, dx):
        # Ensure input tensors match the model's dtype
        k = k.to(self.fdtype)
        absorb = absorb.to(self.fdtype)

        self.k_list[0] = k
        self.absorb_list[0] = absorb
        self.dx_list[0] = dx

        for l in range(1, len(self.num_iteration)):
            k = self.R_coeff(k)
            absorb = self.R_coeff(absorb)
            dx = dx * 2  # Double dx for coarser grid

            self.k_list[l] = k
            self.absorb_list[l] = absorb
            self.dx_list[l] = dx

    def forward(self, rhs):
        # V cycle
        # rhs = torch.cat([rhs.real, rhs.imag], dim=1)
        out_list = [0] * len(self.num_iteration)
        
        # Make sure initial u is correctly typed and on the correct device
        u_initial = torch.zeros_like(rhs)

        for l in range(len(self.num_iteration)):
            u, _, _, _, rhs, res = self.Down_sweep_layers[l]((u_initial, self.k_list[l], self.absorb_list[l], self.dx_list[l], rhs, None))
            out_list[l] = (u, rhs)
            if l < len(self.num_iteration) - 1:
                rhs = self.R_list[l](res)
                u_initial = torch.zeros_like(rhs)


        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, rhs = out_list[j][0], out_list[j][1]
            u = u + self.RT_list[j](out_list[j+1][0])
            u, _, _, _, rhs, _ = self.Up_sweep_layers[j]((u, self.k_list[j], self.absorb_list[j], self.dx_list[j], rhs, None))
            out_list[j] = (u, rhs)
        out = out_list[0][0]   
        # out = torch.view_as_complex(out.permute(0, 2, 3, 1).contiguous())
        return out
    
    def multi_iter(self, rhs, iter_num=10):
        k, absorb, dx = self.k_list[0], self.absorb_list[0], self.dx_list[0]
        u = 0
        rhs = torch.cat([rhs.real, rhs.imag], dim=1)

        for i in range(iter_num):
            res = rhs - self.A_op(u, k, absorb, dx)
            u = u + self.forward(res)
        u = torch.view_as_complex(u.permute(0, 2, 3, 1).contiguous())
        return u


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("=== GMRES + Mixed Precision (FP32) Multigrid Shift Laplacian Solver ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # GMRES will use FP64
    dtype_fp64  = torch.float64
    cdtype_fp64 = torch.complex128
    
    # MG Preconditioner will use FP32
    dtype_fp32 = torch.float32
    cdtype_fp32 = torch.complex64

    
    dx = 1/256
    boundary_width = 32
    comp_size = 255
    domain_size = comp_size + boundary_width * 2
    lamb = 1.54e-5
    abc_factor = 1.
    
    print(f"Device: {device}")
    print(f"GMRES Precision: {dtype_fp64}")
    print(f"MG Preconditioner Precision: {dtype_fp32}")
    print(f"Domain size: {domain_size}x{domain_size}")
    print(f"Boundary width: {boundary_width}")
    
    sos_path = '/home/wangjt/work/datasets/openfwi/CurveFault_B/vel6_1_0.npy'
    # Load data with FP64 for high-precision setup
    sos_data = torch.from_numpy(np.load(sos_path)[:1]).to(device=device, dtype=dtype_fp64)

    sos_data = F.interpolate(sos_data, size=(comp_size, comp_size), mode='bilinear', align_corners=False)

    src_complex = torch.zeros((1, 1, comp_size, comp_size), device=device, dtype=cdtype_fp64)
    center = (domain_size-2*boundary_width) // 2
    src_complex[0, 0, center, center] = 1.0
    
    src_complex = F.pad(src_complex, (boundary_width,)*4, mode='constant', value=0)
    sos = F.pad(sos_data, (boundary_width,)*4, mode='replicate')
    k = 2 * np.pi / lamb / sos
    absorb = set_abc_layer(domain_size, boundary_width, abc_factor).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype_fp64)
    
    print(f"Source shape: {src_complex.shape} ({src_complex.dtype})")
    print(f"Wave number k shape: {k.shape} ({k.dtype})")
    print(f"Absorb shape: {absorb.shape} ({absorb.dtype})")
    
    # Operator A for GMRES uses FP64
    A = helm_op(float_type=dtype_fp64).to(device)
    
    # ===== 设置求解器 =====
    # MG preconditioner uses FP32
    print("\n--- Setting up FP32 Multigrid Preconditioner ---")
    mg = MG(num_iteration=[4, 4, 4, 4], fdtype=dtype_fp32).to(device)
    # Setup MG with FP32 versions of k and absorb
    mg.setup(k=k.to(dtype_fp32), absorb=absorb.to(dtype_fp32), dx=dx)
    
    # The A_func for GMRES operates in FP64
    def A_func(u_fp64):
        u_dual_fp64 = torch.cat([u_fp64.real, u_fp64.imag], dim=1)
        out_dual_fp64 = A(u_dual_fp64, k, absorb, dx)
        return torch.view_as_complex(out_dual_fp64.permute(0, 2, 3, 1).contiguous()).unsqueeze(1)
    
    # The M_func (preconditioner) handles the precision switch
    def M_func(res_fp64):
        # 1. Convert residual from FP64 to FP32
        res_fp32 = res_fp64.to(cdtype_fp32)
        res_dual_fp32 = torch.cat([res_fp32.real, res_fp32.imag], dim=1)
        
        # 2. Apply MG in FP32
        out_dual_fp32 = mg(res_dual_fp32)
        
        # 3. Convert result from FP32 back to FP64
        out_complex_fp32 = torch.view_as_complex(out_dual_fp32.permute(0, 2, 3, 1).contiguous())
        return out_complex_fp32.to(cdtype_fp64).unsqueeze(1)
    
    print("\n=== Warming up ===")
    for i in range(len(mg.num_iteration)):
        print(f"MG Level: {i}, shape: {list(mg.k_list[i].shape)}, dtype: {mg.k_list[i].dtype}")
    
    # Warmup M_func to compile kernels etc.
    with torch.no_grad():
        for i in range(5):
            _ = M_func(src_complex)
    
    print("\n=== Starting Shift Laplacian + GMRES solve (Mixed Precision) ===")
    iter_num = 2000
    tol = 1e-6

    if device.startswith('cuda'):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        start_evt.record()
    
    # GMRES runs in FP64, but calls M_func which internally uses FP32
    u_complex, iter_history = gmres(
        A=A_func, 
        b=src_complex,
        M=M_func, 
        x0=None,
        tol=tol, 
        maxiter=iter_num, 
        verbose=True
    )

    if device.startswith('cuda'):
        end_evt.record()
        torch.cuda.synchronize()
        solve_time = start_evt.elapsed_time(end_evt) / 1000.0
    else:
        solve_time = None
    
    print(f"\n=== Results ===")
    print(f"Converged in {len(iter_history)-1} iterations")
    final_residual = src_complex - A_func(u_complex)
    final_residual_norm = torch.norm(final_residual)
    src_norm = torch.norm(src_complex)
    print(f"Final residual: ||b - Au||/||b|| = {final_residual_norm/src_norm:.2e}")
    if solve_time:
        print(f"Solve time: {solve_time:.4f} seconds")
        
    u_np = u_complex.squeeze().cpu().numpy()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = ax1.imshow(sos_data.squeeze().cpu(), cmap='RdBu_r')
    ax1.set_title(f'speed of sound')
    plt.colorbar(im1, ax=ax1)
    vmax = np.max(np.real(u_np))*0.6
    im2 = ax2.imshow(np.real(u_np), interpolation='none', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax2.set_title(f'Wavefield - Real Part')
    plt.colorbar(im2, ax=ax2)
    im3 = ax3.semilogy(iter_history, 'b-o', linewidth=2, markersize=4)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Relative Residual')
    ax3.set_title('SL-Gmres Convergence History')
    ax3.grid(True)
    ax3.set_aspect('auto')
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.tight_layout()
    plt.savefig("SL_Gmres_out_mixed_precision.png")
    print("\nOutput image saved to SL_Gmres_out_mixed_precision.png")