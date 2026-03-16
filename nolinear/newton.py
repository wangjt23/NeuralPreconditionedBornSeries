import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MgNO_DC_5, MgNO_DC_6
# from baselines.fno2d import FNO2d, SpectralDecoder
import os, logging
import numpy as np
import matplotlib.pyplot as plt

from utilities3 import *
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import argparse
torch.set_printoptions(threshold=100000)
# torch.set_default_tensor_type('torch.DoubleTensor')
"""
Fourier Preconditioner

"""
#fourier convolution 2d block
class fourier_conv_2d(nn.Module):
    def __init__(self, in_, out_, wavenumber1, wavenumber2):
        super(fourier_conv_2d, self).__init__()
        # torch.set_default_dtype(torch.float32)
        self.out_ = out_
        self.wavenumber1 = wavenumber1
        self.wavenumber2 = wavenumber2
        scale = (1 / (in_ * out_))
        self.weights1 = nn.Parameter(scale * torch.rand(in_, out_, wavenumber1, wavenumber2, 2 , dtype=torch.float32))
        self.weights2 = nn.Parameter(scale * torch.rand(in_, out_, wavenumber1, wavenumber2, 2 , dtype=torch.float32))
        # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ,2), (in_channel, out_channel, x,y,2) -> (batch, out_channel, x,y)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
    # def compl_mul2d(self, input, weights):
    #     input_c = torch.view_as_complex(input.contiguous())
    #     weights_c = torch.view_as_complex(weights.contiguous())
    #     out_c = torch.einsum("bixy,ioxy->boxy", input_c, weights_c)
    #     return torch.view_as_real(out_c)
    def forward(self, x):
        #input: batch,channel,x,y
        #out: batch,channel,x,y
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.view_as_real(torch.fft.rfft2(x))#input: batch,channel,x,y->batch,channel,x,y,2
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_,  x.size(-2), x.size(-1)//2 + 1,2, dtype=torch.float32, device=x.device)
        out_ft[:, :, :self.wavenumber1, :self.wavenumber2,:] = \
            self.compl_mul2d(x_ft[:, :, :self.wavenumber1, :self.wavenumber2,:], self.weights1)
        out_ft[:, :, -self.wavenumber1:, :self.wavenumber2,:] = \
            self.compl_mul2d(x_ft[:, :, -self.wavenumber1:, :self.wavenumber2,:], self.weights2)
        #Return to physical space
        x = torch.fft.irfft2(torch.view_as_complex(out_ft), s=(x.size(-2), x.size(-1)))
        return x

class NonLinear_Fourier_layer(nn.Module):
    def __init__(self,  features_, wavenumber):
        super().__init__()
        # torch.set_default_dtype(torch.float32)
        self.W =  nn.Conv2d(features_, features_, 1)
        self.fourier_conv = fourier_conv_2d(features_, features_ , *wavenumber)
        self.act = nn.GELU()
    def forward(self, x):
        x1 = self.fourier_conv(x)
        x2 = self.W(x)
        return self.act(x1 + x2) 
    
class Linear_Fourier_layer(nn.Module):
    def __init__(self,  features_, wavenumber):
        super().__init__()
        # torch.set_default_dtype(torch.float32)
        self.W =  nn.Conv2d(features_, features_, 1,bias=False)
        self.fourier_conv = fourier_conv_2d(features_, features_ , *wavenumber)
    def forward(self, x):
        x1 = self.fourier_conv(x)
        x2 = self.W(x)
        return  x1+ x2
    
class Fourier_precond(nn.Module):
    def __init__(self, N=63, num_features=16, wavenumber=[40,40], padding=4):
        super().__init__()
        torch.set_default_dtype(torch.float32)
        self.N = N
        self.num_features = num_features
        self.wavenumber   = wavenumber
        self.padding      = padding
        self.N_pad = N+padding
        # lifting&projection
        # self.lifting_u = nn.Conv2d(2, num_features, kernel_size=1, bias=True)
        self.lifting_r = nn.Conv2d(1, num_features, kernel_size=1, padding=0, bias=False)
        self.lifting_a = nn.Conv2d(1, num_features, kernel_size=3, padding=1, bias=True)
        self.proj_u    = nn.Conv2d(num_features, 1, kernel_size=1, bias=False)
        # self.ln_r = nn.LayerNorm([2, 128, 128], elementwise_affine=True)
        self.ln_a = nn.LayerNorm([1, self.N_pad, self.N_pad], elementwise_affine=True)
        # Fourier components
        # for a
        self.fno_nolinear = nn.ModuleList()
        for l in range(len(wavenumber)-1):
            self.fno_nolinear.append(NonLinear_Fourier_layer(features_=num_features, wavenumber=[wavenumber[l]]*2))
        self.fno_nolinear.append(Linear_Fourier_layer(features_=num_features, wavenumber=[wavenumber[-1]]*2))
        self.fno_nolinear =nn.Sequential(*self.fno_nolinear)

        # for residual
        lw = self.N_pad//2 # linear wavenumber
        self.fno_linear = Linear_Fourier_layer(features_=num_features, wavenumber=[lw, lw+1])
        self.fno_linear2 = Linear_Fourier_layer(features_=num_features, wavenumber=[lw, lw+1])
        
    # Au = residual
    def forward(self, u_current=None, rhs=None):
        '''
        a   : size=(batch, 1, N, N), dtype=float32
        rhs : size=(batch, 1, N, N), dtype=float32
        '''
        p = self.padding
        # --- padding ---
        # print(u_current.shape)
        a   = F.pad(u_current, (0, p, 0, p), mode='constant', value=0) 
        rhs = F.pad(rhs,       (0, p, 0, p), mode='constant', value=0)

        # lifting
        # print(a.shape)
        a   = self.lifting_a(self.ln_a(a))
        rhs = self.lifting_r(rhs)

        # fno
        out = self.fno_linear(self.fno_nolinear(a) * self.fno_linear2(rhs))

        # projection
        out = self.proj_u(out)

        # --- un padding  ---
        out = out[..., :self.N, :self.N]

        return out.contiguous() # approx A_invb


class Newton_FDM_op(nn.Module):

    #def create_rhs(N, h, s):
    """Create the right-hand side of the linear system."""
    # x = np.linspace(0, 1, N)
    # y = np.linspace(0, 1, N)
    # X, Y = np.meshgrid(x, y)
    # rhs = -s * np.sin(np.pi * X) * np.sin(np.pi * Y)
    # return rhs.flatten()
    def __init__(self, N=63, s=1600):
        super().__init__()
        self.h = 1.0 / (N + 1)
        kernel = torch.tensor([[[[0.,-1,0],[-1,4,-1],[0,-1,0]]]])/self.h**2
        self.register_buffer('kernel', kernel)
        # register the rhs as a buffer
        
        x = torch.linspace(0, 1, N)
        y = torch.linspace(0, 1, N)
        X, Y = torch.meshgrid(x, y)
        rhs = -s * torch.sin(np.pi * X) * torch.sin(np.pi * Y)
        self.register_buffer('rhs', rhs.reshape(N, N))
    
    def laplace_minus_u(self, u_current, du):
        
        return torch.nn.functional.conv2d(du, self.kernel, padding=1) - 2*du*u_current

    def getGrad(self, u_current):

        return self.rhs + u_current**2 - torch.nn.functional.conv2d(u_current, self.kernel, padding=1)

    def forward(self, u_current, du):
        # b = self.getGrad(u_current)
        # f = fixed_rhs + u_flat**2 - A.dot(u_flat)
        # loss = self.h * torch.linalg.norm((self.laplace_minus_u(u_current, du)-f).view(f.size(0), -1), dim=1)
        Au = self.laplace_minus_u(u_current, du)

        return Au
class Newton_FDM_precond_op(nn.Module):
    """
    A PDE loss function for a non-linear Poisson equation with a preconditioner.
    The loss is calculated as ||P(Au - b)||, where P is a DST-based
    preconditioner approximating the inverse of the Laplacian for Dirichlet BCs.
    """
    def __init__(self, reduction='sum', N=63, s=1600):
        super().__init__()
        self.N = N
        self.h = 1.0 / (self.N + 1)
        
        # Finite difference kernel for the Laplacian
        kernel = torch.tensor([[[[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]]]) / self.h**2
        kernel.requires_grad_(False)
        self.register_buffer('kernel', kernel)
        
        # Right-hand side (b) of the PDE
        x = torch.linspace(0, 1, self.N)  # To match the N points inside the domain
        y = torch.linspace(0, 1, self.N)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        rhs = -s * torch.sin(np.pi * X) * torch.sin(np.pi * Y)
        self.register_buffer('rhs', rhs)  # Shape (N, N)

        # Pre-calculate wavevectors squared (k^2) for the preconditioner (DISCRETE CASE)
        # For Dirichlet BCs, use DST with eigenvalues:
        # λ_{m,n} = (4/h²)[sin²(mπh/2) + sin²(nπh/2)]
        m = torch.arange(1, self.N + 1, device='cpu').float().view(-1, 1)
        n = torch.arange(1, self.N + 1, device='cpu').float().view(1, -1)
        self.K2 = (4.0 / self.h**2) * (torch.sin(m * np.pi * self.h / 2)**2 + 
                                      torch.sin(n * np.pi * self.h / 2)**2)
        self.register_buffer('K2_buffer', self.K2)

    def dst1d(self, x, dim=-1):
        """
        1D DST-I (un-normalized).
        """
        x = x.movedim(dim, -1)
        *batch, N = x.shape
        X = x.reshape(-1, N)

        ext = torch.zeros(X.size(0), 2*(N+1), device=X.device, dtype=X.dtype)
        ext[:, 1:N+1] = X
        ext[:, N+2:]  = -X.flip(1)

        F = torch.fft.rfft(ext, dim=1)           # (B, N+2)
        dst = -0.5 * F.imag[:, 1:N+1]       

        dst = dst.reshape(*batch, N).movedim(-1, dim)
        return dst
    
    def idst1d(self, x, dim=-1):
        """
        1D inverse DST-I (un-normalized).
        IDST-I = (2/(N+1)) * DST-I
        """
        # 直接调用 dst1d，再乘系数
        x = x.movedim(dim, -1)
        *batch, N = x.shape
        X = x.reshape(-1, N)

        # 逆变换公式
        idst = (2.0/(N+1)) * self.dst1d(X, dim=-1)

        idst = idst.reshape(*batch, N).movedim(-1, dim)
        return idst
    
    def dst2d(self, x):
        return self.dst1d(self.dst1d(x, dim=-1), dim=-2)
    
    def idst2d(self, x):
        return self.idst1d(self.idst1d(x, dim=-2), dim=-1)
    
    def laplace_minus_u(self, u_current, du):
        """Calculates the action of the Jacobian on the update du."""
        # A(u) * du where A is the Jacobian of the residual
        # A(u) = (laplace - 2*u)
        return torch.nn.functional.conv2d(du, self.kernel, padding=1) - 2 * du * u_current

    def getGrad(self, u_current):
        """Calculates the residual (or gradient of the energy), Au - b."""
        # Here we calculate R(u) = laplace(u) - u^2 - rhs
        laplace_u = torch.nn.functional.conv2d(u_current, self.kernel, padding=1)
        return -laplace_u + u_current**2 + self.rhs

    def apply_preconditioner(self, x, du=None, u_current=None):
        """
        Applies a DST-based preconditioner that accounts for the -2u term.
        P(x) = IDST(1/( -K2 - 2*u_avg ) * DST(x))
        """
        epsilon = 1e-10
        
        # Estimate the effect of -2u term using the average value
        if u_current is not None:
            u_avg = 14.
        else:
            u_avg = 0.0
        
        # Jacobian eigenvalues: Δ - 2u ≈ λ_mn - 2*u_avg
        jacobian_eigenvalues = self.K2_buffer.to(x.device) #- 2 * u_avg
        # print(jacobian_eigenvalues.max())
        # print(jacobian_eigenvalues.min())
        # exit()
        # Avoid division by zero
        preconditioner_freq = 1.0 / (jacobian_eigenvalues + epsilon)
        
        # Reshape if needed
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        
        # Apply DST preconditioner
        x_dst = self.dst2d(x)
        preconditioned_x_dst = x_dst * preconditioner_freq
        preconditioned_x = self.idst2d(preconditioned_x_dst)
        
        # Restore channel dimension
        if len(preconditioned_x.shape) == 3:
            preconditioned_x = preconditioned_x.unsqueeze(1)
            
        return preconditioned_x

    def forward(self, u_current, du):
        """
        Calculates the preconditioned loss.
        Loss = ||P * (Jacobian(u) * du - Grad(u))||
        """
        f = self.getGrad(u_current)
        residual_newton = self.laplace_minus_u(u_current, du) - f
        
        # Apply the DST-based preconditioner
        residual_newton = self.apply_preconditioner(residual_newton, u_current=u_current)
        
        return residual_newton
class FNO_solver(nn.Module):
    def __init__(self, N=63, s=1600, num_features=16, wavenumber=[40,40], padding=4, use_precond_res=False, use_l2_loss=False):
        super().__init__()
        torch.set_default_dtype(torch.float32)
        self.precond = Fourier_precond(N=N, num_features=num_features, wavenumber=wavenumber, padding=padding)
        self.A_op = Newton_FDM_op(N=N,s=s)
        self.A_precond_op = Newton_FDM_precond_op(N=N,s=s)
        self.use_precond_res = use_precond_res
        self.use_l2_loss = use_l2_loss

    def forward(self, u_current, du):
        Au = self.A_op(u_current, du)
        b  = self.A_op.getGrad(u_current)
        res = b - Au
        if self.use_precond_res:
            res_ = self.A_precond_op.apply_preconditioner(res, u_current=u_current)
        else:
            res_ = res
        du = du + self.precond(u_current, res_)

        return du, res
    def train_forward(self, u_current, rhs):
        res = rhs
        if self.use_precond_res:
            res = self.A_precond_op.apply_preconditioner(res, u_current=u_current)
        out = self.precond(u_current, res)
        return out
    def loss_fn(self, u_current, du, rhs):
        Au = self.A_op(u_current, du)
        loss = Au - rhs
        
        return torch.norm(loss)/torch.norm(rhs)
    def loss_fn_precond(self, u_current, du, rhs):
        Au = self.A_op(u_current, du)
        loss = Au - rhs
        loss = self.A_precond_op.apply_preconditioner(loss,u_current=u_current)
        norm_b = torch.norm(self.A_precond_op.apply_preconditioner(rhs))
        # norm_b = torch.norm(rhs)
        return torch.norm(loss)/norm_b
    
    def loss_for_NPBS(self, u_current, du, rhs):
        # NPBS+$\mathcal{L}_{\mathrm{bs}}^{\ell_2}$
        if self.use_l2_loss:
            return self.loss_fn( u_current, du, rhs)
        # NPBS+$\mathcal{L}_{\mathrm{bs}}^{R_\eta}$
        else:
            return self.loss_fn_precond( u_current, du, rhs)



    def solve(self, u_current, iter_num=10, tol=1e-6):
        with torch.no_grad():
            du = torch.zeros_like(u_current)
            b = self.A_op.getGrad(u_current)
            norm_b = torch.norm(b)
            # iterations_taken = 0
            iter_his = [1.]
            final_res_norm = float('inf')

            for i in range(iter_num):
                # self.forward 更新 du: du_{k+1} = du_k + P*(b - A*du_k)
                # 并返回新的残差 res = b - A*du_{k+1}
                du, res = self.forward(u_current, du)
                
                # 计算残差的范数
                # .mean() 用于处理 batch, 如果 batch_size=1 则无影响
                res_norm = torch.norm(res)/norm_b
                final_res_norm = res_norm.item()
                # iterations_taken = i + 1
                iter_his.append(final_res_norm)
                # 打印每一次迭代信息 (可选)
                print(f"Iter: {i+1}, error: {final_res_norm}")

                if final_res_norm < tol:
                    break
            
            # 返回最终解，最终残差范数，以及所用迭代次数
            return du, final_res_norm, iter_his

def objective(dataOpt, modelOpt, optimizerScheduler_args,
                tqdm_disable=True, 
              log_if=False, validate=False, model_type='MgNO_DC_5', 
              model_save=False, tune_if=False, test_mode=False,):
    
    ################################################################
    # configs
    ################################################################

    print(os.path.basename(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = getSavePath(dataOpt['data'], model_type)
    MODEL_PATH_PARA = getSavePath(dataOpt['data'], model_type, flag='para')
    if log_if:
        logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename=MODEL_PATH,
                        filemode='w')
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info(model_type)
        logging.info(f"dataOpt={dataOpt}")
        logging.info(f"modelOpt={modelOpt}")
        logging.info(f"optimizerScheduler_args={optimizerScheduler_args}")
  
    
    ################################################################
    # load data and data normalization
    ################################################################
    u_list, delta_u_list = torch.load('./data/newton_method_iterations_5.pt', weights_only=False)
    u_list_2, delta_u_list_2, solution = torch.load('./data/newton_method_iterations_7_solution.pt', weights_only=False)
    V = torch.load('./data/V.pt', weights_only=False)
    V = V.float().contiguous().to(device)
    y = delta_u_list.float().to(device).squeeze()
    x = u_list.to(device).float()
    y_2 = delta_u_list_2.float().to(device).squeeze()
    x_2 = u_list_2.to(device).float()
    if dataOpt['loss_type'] == 'pde':
        y = y.unsqueeze(1)
        y_2 = y_2.unsqueeze(1)
    # y = y/y.max()
    # x = x/x.max()
    dataOpt['dataSize'] = {'train': range(5000), 'test': range(5000, 5500), 'val':range(600,650)}

    x_train = x[dataOpt['dataSize']['train'],...]
    y_train = y[dataOpt['dataSize']['train'],...]
    x_test = x[dataOpt['dataSize']['test'],...]
    y_test = y[dataOpt['dataSize']['test'],...]
    x_val = x[dataOpt['dataSize']['val'],...]
    y_val = y[dataOpt['dataSize']['val'],...]

    x_train_2 = x_2[dataOpt['dataSize']['train'],...]
    y_train_2 = y_2[dataOpt['dataSize']['train'],...]
    x_test_2 = x_2[dataOpt['dataSize']['test'],...]
    y_test_2 = y_2[dataOpt['dataSize']['test'],...]

    x_train = torch.cat((x_train, x_train_2), dim=0)
    y_train = torch.cat((y_train, y_train_2), dim=0)
    x_test = torch.cat((x_test, x_test_2), dim=0)
    y_test = torch.cat((y_test, y_test_2), dim=0)
    

    if dataOpt['GN']:
        x_normalizer = GaussianNormalizer(x_train) #UnitGaussianNormalizer
        # x_train = x_normalizer.encode(x_train)
        # x_test = x_normalizer.encode(x_test)

    if model_type == 'FNO':
        x_train = x_train.squeeze()
        x_test = x_test.squeeze()
        x_train = x_train[:, ..., np.newaxis]
        x_test = x_test[:, ..., np.newaxis]   
  


    y_normalizer = GaussianNormalizer(y_train)
   
    # plot the data
    
    # visual2d(x_train[0, ..., 0].cpu().numpy(), '1')
    # visual2d(x_train[1, ..., 0].cpu().numpy(), '2')
    # visual2d(x_train[2, ..., 0].cpu().numpy(), '3')
    # visual2d(x_train[3, ..., 0].cpu().numpy(), '4')
    # visual2d(y_train[0, ...].cpu().numpy(), '5')
    # visual2d(y_train[1, ...].cpu().numpy(), '6')
    # visual2d(y_train[2, ...].cpu().numpy(), '7')
    # visual2d(y_train[3, ...].cpu().numpy(), '8')
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    def resize_to_64(tensor):
        return F.interpolate(tensor, size=(64, 64), mode='bilinear', align_corners=False)

    x_train = resize_to_64(x_train)
    y_train = resize_to_64(y_train.unsqueeze(1))
    x_test  = resize_to_64(x_test)
    y_test  = resize_to_64(y_test.unsqueeze(1))
    train_loader = DataLoader(TensorDataset(x_train.contiguous(), y_train.contiguous()), batch_size=dataOpt['batch_size'], shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test.contiguous(), y_test.contiguous()), batch_size=dataOpt['batch_size'], shuffle=False)
    
    train_loader_l2 = DataLoader(TensorDataset(x_train[:5000,...].contiguous(), y_train[:5000,...].contiguous()), batch_size=dataOpt['batch_size'], shuffle=True)
    # train_loader = DataLoader(TensorDataset(x_train.contiguous().to(device), y_train.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=True)
    # test_loader = DataLoader(TensorDataset(x_test.contiguous().to(device), y_test.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=False)
    if validate:
        val_loader = DataLoader(TensorDataset(x_val.contiguous().to(device), y_val.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=False)

    # load the model and evaluate
    if test_mode:
        model = torch.load(dataOpt['MODEL_PATH_LOAD'])
        model.eval()
        l2loss = LpLoss(size_average=False) 
        test_l2 = 0.
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_l2 += l2loss(out, y).item()
                
     
        test_l2/= len(dataOpt['dataSize']['test'])
   
        logging.info(f" test l2 loss: {test_l2:.3e}")
        logging.info(model)      
        return test_l2
    
    ################################################################
    # training and evaluation
    ################################################################
    modelOpt['normalizer'] = y_normalizer
    
    if model_type == 'MgNO_DC_6':
        model = MgNO_DC_6(**modelOpt).to(device)
    # elif model_type == 'FNO':
    #     # model = FNO2d(modes1=12, modes2=12, width=80, normalizer=y_normalizer).to(device)
    #     model = SpectralDecoder(lift=1, normalizer=y_normalizer, modes=12, num_spectral_layers=5, width=64, init_scale=2).to(device)
    elif model_type == "FNO_solver":
        model = FNO_solver(N=64, s=1600., num_features=20, wavenumber=[40, 40], padding=64).to(device)
    elif model_type == "FNO_solver_precond_bs_l_2":
        model = FNO_solver(N=64, s=1600., num_features=20, wavenumber=[40, 40], padding=64,
                           use_precond_res=True,use_l2_loss=True).to(device)
    elif model_type == "FNO_solver_precond_bs_R_eta":
        model = FNO_solver(N=64, s=1600., num_features=20, wavenumber=[40, 40], padding=64,
                           use_precond_res=True,use_l2_loss=False).to(device)
    else:
        raise NameError('invalid model_type')
    
    if log_if:    
        logging.info(count_params(model))
        logging.info(model)
    optimizer, scheduler = getOptimizerScheduler(model.parameters(), **optimizerScheduler_args)
    
    h1loss = HsLoss(d=2, p=2, k=1, a=dataOpt['loss_weight'], size_average=False, res=y_train.size(2), relative=False)
    h1loss.cuda(device)

    h1loss = HsLoss_2(d=2, p=2, k=2, size_average=False, res=y_train.size(2), relative=False).to(device)
  
    l2loss = LpLoss(size_average=False, relative=False )   #
    if dataOpt['loss_type']=='pde':
        pde_loss = PDEloss_2().to(device)
        # pde_loss = PDEloss_2_preconditioned().to(device)
        # pde_loss = PDEloss_2_preconditioned_ILU().to(device)
    elif dataOpt['loss_type'] == 'solver':
        pass
    l2loss_rel = LpLoss(size_average=False, relative=True )
    ############################
    def train(train_loader):
        model.train()
        train_l2, train_h1 = 0, 0
        # train_f_dist = torch.zeros(y_train.size(1))

        for batch_idx, (x, y) in enumerate(train_loader):# x: u_current, y: du
           
            x = x.to(device)
            y = y.to(device)
            # print(y.shape)
            optimizer.zero_grad()
            if dataOpt['loss_type'] in {'l2', 'h1'}:
                out = model(x).squeeze()
            elif dataOpt['loss_type']=='pde':
                if dataOpt['GN']:
                    # grad = pde_loss.getGrad(x)
                    # x_en = x_normalizer.encode(torch.cat((x, grad), dim=1))
                    x_en = x_normalizer.encode(x)
                    out = model(x_en)
                else:
                    out = model(x)
            elif dataOpt['loss_type']=='solver':
                # print(torch.norm(model.A_op.getGrad(x)))
                # print(torch.norm(torch.randn_like(x)))
                # exit()
                if np.random.rand()<0:
                    rhs = model.A_op.getGrad(x)
                else:
                    rhs = torch.randn_like(x)
                out = model.train_forward(x, rhs)

            if dataOpt['loss_type']=='h1':
                with torch.no_grad():
                    train_l2loss = l2loss(out, y)

                train_h1loss = h1loss(out, y)
                train_h1loss.backward()
            elif dataOpt['loss_type']=='l2':
                with torch.no_grad():
                    train_h1loss = h1loss(out, y)

                train_l2loss = l2loss(out, y)
                train_l2loss.backward()
            elif dataOpt['loss_type']=='pde':
                with torch.no_grad():
                    train_l2loss = l2loss(out, y)

                train_h1loss = pde_loss(x, out)
                train_h1loss.backward()              
            elif dataOpt['loss_type']=='solver':
                with torch.no_grad():
                    if model_type in ("FNO_solver_precond_bs_l_2","FNO_solver_precond_bs_R_eta"):
                        train_l2loss = model.loss_fn(x, out, rhs)
                    else:
                        train_l2loss = model.loss_fn(x, out, rhs)

                if model_type in ("FNO_solver_precond_bs_l_2","FNO_solver_precond_bs_R_eta"):
                    train_h1loss = model.loss_for_NPBS(x, out, rhs)
                else:
                    train_h1loss = model.loss_fn(x, out, rhs)
                train_h1loss.backward()        
            optimizer.step()
            train_h1 += train_h1loss.item()
            train_l2 += train_l2loss.item()
            # train_f_dist += sum(torch.squeeze(torch.abs(f_l2x-f_l2y))).cpu()

        train_l2/= len(dataOpt['dataSize']['train'])
        train_h1/= len(dataOpt['dataSize']['train'])
        # train_f_dist/= len(dataOpt['dataSize']['train'])
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
         
        return lr, train_l2, train_h1
            
    @torch.no_grad()
    def test(test_loader):
        model.eval()
        test_l2, test_h1 = 0., 0.
        # test_f_dist = torch.zeros(y_test.size(1))
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                if dataOpt['loss_type'] in {'l2', 'h1'}:
                    out = model(x).squeeze()
                elif dataOpt['loss_type']=='pde':
                    if dataOpt['GN']:
                        # grad = pde_loss.getGrad(x)
                        # x_en = x_normalizer.encode(torch.cat((x, grad), dim=1))
                        x_en = x_normalizer.encode(x)
                        out = model(x_en)
                    else:
                        out = model(x)
                elif dataOpt['loss_type']=='solver':
                    if 1:#np.random.rand()<0.3:
                        rhs = model.A_op.getGrad(x)
                    else:
                        rhs = torch.randn_like(x)
                    out = model.train_forward(x, rhs)
                
                if dataOpt['loss_type']=='solver':
                    if model_type in ("FNO_solver_precond_bs_l_2","FNO_solver_precond_bs_R_eta"):
                        test_h1loss = model.loss_for_NPBS(x, out, rhs)
                        test_l2 += model.loss_fn(x, out, rhs).item()
                    else:
                        test_h1loss = model.loss_fn(x, out, rhs)
                        test_l2 += model.loss_fn(x, out, rhs).item()
                else:
                    test_h1loss = h1loss(out, y)
                    test_l2 += l2loss(out, y).item()
                test_h1 += test_h1loss.item()
                # test_f_dist += sum(torch.squeeze(torch.abs(f_l2x-f_l2y))).cpu()
                
        test_l2/= len(dataOpt['dataSize']['test'])
        test_h1/= len(dataOpt['dataSize']['test'] )          
        # test_f_dist/= len(dataOpt['dataSize']['test'] ) 

        return  test_l2, test_h1 #, test_f_dist

        
    ############################  
    ###### start to train ######
    ############################
    
    train_h1_rec, train_l2_rec, test_l2_rec, test_h1_rec = [], [], [], []
    if validate:
        val_l2_rec, val_h1_rec = [], [],
    
    best_l2, best_test_l2, best_test_h1, arg_min_epoch = 1.0, 1.0, 1.0, 0  
    with tqdm(total=optimizerScheduler_args['epochs']*2, disable=tqdm_disable) as pbar_ep:
                            
        # for epoch in range(optimizerScheduler_args['epochs']):
        #     desc = f"epoch: [{epoch+1}/{optimizerScheduler_args['epochs']}]"
        #     dataOpt['loss_type'] = 'l2'
        #     lr, train_l2, train_h1 = train(train_loader_l2)
            
        #     test_l2, test_h1 = test(test_loader)
        #     if validate:
        #         val_l2, val_h1 = test(val_loader)
            
        #     train_l2_rec.append(train_l2); train_h1_rec.append(train_h1); 
        #     test_l2_rec.append(test_l2); test_h1_rec.append(test_h1); 
        #     if validate:
        #         val_l2_rec.append(val_l2); val_h1_rec.append(val_h1)
        #     if validate:
        #         if val_l2 < best_l2:
        #             best_l2 = val_l2
        #             arg_min_epoch = epoch
        #             best_test_l2 = test_l2
        #             best_test_h1 = test_h1
           
        #     desc += f" | current lr: {lr:.3e}"
        #     desc += f"| train l2 loss: {train_l2:.3e} "
        #     desc += f"| train h1 loss: {train_h1:.3e} "
        #     desc += f"| test l2 loss: {test_l2:.3e} "
        #     desc += f"| test h1 loss: {test_h1:.3e} "
        #     if validate:
        #         desc += f"| val l2 loss: {val_l2:.3e} "
        #         desc += f"| val h1 loss: {val_h1:.3e} "
           
        #     pbar_ep.set_description(desc)
        #     pbar_ep.update()
        #     if log_if:
        #         logging.info(desc) 


        # if log_if and validate: 
        #     logging.info(f" test h1 loss: {best_test_h1:.3e}, test l2 loss: {best_test_l2:.3e}")
                 
        # if log_if:
        #     logging.info('train l2 rec:')
        #     logging.info(train_l2_rec)
        #     logging.info('train h1 rec:')
        #     logging.info(train_h1_rec)
        #     logging.info('test l2 rec:')
        #     logging.info(test_l2_rec)
        #     logging.info('test h1 rec:')
        #     logging.info(test_h1_rec)
        
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizerScheduler_args['lr'],
        #                        div_factor=2, 
        #                        final_div_factor=2e1,
        #                        pct_start=0.1,
        #                        steps_per_epoch=1, 
        #                        epochs=optimizerScheduler_args['epochs'])

        for epoch in range(optimizerScheduler_args['epochs']):
            desc = f"epoch: [{epoch+1}/{optimizerScheduler_args['epochs']}]"
            
            # dataOpt['loss_type'] = 'pde'
            lr, train_l2, train_h1 = train(train_loader)
            test_l2, test_h1 = test(test_loader)
            if validate:
                val_l2, val_h1 = test(val_loader)
            
            train_l2_rec.append(train_l2); train_h1_rec.append(train_h1); 
            test_l2_rec.append(test_l2); test_h1_rec.append(test_h1); 
            if validate:
                val_l2_rec.append(val_l2); val_h1_rec.append(val_h1)
            if validate:
                if val_l2 < best_l2:
                    best_l2 = val_l2
                    arg_min_epoch = epoch
                    best_test_l2 = test_l2
                    best_test_h1 = test_h1
           
            desc += f" | current lr: {lr:.3e}"
            desc += f"| train aux l2 loss: {train_l2:.3e} "
            desc += f"| train loss: {train_h1:.3e} "
            desc += f"| test aux l2 loss: {test_l2:.3e} "
            desc += f"| test loss: {test_h1:.3e} "
            if validate:
                desc += f"| val aux l2 loss: {val_l2:.3e} "
                desc += f"| val loss: {val_h1:.3e} "
           
            pbar_ep.set_description(desc)
            pbar_ep.update()
            if log_if:
                logging.info(desc) 
    # plot the l2 convergence history
    if log_if:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].plot(train_l2_rec, label='train_aux')
        ax[0].plot(test_l2_rec, label='test_aux')
        ax[0].set_yscale('log')
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('aux loss')
        ax[0].legend()
        ax[0].grid(True)
        plt.savefig(MODEL_PATH+'.png')
        plt.close()
   
    
    if model_save:
        torch.save(model, MODEL_PATH_PARA)
            
    return test_l2


# def count_parameters(model):
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return total_params, trainable_params
# if __name__ == "__main__":
#     # # model    = MG_precond(num_features=20, add_channels=5, mg_levels=[1,1,1,1,1])
#     # model = MG_fem(in_channels=20,add_channels=5,out_channels=20,num_iteration=[1,1,1,1,1])
#     # # model    = Fourier_precond(num_features=16, wavenumber=[40,40])
#     # V        = torch.rand(2,20,128,128,dtype=torch.double)
#     # residual = torch.rand(2,20,128,128,dtype=torch.double)
#     # u = residual
#     # out = model(u,residual,V)
#     # print(model)

#     # model = Fourier_precond(20,[64,64])
#     # model = MG_precond(num_features=20,add_channels=15,mg_levels=[1,1,1,1])
#     N = 63
#     # model = Fourier_precond(N=N, num_features=16, wavenumber=[40,40], padding=65)
#     model = MG_precond(N=N,u_channels_list=[5,10,15], a_channels_list=[10,15,20], mg_levels=[1,2,4])
#     # model = MG_precond2(num_features=30, sweeps=2, levels=4)
#     print(model)
#     u_curr = torch.rand(4,1,N,N,dtype=torch.float32)

#     x = torch.ones(4,1,N,N,dtype=torch.float32)
#     y = torch.rand(4,1,N,N,dtype=torch.float32)
#     result1 = model(u_current=u_curr,rhs=(x+y)/10)
#     result2 = model(u_current=u_curr, rhs=y)
#     result3 = model(u_current=u_curr, rhs=x)
#     is_linear = torch.allclose(result1, (result2+result3)/10, atol=1e-7)
#     print("linear to the residual term? ",is_linear)
#     total_params, trainable_params = count_parameters(model)
#     print(f"Total parameters: {total_params}, size: {total_params*4/(1024**2)} Mb")


if __name__ == "__main__":

    import newton
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--data", type=str, default="ns_merge", help="data name, darcy, darcy20c6, darcy15c10, darcyF, darcy_contin")
    parser.add_argument(
            "--model_type", type=str, default="DeepONet", help="model type")
    parser.add_argument(
            "--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument(
            "--batch_size", type=int, default=10, help="batch size")
    parser.add_argument(
            "--optimizer_type", type=str, default="adam", help="optimizer type")
    parser.add_argument(
            "--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument(
            "--final_div_factor", type=float, default=10, help="final_div_factor")
    parser.add_argument(
            "--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
            "--loss_type", type=str, default="pde", help="loss type, l2, h1, pde")
    parser.add_argument(
            "--GN", action='store_true', help="use normalized x")
    parser.add_argument(
            "--sample_x", action='store_true', help="sample x")
    parser.add_argument(
            "--sampling_rate", type=int, default=1, help="sampling rate")
    parser.add_argument(
            "--normalizer", action='store_true', help="use normalizer")
    parser.add_argument(
            "--normalizer_type", type=str, default="GN", help="PGN, GN")
    parser.add_argument(
            "--num_layer", type=int, default=5, help="number of layers")
    parser.add_argument(
            "--num_channel_u", type=int, default=24, help="number of channels for u")
    parser.add_argument(
            "--num_channel_f", type=int, default=1, help="number of channels for f")
    parser.add_argument(
            '--num_iteration', type=list, nargs='+', default=[[1,0], [1,0], [1,0], [1,1], [2,0]], help='number of iterations in each layer')
    parser.add_argument(
            '--padding_mode', type=str, default='zeros', help='padding mode')
    parser.add_argument(
            '--last_layer', type=str, default='linear', help='last layer type')

    parser.add_argument(
            "--test", action='store_true', help="load model and test")
    parser.add_argument(
            "--MODEL_PATH_LOAD", type=str, default="PATH_LOAD", help="PATH_LOAD")
    args = parser.parse_args()
    args = vars(args)

    for i in range(len(args['num_iteration'])):
        for j in range(len(args['num_iteration'][i])):
            args['num_iteration'][i][j] = int(args['num_iteration'][i][j])
        

    if  args['sample_x']:
        if args['data'] in {'darcy', 'darcy20c6', 'darcy15c10', 'darcyF', 'darcy_contin'}:
            args['sampling_rate'] = 2
        elif args['data'] == 'a4f1':
            args['sampling_rate'] = 4
        elif args['data'] == 'helm':
            args['sampling_rate'] = 1
        elif args['data'] == 'pipe':
            args['sampling_rate'] = 1


    
  
        
    dataOpt = {}
    dataOpt['data'] = args['data']
    dataOpt['sampling_rate'] = args['sampling_rate']
    dataOpt['sample_x'] = args['sample_x']
    dataOpt['batch_size'] = args['batch_size']
    dataOpt['loss_type']=args['loss_type']
    dataOpt['loss_weight'] = [2,]
    dataOpt['normalizer_type'] = args['normalizer_type']
    dataOpt['GN'] = args['GN']
    dataOpt['MODEL_PATH_LOAD'] = args['MODEL_PATH_LOAD']
    dataOpt = getDataSize(dataOpt)

    modelOpt = {}
    modelOpt['num_layer'] = args['num_layer']
    modelOpt['num_channel_u'] = args['num_channel_u']
    modelOpt['num_channel_f'] = args['num_channel_f']
    modelOpt['num_classes'] = 1
    modelOpt['num_iteration'] = args['num_iteration']
    modelOpt['in_chans'] = 1
    modelOpt['normalizer'] = args['normalizer'] 
    modelOpt['output_dim'] = 1
    modelOpt['activation'] = 'gelu'    
    modelOpt['padding_mode'] = args['padding_mode']
    modelOpt['last_layer'] = args['last_layer']

    optimizerScheduler_args = {}
    optimizerScheduler_args['optimizer_type'] = args['optimizer_type']
    optimizerScheduler_args['lr'] = args['lr']
    optimizerScheduler_args['weight_decay'] = args['weight_decay']
    optimizerScheduler_args['epochs'] = args['epochs']
    optimizerScheduler_args['final_div_factor'] = args['final_div_factor']
    optimizerScheduler_args['div_factor'] = 2

    newton.objective(dataOpt, modelOpt, optimizerScheduler_args, model_type=args['model_type'],
    validate=False, tqdm_disable=True, log_if=True, 
    model_save=True, test_mode=args['test'])





    

    