import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.lifting_a = nn.Conv2d(3, num_features, kernel_size=3, padding=1, bias=True)
        self.proj_u    = nn.Conv2d(num_features, 1, kernel_size=1, bias=False)
        # self.ln_r = nn.LayerNorm([2, 128, 128], elementwise_affine=True)
        self.ln_a = nn.LayerNorm([3, self.N_pad, self.N_pad], elementwise_affine=True)
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
    def forward(self, a=None, rhs=None):
        '''
        a   : size=(batch, 1, N, N), dtype=float32
        rhs : size=(batch, 1, N, N), dtype=float32
        '''
        p = self.padding
        # --- padding ---
        a   = F.pad(a, (0, p, 0, p), mode='constant', value=0) 
        rhs = F.pad(rhs, (0, p, 0, p), mode='constant', value=0)

        # lifting
        a   = self.lifting_a(self.ln_a(a))
        rhs = self.lifting_r(rhs)

        # fno
        out = self.fno_linear(self.fno_nolinear(a) * self.fno_linear2(rhs))

        # projection
        out = self.proj_u(out)

        # --- un padding  ---
        out = out[..., :self.N, :self.N]

        return out.contiguous() # approx A_invb

class SpectralPDELoss(nn.Module):
    """
    Computes the physics-informed loss using spectral (FFT) derivatives.
    This corrected version computes the preconditioner G on a per-sample basis.
    """
    def __init__(self, N, dx, device='cpu', fdtype=torch.float32):
        super(SpectralPDELoss, self).__init__()
        self.N = torch.tensor(N, dtype=torch.int)
        self.dx = dx
        self.device = device
        self.fdtype = fdtype
        self.complex_dtype = torch.complex64 if fdtype == torch.float32 else torch.complex128
        
        # --- Fourier Grid Setup (remains the same) ---
        freq_x = 2 * torch.pi * torch.fft.fftfreq(N[0], d=dx)
        freq_y = 2 * torch.pi * torch.fft.fftfreq(N[1], d=dx)
        self.xi_x = freq_x.view(1, 1, -1, 1).to(device)
        self.xi_y = freq_y.view(1, 1, 1, -1).to(device)
        self.xi2 = self.xi_x**2 + self.xi_y**2
 
    # --- Derivative methods are correct and remain the same ---
    def _spectral_derivative(self, u, order, axis):
        u_hat = torch.fft.fftn(u, dim=[-2, -1])
        xi = self.xi_x if axis == 1 else self.xi_y
        deriv_hat = ((1j * xi)**order) * u_hat
        return torch.fft.ifftn(deriv_hat, dim=[-2, -1]).real

    def _spectral_laplacian(self, u):
        u_hat = torch.fft.fftn(u, dim=[-2, -1])
        lap_hat = (-self.xi2) * u_hat
        return torch.fft.ifftn(lap_hat, dim=[-2, -1]).real
    
    def _spectral_gradient(self, field):
        grad_y = self._spectral_derivative(field, order=1, axis=0)
        grad_x = self._spectral_derivative(field, order=1, axis=1)
        return grad_x, grad_y

    def apply_A_op(self, u, coefficients):
        """Helper function to apply the operator A."""
        kappa = coefficients[:, 0:1, :, :]
        vx = coefficients[:, 1:2, :, :]
        vy = coefficients[:, 2:3, :, :]
        sigma = 1.0

        grad_u_x, grad_u_y = self._spectral_gradient(u)
        lap_u = self._spectral_laplacian(u)
        grad_kappa_x, grad_kappa_y = self._spectral_gradient(kappa)

        term1 = -kappa * lap_u
        term2 = -(grad_kappa_x * grad_u_x + grad_kappa_y * grad_u_y)
        term3 = vx * grad_u_x + vy * grad_u_y
        term4 = sigma * u
        
        return term1 + term2 + term3 + term4

    def apply_G_op(self, field, G_hat):
        """Helper function to apply a pre-computed Green's function G."""
        field_hat = torch.fft.fftn(field.to(self.complex_dtype), dim=[-2,-1])
        return torch.fft.ifftn(G_hat * field_hat, dim=[-2,-1]).real

    def apply_G_op_outer(self, field,  coefficients):

        # CORRECTED: Compute L0 coefficients per-sample in the batch
        # The shape will be (B, 1, 1, 1) for broadcasting
        kappa0 = torch.mean(coefficients[:, 0:1, :, :], dim=(-2, -1), keepdim=True)
        vx0 = torch.mean(coefficients[:, 1:2, :, :], dim=(-2, -1), keepdim=True)
        vy0 = torch.mean(coefficients[:, 2:3, :, :], dim=(-2, -1), keepdim=True)
        
        sigma0 = 1.0
        eta = 1.0 # Damping term, as discussed in the papers

        # L̂₀(ξ) = κ₀|ξ|² + i(v₀⋅ξ) + (σ₀+η)
        # This now works for a batch, creating a G_hat of shape (B, 1, H, W)
        L0_hat = (kappa0 * self.xi2 +
                  1j * (vx0 * self.xi_x + vy0 * self.xi_y) +
                  (sigma0 + eta))
        
        G_hat = (1.0 / L0_hat).to(self.complex_dtype)
        field_hat = torch.fft.fftn(field.to(self.complex_dtype), dim=[-2,-1])
        return torch.fft.ifftn(G_hat * field_hat, dim=[-2,-1]).real

    def forward(self, u, coefficients, use_pre=False):
        f = coefficients[:, 3:4, :, :]
        
        # Calculate the residual of the original differential equation
        # Au = self.apply_A_op(u, coefficients)
        # residual = Au - f
        # If preconditioning is disabled, compute the standard relative loss
        if not use_pre:
            Au = self.apply_A_op(u, coefficients)
            residual = Au - f
            residual_norm = torch.norm(residual.view(residual.shape[0], -1), p=2, dim=1)
            f_norm = torch.norm(f.view(f.shape[0], -1), p=2, dim=1)
            loss = residual_norm / (f_norm + 1e-8)
            return loss.mean()

        # --- If preconditioning is enabled, compute G and the preconditioned loss ---
        
        # CORRECTED: Compute L0 coefficients per-sample in the batch
        # The shape will be (B, 1, 1, 1) for broadcasting
        kappa0 = torch.mean(coefficients[:, 0:1, :, :], dim=(-2, -1), keepdim=True)
        vx0 = torch.mean(coefficients[:, 1:2, :, :], dim=(-2, -1), keepdim=True)
        vy0 = torch.mean(coefficients[:, 2:3, :, :], dim=(-2, -1), keepdim=True)
        
        sigma0 = 1.0
        eta = 1.0 # Damping term, as discussed in the papers

        # L̂₀(ξ) = κ₀|ξ|² + i(v₀⋅ξ) + (σ₀+η)
        # This now works for a batch, creating a G_hat of shape (B, 1, H, W)
        L0_hat = (kappa0 * self.xi2 +
                  1j * (vx0 * self.xi_x + vy0 * self.xi_y) +
                  (sigma0 + eta))
        
        G_hat = (1.0 / L0_hat).to(self.complex_dtype)

        # Apply G to the residual and f
        r  = self.apply_A_op(u, coefficients) - f
        Gr = self.apply_G_op(r, G_hat)
        Gf = self.apply_G_op(f, G_hat)
        # preconditioned_residual = self.apply_G_op(residual, G_hat)
        # preconditioned_f = self.apply_G_op(f, G_hat)

        # The loss is ||G(Au - f)|| / ||G(f)||
        preconditioned_residual_norm = torch.norm((Gr).view(Gr.shape[0], -1), p=2, dim=1)
        preconditioned_f_norm = torch.norm(Gf.view(Gf.shape[0], -1), p=2, dim=1)

        loss = preconditioned_residual_norm / (preconditioned_f_norm + 1e-8)
        
        return loss.mean()
class FNO_solver(nn.Module):
    def __init__(self, N=63, dx=1/128, num_features=16, wavenumber=[40,40],padding=4, device='cpu', dtype=torch.float32, use_precond_res=True, use_precond_loss=True):
        super().__init__()
        # torch.set_default_dtype(torch.float32)
        self.dtype = dtype
        self.use_precond_res = use_precond_res
        self.use_precond_loss = use_precond_loss
        self.precond = Fourier_precond(N=N[0], num_features=num_features, wavenumber=wavenumber, padding=padding)
        self.pdeloss = SpectralPDELoss(N=N,dx=dx, device=device)

    def forward(self, u, coeffs):
        Au = self.pdeloss.apply_A_op(u, coeffs)
        b = coeffs[:, 3:4, ...]
        a = coeffs[:, 0:3, ...].to(torch.float32)
        res = (b - Au)
        if self.use_precond_res:
            res_ = self.pdeloss.apply_G_op_outer(res, coeffs).to(torch.float32)
        else:
            res_ = res.to(torch.float32)
        u = u + self.precond(a, res_).to(self.dtype)
        return u, res
    def train_forward(self, coeffs, rhs):
        a = coeffs[:, 0:3, ...].to(torch.float32)
        if self.use_precond_res:
            rhs = self.pdeloss.apply_G_op_outer((rhs), coeffs).to(torch.float32)
        else:
            rhs = rhs.to(torch.float32)
        out = self.precond(a, rhs)
        return out
    def loss_fn(self, u, coeffs, rhs):
        coeffs[:, 3:4, ...] = rhs
        return self.pdeloss(u, coeffs, use_pre=self.use_precond_loss)




    def solve(self, coeffs, iter_num=10, tol=1e-6):
        with torch.no_grad():
            b = coeffs[:, 3:4, ...]
            u = torch.zeros_like(b)
            
            norm_b = torch.norm(b)
            iterations_taken = 0
            # final_res_norm = float('inf')
            loss_his = []
            for i in range(iter_num):
                # self.forward 更新 du: du_{k+1} = du_k + P*(b - A*du_k)
                # 并返回新的残差 res = b - A*du_{k+1}
                u, res = self.forward(u, coeffs)
                
                # 计算残差的范数
                # .mean() 用于处理 batch, 如果 batch_size=1 则无影响
                res_norm = (torch.norm(res)/norm_b).item()
                iterations_taken = i + 1

                # 打印每一次迭代信息 (可选)
                print(f"Iter: {i+1}, error: {res_norm}")
                loss_his.append(res_norm)
                if res_norm < tol or res_norm > 100:
                    break
            
            # 返回最终解，最终残差范数，以及所用迭代次数
            return u, res_norm, loss_his