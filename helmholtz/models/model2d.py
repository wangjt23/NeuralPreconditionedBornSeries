from ._precision import preserve_default_dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
from .optional import load_mgcfnn_model

# ---------------------------------------------------------------------------
# MgNO building blocks (ported from repos/MgNO-main/models.py)
# Reference: He et al., "MgNO: Efficient Parameterization of Linear Operators
# via Multigrid", ICLR 2024.
# ---------------------------------------------------------------------------

class _MgIte(nn.Module):
    def __init__(self, A, S):
        super().__init__()
        self.A = A
        self.S = S

    def forward(self, out):
        if isinstance(out, tuple):
            u, f = out
            u = u + self.S(f - self.A(u))
        else:
            f = out
            u = self.S(f)
        return (u, f)


class _MgIte_init(nn.Module):
    def __init__(self, S):
        super().__init__()
        self.S = S

    def forward(self, f):
        return (self.S(f), f)


class _Restrict(nn.Module):
    def __init__(self, Pi=None, R=None, A=None):
        super().__init__()
        self.Pi = Pi
        self.R = R
        self.A = A

    def forward(self, out):
        u, f = out
        f = self.R(f - self.A(u)) if self.A is not None else self.R(f)
        return (self.Pi(u), f)


class _MgConv_DC(nn.Module):
    """V-cycle MgConv block without LayerNorm; works with any spatial size."""

    def __init__(self, num_iteration, num_channel_u, num_channel_f,
                 padding_mode='zeros', bias=False):
        super().__init__()
        self.num_iteration = num_iteration
        self.RTlayers = nn.ModuleList([
            nn.ConvTranspose2d(num_channel_u, num_channel_u,
                               kernel_size=4, stride=2, padding=1, bias=False)
            for _ in range(len(num_iteration) - 1)
        ])
        layers = []
        for l, (n_pre, n_post) in enumerate(num_iteration):
            post_smooth = []
            for i in range(n_pre):
                S = nn.Conv2d(num_channel_f, num_channel_u, 3, padding=1,
                              bias=bias, padding_mode=padding_mode)
                layers.append(_MgIte_init(S) if (l == 0 and i == 0) else
                               _MgIte(nn.Conv2d(num_channel_u, num_channel_f, 3, padding=1,
                                                bias=bias, padding_mode=padding_mode), S))
            if n_post:
                for _ in range(n_post):
                    S = nn.Conv2d(num_channel_f, num_channel_u, 3, padding=1,
                                  bias=bias, padding_mode=padding_mode)
                    A = nn.Conv2d(num_channel_u, num_channel_f, 3, padding=1,
                                  bias=bias, padding_mode=padding_mode)
                    post_smooth.append(_MgIte(A, S))
            else:
                post_smooth.append(nn.Identity())
            setattr(self, f'layer{l}', nn.Sequential(*layers))
            setattr(self, f'post_smooth_layer{l}', nn.Sequential(*post_smooth))
            if l < len(num_iteration) - 1:
                Pi = nn.Conv2d(num_channel_u, num_channel_u, 3, stride=2, padding=1,
                               bias=False, padding_mode=padding_mode)
                R  = nn.Conv2d(num_channel_f, num_channel_f, 3, stride=2, padding=1,
                               bias=False, padding_mode=padding_mode)
                layers = [_Restrict(Pi=Pi, R=R)]

    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f
        for l in range(len(self.num_iteration)):
            out = getattr(self, f'layer{l}')(out)
            out_list[l] = out
        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, f = out_list[j]
            u_post = u + self.RTlayers[j](out_list[j + 1][0])
            out_list[j] = getattr(self, f'post_smooth_layer{j}')((u_post, f))
        return out_list[0][0]


class PhysConv(nn.Module):
    def __init__(self, m_channels_in, x_channels_in, channels_out):
        super().__init__()
        self.conv_m = nn.Sequential(nn.Conv2d(m_channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.GELU())
        self.conv_x = nn.Conv2d(x_channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, m, x):
        out = self.conv_m(m) * self.conv_x(x)
        return out

class PhysConvLight(nn.Module):
    def __init__(self, x_channels_in, channels_out):
        super().__init__()
        self.conv_x = nn.Conv2d(x_channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, c, x):
        out = c * self.conv_x(x)
        return out

class MPMG_precond(nn.Module):
    def __init__(self,
                 u_channels_list=[2, 20, 30, 40, 50],
                 a_channels_list=[32, 32, 32, 32, 32],
                 mg_levels=[1, 2, 4, 8, 8]):
        super().__init__()
        hidden = u_channels_list[0]
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1, bias=False),
        )

    def setup(self, coeff):
        return None, None

    def forward(self, rhs, a_list, s_list):
        return self.net(rhs)

#### MPMG preconditoned Born Series
class MPMG_BS(nn.Module):
    @preserve_default_dtype
    def __init__(self,
                 freq=1,
                 sos_mean=3000.,
                 sos_max=3000.,
                 sos_min=3000.,
                 N = 96,
                 dx=0.125,
                 boundary_width=8,
                 boundary_type='pbl',
                 u_channels_list=[5,10,15,20,25],
                 a_channels_list=[10,15,20,25,30],
                 mg_levels=[1,2,4,8,8],
                 eta_factor=1.0,
                 fdtype=torch.float32,
                 device="cpu"
                 ):
        super().__init__()
        torch.set_default_dtype(fdtype)
        self.float_dtype = fdtype
        self.complex_dtype = torch.complex128 if fdtype==torch.float64 else torch.complex64
        self.device = device
        # PDE params
        self.freq = freq
        self.eta_factor = eta_factor
        self.omega0 = 2*torch.pi*freq
        self.dx = dx
        self.N = N
        self.sos_max = sos_max
        self.sos_min = sos_min
        self.sos_mean = sos_mean
        ## set grid
        self.boundary_width = boundary_width
        self.N_pad = self.N+self.boundary_width*2
        # spatial coordinates 
        self.px_range = 2*torch.pi*torch.fft.fftfreq(self.N_pad,d=self.dx).view(-1,1)
        self.py_range = (2*torch.pi*torch.fft.fftfreq(self.N_pad,d=self.dx))
        self.p_dist2 = (self.px_range**2+self.py_range**2).to(self.device)

        ## boundary setting
        self.boundary_type = boundary_type

        self.Bl = int((self.N_pad - self.N)/2)
        self.Br = int((self.N_pad - self.N)/2)
        
        x = torch.cat((torch.arange(self.Bl, 0, -1), torch.zeros(self.N), torch.arange(1, self.Br + 1))).view(-1,1)
        y = torch.cat((torch.arange(self.Bl, 0, -1), torch.zeros(self.N), torch.arange(1, self.Br + 1)))
        self.boundary_dist = torch.sqrt(x**2 + y**2).to(self.device)
        ## add MPMG preconditioner
        self.precond = MPMG_precond(u_channels_list,
                                    a_channels_list,
                                    mg_levels).to(device)
    
    # def cal_update(self, residual)

    def forward(self, u):
        residual = (self.G(self.V*u+self.b/self.dx**2) - u)
        residual = torch.cat([residual.real, residual.imag], dim=1).to(torch.float32)
        update   = self.precond(residual, self.a_list, self.s_list).permute(0,2,3,1).unsqueeze(1).contiguous()
        u        = u + torch.view_as_complex(update).to(self.complex_dtype)
        
        return u
    
    def setup(self, sos, src=None):
        self.sos = sos
        self.setup_boundary()
        
        ## CBS params
        self.k0 = ((1/torch.mean(self.sos,dim=[-2,-1]))*self.omega0).unsqueeze(-1).unsqueeze(-1)
        self.epsilon = torch.amax(torch.abs(self.inv_sos2*self.omega0**2 - self.k0**2), dim=[-1,-2]).unsqueeze(-1).unsqueeze(-1)
        self.epsilon = self.eta_factor * self.epsilon
        # scatter potential
        self.V = self.inv_sos2*self.omega0**2 - self.k0**2 - 1.0j*self.epsilon
        # g0_tilde
        self.g0_tilde = 1/(self.p_dist2-self.k0**2 - 1.0j*self.epsilon)
        # b
        if src != None:
            self.b = F.pad(src, (self.boundary_width,)*4, mode='replicate')
        # precond
        # self.gamma = 1.j/self.epsilon*self.V
        # mpmg precond
        coeff1 = 2*(1/self.sos - 1/self.sos_mean)/(1/self.sos_min - 1/self.sos_max)
        coeff2 = (self.V.imag / self.epsilon + 1)*3
        coeff = torch.cat([coeff1,coeff2], dim=1).to(torch.float32)
        # import matplotlib.pyplot as plt
        # print(coeff1[0,0].max())
        # plt.imshow(coeff2[0,0].cpu())
        # plt.colorbar()
        # plt.savefig('test.png')
        # exit()
        ## setup precond
        self.a_list, self.s_list = self.precond.setup(coeff)


    def G(self, u):
        out = torch.fft.ifftn(self.g0_tilde * torch.fft.fftn(u,dim=[-1,-2],norm='backward'), dim=[-1,-2],norm='backward')
        return out

    def G_inv(self, u):
        out = torch.fft.ifftn(1/self.g0_tilde * torch.fft.fftn(u,dim=[-1,-2],norm='backward'), dim=[-1,-2],norm='backward')
        return  out
    
    def A_op(self, u):
        Au = (self.G_inv(u) - self.V*u)*(self.dx**2)
        return Au

    def cal_residual(self, u, rhs):
        return rhs - self.A_op(u)
    
    def setup_boundary(self): 
        # add padding
        self.sos = F.pad(self.sos, (self.Bl, self.Br, self.Bl, self.Br), mode='replicate')
        self.inv_sos2 = 1/self.sos**2

        factor = torch.abs(torch.mean(self.inv_sos2,dim=[-1,-2])).unsqueeze(-1).unsqueeze(-1)


        if self.boundary_type == "abc":
            # abc
            absorb = self.set_abc_layer(self.inv_sos2.size(-1),torch.max(self.Br),0.5)

            self.inv_sos2 = self.inv_sos2 - factor*absorb*1.j
            
        elif self.boundary_type == "pbl":
            # pbl
            # debug: why divide self.dx ?

            k0 = (1/torch.mean(self.sos, dim=[-1,-2]))*self.omega0*self.dx# k0 in 1/pixels
            # k0 = (1/1500.)*2*torch.pi/(self.lamb/self.dx)
            # k0 = torch.sqrt(torch.mean(self.inv_sos2, dim=[-1,-2]))*2*torch.pi/(self.lamb)# wjt's 
            k0 = k0.unsqueeze(-1).unsqueeze(-1)
            # the shape of the boundary is determined by f_boundary_curve, a function
            # that takes a position (in pixels, 0=start of boundary) and returns
            # Delta inv_sos2 for the boundary. 

            absorb = self.set_pbl_layer(r=self.boundary_dist,c=k0/2,k0=k0)
            self.inv_sos2 = self.inv_sos2 + factor*absorb.to(self.device)
    def set_pbl_layer(self, r, c, k0):
        # f_boundary_curve = 1 / k0**2 * (c**3 * r * (2.0 + (2.0j * k0 - c) * r)) / \
        #     (2.0 + 2.0 * c * r + c**2 * r**2) / k0**2 
        # f_boundary_curve = 1 / k0**2 * (c**4 * r**2 * (3.0 + (2.0j * k0 - c) * r)) / \
        #     (6 + 6 * c * r + 3 * c**2 * r**2 + c**3 * r**3)
        f_boundary_curve = 1 / k0**2 * (c**5 * r**3 * (4.0 + (2.0j * k0 - c) * r)) / \
            (24 + 24 * c * r + 12 * c**2 * r**2 + 4 * c**3 * r**3 + c**4 * r**4)
        # f_boundary_curve = 1 / k0**2 * (c**6 * r**4 * (5.0 + (2.0j * k0 - c) * r)) / \
        #     (120 + 120 * c * r + 60 * c**2 * r**2 + 20 * c**3 * r**3 + 5 * c**4 * r**4 + c**5 * r**5)
        # f_boundary_curve = 1 / k0**2 * (c**7 * r**5 * (6.0 + (2.0j * k0 - c) * r)) / \
        #     (720 + 720 * c * r + 360 * c**2 * r**2 + 120 * c**3 * r**3 + 30 * c**4 * r**4 + 6 * c**5 * r**5 + c**6 * r**6)
        return f_boundary_curve
    
    def set_abc_layer(self, domain_size, width, value):
        # not just set value, set value linearly increase to value within boundary layer
        abc_coord = torch.arange(width,device=self.device)
        # Sigma function (quadratic)
        layer_outer = value * (1 - abc_coord / width) ** 2
        layer_1d = torch.zeros((domain_size,),device=self.device)
        layer_1d[:width] = layer_outer
        layer_1d[-width:] = torch.flip(layer_outer, dims=[0])
        layer_x_2d, layer_y_2d = torch.meshgrid(layer_1d, layer_1d, indexing='xy')
        return layer_x_2d + layer_y_2d
    
"""
Fourier Preconditioner

"""
#fourier convolution 2d block
class fourier_conv_2d(nn.Module):
    @preserve_default_dtype
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

class Fourier_layer(nn.Module):
    @preserve_default_dtype
    def __init__(self,  features_, wavenumber, is_last = False):
        super(Fourier_layer, self).__init__()
        # torch.set_default_dtype(torch.float32)
        self.W =  nn.Conv2d(features_, features_, 1)
        self.fourier_conv = fourier_conv_2d(features_, features_ , *wavenumber)
        if is_last== False: 
            self.act = F.gelu
        else: 
            self.act = nn.Identity()
    def forward(self, x):
        x1 = self.fourier_conv(x)
        x2 = self.W(x)
        return self.act(x1 + x2) 
class Linear_Fourier_layer(nn.Module):
    @preserve_default_dtype
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
    @preserve_default_dtype
    def __init__(self, num_features=16, wavenumber=[40,40,40],N_pad=288):
        super().__init__()
        torch.set_default_dtype(torch.float32)
        self.num_features = num_features
        self.wavenumber    = wavenumber
        # lifting&projection
        # self.lifting_u = nn.Conv2d(2, num_features, kernel_size=1, bias=True)
        self.lifting_r = nn.Conv2d(2, num_features, kernel_size=1, padding=0, bias=False)
        self.lifting_a = nn.Conv2d(2, num_features, kernel_size=3, padding=1, bias=True)
        self.proj_u    = nn.Conv2d(num_features, 2, kernel_size=1, bias=False)
        # self.ln_a = nn.LayerNorm([2, 288, 288], elementwise_affine=True)
        # Fourier components
        # for a
        self.fno_nolinear = nn.ModuleList()
        for l in range(len(wavenumber)-1):
            self.fno_nolinear.append(Fourier_layer(features_=num_features, wavenumber=[wavenumber[l]]*2))
        self.fno_nolinear.append(Fourier_layer(features_=num_features, 
                                        wavenumber=[wavenumber[-1]]*2,
                                        is_last= True))
        self.fno_nolinear =nn.Sequential(*self.fno_nolinear)

        # for residual
        self.fno_linear = Linear_Fourier_layer(features_=num_features, wavenumber=[N_pad//2+1,N_pad//2+1])
        self.fno_linear2 = Linear_Fourier_layer(features_=num_features, wavenumber=[N_pad//2+1,N_pad//2+1])
    
    def setup(self, coeff):
        return coeff, None
    # Au = residual
    def forward(self, rhs, a_list, s_list):
        '''
        V       : size=(batch, 1, N_pad, N_pad), dtype=complex
        residual: size=(batch, 1, N_pad, N_pad), dtype=complex
        g0_tilde: size=(batch, 1, N_pad, N_pad), dtype=complex
        epsilon : size=(batch, 1, 1    , 1    ), dtype=double
        '''
        # # lifting
        # a        = self.lifting_a(self.ln_a(a_list))
        a        = self.lifting_a(a_list)
        res      = self.lifting_r(rhs)
        # fno
        out = self.fno_linear(self.fno_nolinear(a)*self.fno_linear2(res))
        
        # projection
        out =self.proj_u(out)

        # out = (torch.view_as_complex(out.permute(0,2,3,1).unsqueeze(1).contiguous()).to(torch.complex64))
        return  out
    
class PAFNO_BS(MPMG_BS):
    def __init__(self, loss_type, num_features=25, wavenumber=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        if wavenumber is None:
            wavenumber = [45, 45, 45]
        if max(wavenumber) > self.N_pad // 2 + 1:
            raise ValueError("Fourier modes exceed the padded grid; reduce wavenumber or increase N")
        self.precond = Fourier_precond(num_features=num_features, wavenumber=wavenumber, N_pad=self.N_pad).to(self.device)
    def forward(self, u):
        ## iterate
        # residual = b - Au
        # residual = (self.b - self.A_op(u))
        # u = u + P*(b-Au)
        # test fp32
        # update32 = self.gamma32*self.G(u=residual.to(torch.complex64)/(self.dx**2),g0_tilde=self.g0_tilde32)
        # u = u + update32.to(self.complex_dtype)
        # old
        # update = self.gamma*self.G(u=residual/(self.dx**2),g0_tilde=self.g0_tilde)
        # u = u + update
        # fast cbs
        if self.loss_type == 1:
            residual = (self.b - self.A_op(u)) # b-Ax
        else:
            residual = (self.G(self.V*u+self.b/self.dx**2) - u) # G(b-Ax)
        
        residual = torch.cat([residual.real, residual.imag], dim=1).to(torch.float32)
        update   = self.precond(residual, self.a_list, self.s_list).permute(0,2,3,1).unsqueeze(1).contiguous()
        u        = u + torch.view_as_complex(update).to(self.complex_dtype)
        
        return u


"""
MGCFNN preconditioner and Born-Series solver
"""

class MGCFNN_precond(nn.Module):
    """
    MGCFNN-based preconditioner that is a drop-in replacement for
    Fourier_precond / MPMG_precond.

    Architecture: 3-level multigrid with
      - Level 0 (fine)  : spatial conv, 1 sweep
      - Level 1 (mid)   : Fourier layer, wave_number=15, 2 sweeps
      - Level 2 (coarse): Fourier layer, wave_number=10, 4 sweeps
    channels = 32 throughout.
    """
    _FLOW_PARAM = {
        'channels': 32,
        'param': [
            {
                'flow_name': 'myflow',
                'param': {
                    'coeff_net_param': {
                        'arch_name': 'coeff_down_net',
                        'param': {
                            'resblk_param': {
                                'net_type': 'nonconv2d',
                                'net_param_dict': {
                                    'channels': 32,
                                    'kernel_size': 5,
                                    'padding_mode': 'replicate',
                                    'active': 'tanh',
                                },
                                'sweeps': 2,
                                'apply_coeff': 'none',
                                'short_cut_net': True,
                            },
                            'restrict_param': {
                                'channels': 32,
                                'kernel_size': 3,
                                'bias': False,
                            },
                            'levels': 3,
                        },
                    },
                    'solve_net_param': {
                        'arch_name': 'mgcnn',
                        'param': {
                            'resblk_param': [
                                {
                                    'net_type': 'conv2d',
                                    'net_param_dict': {
                                        'channels': 32,
                                        'kernel_size': 5,
                                        'padding_mode': 'zeros',
                                        'active': 'none',
                                    },
                                    'sweeps': 1,
                                    'apply_coeff': 'pre-dot',
                                    'short_cut_net': True,
                                },
                                {
                                    'net_type': 'fourier2d_cmplx',
                                    'net_param_dict': {
                                        'channels': 32,
                                        'wave_number': 15,
                                        'padding_factor': 1.2,
                                    },
                                    'sweeps': 2,
                                    'apply_coeff': 'pre-dot',
                                    'short_cut_net': True,
                                },
                                {
                                    'net_type': 'fourier2d_cmplx',
                                    'net_param_dict': {
                                        'channels': 32,
                                        'wave_number': 10,
                                        'padding_factor': 1.2,
                                    },
                                    'sweeps': 4,
                                    'apply_coeff': 'pre-dot',
                                    'short_cut_net': True,
                                },
                            ],
                            'transfer_param': [
                                {'channels': 32, 'kernel_size': 3, 'bias': False},
                                {'channels': 32, 'kernel_size': 3, 'bias': False},
                                {'channels': 32, 'kernel_size': 3, 'bias': False},
                            ],
                            'levels': 3,
                            'ml_type': 'diff',
                        },
                    },
                },
            }
        ],
    }

    def __init__(self):
        super().__init__()
        self.mymodel = load_mgcfnn_model()(
            in_channels=2,
            out_channels=2,
            coeff_in_channels=2,
            flow_param=self._FLOW_PARAM,
        )

    def setup(self, coeff):
        """
        Process the coefficient tensor and cache its feature maps.
        Returns (None, None) as a dummy (a_list, s_list) pair so that
        MPMG_BS.setup() can store them without error; MGCFNN_precond.forward
        ignores these arguments and uses the internally cached state.
        """
        self.mymodel.setup(coeff)
        return None, None

    def forward(self, rhs, _a_list, _s_list):
        """
        rhs : (batch, 2, N_pad, N_pad) real float32
        _a_list, _s_list are ignored; coeff state is cached inside self.mymodel.
        Returns : (batch, 2, N_pad, N_pad) real float32
        """
        return self.mymodel(rhs)


class MGCFNN_BS(MPMG_BS):
    """
    Born-Series solver preconditioned by MGCFNN.
    Replaces the MPMG_precond in MPMG_BS with MGCFNN_precond.
    Identical forward-iteration logic to PAFNO_BS.
    """
    def __init__(self, loss_type, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.precond = MGCFNN_precond().to(self.device)

    def forward(self, u):
        if self.loss_type == 1:
            residual = self.b - self.A_op(u)                           # b - Ax
        else:
            residual = self.G(self.V * u + self.b / self.dx ** 2) - u  # G(b - Ax)
        residual = torch.cat([residual.real, residual.imag], dim=1).to(torch.float32)
        update = (self.precond(residual, self.a_list, self.s_list)
                  .permute(0, 2, 3, 1).unsqueeze(1).contiguous())
        u = u + torch.view_as_complex(update).to(self.complex_dtype)
        return u



"""
MGNO preconditioner and Born-Series solver

Design mirrors MPMG_precond:
  - setup(coeff): coeff → _MGNO_coeff_net (nonlinear) → a_list[l] per level
  - forward(rhs, a_list, _): rhs → stacked V-cycle blocks, each level gated by a_list[l]

Building blocks ported from repos/MgNO-main/models.py (_MgConv_DC above).
No hardcoded spatial sizes.
"""

class _MGNO_coeff_net(nn.Module):
    """Nonlinear coefficient encoder producing per-level feature maps.

    coeff (B, 2, H, W) -> a_list: [(B, ch, H), (B, ch, H/2), ..., (B, ch, H/2^(L-1))]
    """
    def __init__(self, num_levels, num_channel):
        super().__init__()
        self.num_levels = num_levels
        self.lifting = nn.Conv2d(2, num_channel, kernel_size=3, padding=1, bias=True)
        self.nonlinear = nn.ModuleList([
            nn.Sequential(
                nn.GELU(),
                nn.Conv2d(num_channel, num_channel, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Conv2d(num_channel, num_channel, kernel_size=3, padding=1, bias=True),
            ) for _ in range(num_levels)
        ])
        self.restrict = nn.ModuleList([
            nn.Sequential(
                nn.GELU(),
                nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=2, padding=1, bias=True),
            ) for _ in range(num_levels - 1)
        ])

    def forward(self, coeff):
        a = self.lifting(coeff)
        a_list = []
        for l in range(self.num_levels):
            a_list.append(self.nonlinear[l](a))
            if l < self.num_levels - 1:
                a = self.restrict[l](a)
        return a_list


class _MgConv_DC_coeff(nn.Module):
    """V-cycle MgConv with per-level coefficient gating.

    At each level l the smoother correction is multiplied elementwise by a_list[l]:
        init step  : u  = a_list[l] * S(f)
        smooth step: u += a_list[l] * S(f - A(u))
    """
    def __init__(self, num_iteration, num_channel, padding_mode='zeros', bias=False):
        super().__init__()
        self.num_iteration = num_iteration
        L = len(num_iteration)
        self.RTlayers = nn.ModuleList([
            nn.ConvTranspose2d(num_channel, num_channel, kernel_size=4, stride=2, padding=1, bias=False)
            for _ in range(L - 1)
        ])
        for l, (n_pre, n_post) in enumerate(num_iteration):
            setattr(self, f'pre_S_{l}', nn.ModuleList([
                nn.Conv2d(num_channel, num_channel, 3, padding=1, bias=bias, padding_mode=padding_mode)
                for _ in range(n_pre)
            ]))
            setattr(self, f'pre_A_{l}', nn.ModuleList([
                nn.Identity() if (l == 0 and i == 0) else
                nn.Conv2d(num_channel, num_channel, 3, padding=1, bias=bias, padding_mode=padding_mode)
                for i in range(n_pre)
            ]))
            setattr(self, f'post_S_{l}', nn.ModuleList([
                nn.Conv2d(num_channel, num_channel, 3, padding=1, bias=bias, padding_mode=padding_mode)
                for _ in range(n_post)
            ]))
            setattr(self, f'post_A_{l}', nn.ModuleList([
                nn.Conv2d(num_channel, num_channel, 3, padding=1, bias=bias, padding_mode=padding_mode)
                for _ in range(n_post)
            ]))
            if l < L - 1:
                setattr(self, f'Pi_{l}', nn.Conv2d(num_channel, num_channel, 3, stride=2, padding=1,
                                                    bias=False, padding_mode=padding_mode))
                setattr(self, f'R_{l}',  nn.Conv2d(num_channel, num_channel, 3, stride=2, padding=1,
                                                    bias=False, padding_mode=padding_mode))

    def forward(self, f, a_list):
        L = len(self.num_iteration)
        out_list = [None] * L
        u = None
        for l in range(L):
            pre_S = getattr(self, f'pre_S_{l}')
            pre_A = getattr(self, f'pre_A_{l}')
            for i, (S, A) in enumerate(zip(pre_S, pre_A)):
                if l == 0 and i == 0:
                    u = a_list[l] * S(f)
                else:
                    u = u + a_list[l] * S(f - A(u))
            out_list[l] = (u, f)
            if l < L - 1:
                f = getattr(self, f'R_{l}')(f)
                u = getattr(self, f'Pi_{l}')(u)
        for j in range(L - 2, -1, -1):
            u, f = out_list[j]
            correction = self.RTlayers[j](out_list[j + 1][0])
            # Stride-two restriction rounds odd sizes up. Remove the extra
            # trailing cell introduced by prolongation before adding it back.
            u = u + correction[..., :u.size(-2), :u.size(-1)]
            for S, A in zip(getattr(self, f'post_S_{j}'), getattr(self, f'post_A_{j}')):
                u = u + a_list[j] * S(f - A(u))
            out_list[j] = (u, f)
        return out_list[0][0]


class MGNO_precond(nn.Module):
    """MgNO-based preconditioner with nonlinear coefficient conditioning.

    Mirrors MPMG_precond:
      setup(coeff) runs coeff through _MGNO_coeff_net -> a_list per level.
      forward(rhs, a_list, _) lifts rhs, runs stacked V-cycle blocks gated by
      a_list, then projects to 2 output channels.
    """
    @preserve_default_dtype
    def __init__(self, num_channel_u=32, num_iteration=None, activation='gelu'):
        super().__init__()
        torch.set_default_dtype(torch.float32)
        if num_iteration is None:
            num_iteration = [[2, 2], [2, 2], [2, 2], [2, 2]]
        num_levels = len(num_iteration)
        ch = num_channel_u

        self.coeff_net   = _MGNO_coeff_net(num_levels, ch)
        self.lifting_rhs = nn.Conv2d(2, ch, kernel_size=1, bias=False)
        # Single V-cycle, mirroring MPMG_precond's single mpmg_operator.
        # Stacking linear V-cycles adds no expressiveness (linear ∘ linear = linear).
        self.vcycle  = _MgConv_DC_coeff(num_iteration, ch)
        self.proj_u  = nn.Conv2d(ch, 2, kernel_size=1, bias=False)

    def setup(self, coeff):
        """coeff: (B, 2, N_pad, N_pad) float32 — returns (a_list, None)."""
        a_list = self.coeff_net(coeff)
        return a_list, None

    def forward(self, rhs, a_list, _s_list):
        x = self.lifting_rhs(rhs)
        x = self.vcycle(x, a_list)
        return self.proj_u(x)


class MGNO_BS(MPMG_BS):
    """Born-Series solver preconditioned by MgNO (MgConv_DC-based).

    Drop-in replacement for MGCFNN_BS; no hardcoded spatial sizes.
    """
    def __init__(self, loss_type, num_channel_u=32,
                 num_iteration=None, activation='gelu', **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.precond = MGNO_precond(
            num_channel_u=num_channel_u,
            num_iteration=num_iteration,
            activation=activation,
        ).to(self.device)

    def forward(self, u):
        if self.loss_type == 1:
            residual = self.b - self.A_op(u)
        else:
            residual = self.G(self.V * u + self.b / self.dx ** 2) - u
        residual = torch.cat([residual.real, residual.imag], dim=1).to(torch.float32)
        update = (self.precond(residual, self.a_list, self.s_list)
                  .permute(0, 2, 3, 1).unsqueeze(1).contiguous())
        u = u + torch.view_as_complex(update).to(self.complex_dtype)
        return u


"""
U-Net preconditioner and Born-Series solver

Nonlinear in coeff (reuses _MGNO_coeff_net), strictly linear in rhs.
Replaces the V-cycle solver with a U-Net encoder-decoder with skip connections.
All rhs-path convolutions have bias=False and no activations; coeff features
enter multiplicatively to preserve P(α·r1 + β·r2) = α·P(r1) + β·P(r2).
"""


class _UNet_linear_solver(nn.Module):
    def __init__(self, num_levels, num_channel):
        super().__init__()
        self.num_levels = num_levels
        ch = num_channel
        self.enc_conv1 = nn.ModuleList([
            nn.Conv2d(ch, ch, 3, padding=1, bias=False) for _ in range(num_levels)])
        self.enc_conv2 = nn.ModuleList([
            nn.Conv2d(ch, ch, 3, padding=1, bias=False) for _ in range(num_levels)])
        self.enc_downs = nn.ModuleList([
            nn.Conv2d(ch, ch, 3, stride=2, padding=1, bias=False) for _ in range(num_levels - 1)])
        self.dec_ups = nn.ModuleList([
            nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1, bias=False)
            for _ in range(num_levels - 1)])
        self.dec_conv1 = nn.ModuleList([
            nn.Conv2d(ch * 2, ch, 3, padding=1, bias=False) for _ in range(num_levels - 1)])
        self.dec_conv2 = nn.ModuleList([
            nn.Conv2d(ch, ch, 3, padding=1, bias=False) for _ in range(num_levels - 1)])

    def forward(self, x, a_list):
        L = self.num_levels
        skips = []
        for l in range(L):
            x = a_list[l] * self.enc_conv1[l](x)
            x = self.enc_conv2[l](x)
            skips.append(x)
            if l < L - 1:
                x = self.enc_downs[l](x)
        for j in range(L - 1):
            l = L - 2 - j
            x = self.dec_ups[j](x)
            skip = skips[l]
            if x.shape[-2:] != skip.shape[-2:]:
                dH = skip.size(2) - x.size(2)
                dW = skip.size(3) - x.size(3)
                x = F.pad(x, [dW // 2, dW - dW // 2, dH // 2, dH - dH // 2])
            x = torch.cat([skip, x], dim=1)
            x = a_list[l] * self.dec_conv1[j](x)
            x = self.dec_conv2[j](x)
        return x


class UNet_precond(nn.Module):
    def __init__(self, num_channel_u=32, num_levels=4):
        super().__init__()
        ch = num_channel_u
        self.coeff_net = _MGNO_coeff_net(num_levels, ch)
        self.lifting_rhs = nn.Conv2d(2, ch, kernel_size=1, bias=False)
        self.solver = _UNet_linear_solver(num_levels, ch)
        self.proj_u = nn.Conv2d(ch, 2, kernel_size=1, bias=False)

    def setup(self, coeff):
        a_list = self.coeff_net(coeff)
        return a_list, None

    def forward(self, rhs, a_list, _s_list):
        x = self.lifting_rhs(rhs)
        x = self.solver(x, a_list)
        return self.proj_u(x)


class UNet_BS(MPMG_BS):
    def __init__(self, loss_type, num_channel_u=32, num_levels=4, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.precond = UNet_precond(
            num_channel_u=num_channel_u,
            num_levels=num_levels,
        ).to(self.device)

    def forward(self, u):
        if self.loss_type == 1:
            residual = self.b - self.A_op(u)
        else:
            residual = self.G(self.V * u + self.b / self.dx ** 2) - u
        residual = torch.cat([residual.real, residual.imag], dim=1).to(torch.float32)
        update = (self.precond(residual, self.a_list, self.s_list)
                  .permute(0, 2, 3, 1).unsqueeze(1).contiguous())
        u = u + torch.view_as_complex(update).to(self.complex_dtype)
        return u


# ============================================================
# EncoderSolver preconditioner
# Adapted from CompactImplicitHelmholtz (Treister group).
# ============================================================

class ES_DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Softplus(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Softplus())

    def forward(self, x):
        return self.double_conv(x)










class ES_ImplicitStep(nn.Module):
    """FFT-based Green's function implicit layer (Treister group)."""
    def __init__(self, h, c_in):
        super().__init__()
        self.h = h
        real = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float)
        imag = torch.eye(3)
        self.weights = nn.Parameter(
            torch.stack((real, imag)).repeat(c_in // 2, 1, 1).reshape(1, c_in, 3, 3))
        self._green = None
        self._green_cache_key = None

    def forward(self, r):
        m = self.weights.shape
        n = r.shape
        height, width = n[-2], n[-1]
        batch_size, channels = n[0], n[1]

        r_padded = torch.zeros(batch_size, channels, 2 * height, 2 * width, device=r.device)
        r_padded[:, :, height // 2: 3 * height // 2, width // 2: 3 * width // 2] = r
        r_padded = torch.fft.rfft2(r_padded)

        # Training changes the kernel after every optimizer step. Cached
        # transforms must also be invalidated for new weights, shapes/devices,
        # or dtypes; never retain an autograd graph for a subsequent forward.
        cacheable = not self.training and not torch.is_grad_enabled()
        cache_key = (height, width, r.device, r.dtype, self.weights._version)
        if not cacheable or self._green is None or self._green_cache_key != cache_key:
            point_source = torch.zeros(n[-2], n[-1], dtype=torch.float, device=r.device)
            point_source[n[-2] // 2, n[-1] // 2] = 1
            mid1 = (m[-1] - 1) // 2
            mid2 = (m[-2] - 1) // 2
            Kp = torch.zeros(m[1], 3 * n[-2], 3 * n[-1], dtype=torch.float, device=r.device)
            Kp[:, :mid1 + 1, :mid2 + 1]   = self.weights[:, :, mid1:,       mid2:]
            Kp[:, -mid1:,    :mid2 + 1]   = self.weights[:, :, :mid1,  -(mid2 + 1):]
            Kp[:, :mid1 + 1, -mid2:]      = self.weights[:, :, -(mid1 + 1):, :mid2]
            Kp[:, -mid1:,    -mid2:]      = self.weights[:, :, :mid1,        :mid2]
            k_hat = torch.fft.rfft2(Kp)
            b_pad = torch.zeros(3 * n[-2], 3 * n[-1], dtype=torch.float, device=r.device)
            b_pad[n[-2]:n[-2] * 2, n[-1]:n[-1] * 2] = point_source
            b_hat = torch.fft.rfft2(b_pad)
            t = k_hat / ((k_hat ** 2) + self.h)
            xKh = torch.fft.irfft2(b_hat * t)
            xKh = xKh[:, n[-2] // 2: 5 * n[-2] // 2, n[-1] // 2: 5 * n[-1] // 2]
            green = torch.fft.rfft2(torch.fft.fftshift(xKh))
            self._green = green if cacheable else None
            self._green_cache_key = cache_key if cacheable else None
        else:
            green = self._green

        u = torch.fft.irfft2(green * r_padded)
        return u[:, :, height // 2: 3 * height // 2, width // 2: 3 * width // 2]








class _ES_coeff_net(nn.Module):
    """ES-style nonlinear coefficient encoder using DoubleConvBlock (BN + Softplus)."""
    def __init__(self, num_levels, num_channel):
        super().__init__()
        self.num_levels = num_levels
        self.lifting = nn.Sequential(
            nn.Conv2d(2, num_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.Softplus(),
        )
        self.blocks = nn.ModuleList([
            ES_DoubleConvBlock(num_channel, num_channel) for _ in range(num_levels)
        ])
        self.restrict = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channel, num_channel, 3, stride=2, padding=1),
                nn.BatchNorm2d(num_channel),
                nn.Softplus(),
            ) for _ in range(num_levels - 1)
        ])

    def forward(self, coeff):
        a = self.lifting(coeff)
        a_list = []
        for l in range(self.num_levels):
            a_list.append(self.blocks[l](a))
            if l < self.num_levels - 1:
                a = self.restrict[l](a)
        return a_list


class _ES_linear_solver(nn.Module):
    """ES-style U-Net solver: expand→depthwise→shrink→mix + ImplicitStep, linear in rhs."""
    def __init__(self, num_levels, num_channel, implicit=True):
        super().__init__()
        self.num_levels = num_levels
        self.implicit = implicit
        ch = num_channel
        self.enc_expand = nn.ModuleList([
            nn.Conv2d(ch, 2 * ch, 1, bias=False) for _ in range(num_levels)])
        self.enc_dw = nn.ModuleList([
            nn.Conv2d(2 * ch, 2 * ch, 3, groups=2 * ch, padding=1, bias=False)
            for _ in range(num_levels)])
        self.enc_coeff_proj = nn.ModuleList([
            nn.Conv2d(ch, 2 * ch, 1, bias=True) for _ in range(num_levels)])
        self.enc_shrink = nn.ModuleList([
            nn.Conv2d(2 * ch, ch, 1, bias=False) for _ in range(num_levels)])
        self.enc_mix = nn.ModuleList([
            nn.Conv2d(ch, ch, 3, padding=1, bias=False) for _ in range(num_levels)])
        self.enc_downs = nn.ModuleList([
            nn.Conv2d(ch, ch, 5, stride=2, padding=2, bias=False)
            for _ in range(num_levels - 1)])
        if implicit:
            self.implicit_step = ES_ImplicitStep(1e-5, ch)
        self.dec_ups = nn.ModuleList([
            nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1, bias=False)
            for _ in range(num_levels - 1)])
        self.dec_conv1 = nn.ModuleList([
            nn.Conv2d(ch * 2, ch, 3, padding=1, bias=False)
            for _ in range(num_levels - 1)])
        self.dec_conv2 = nn.ModuleList([
            nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            for _ in range(num_levels - 1)])

    def forward(self, x, a_list):
        L = self.num_levels
        skips = []
        for l in range(L):
            x = self.enc_expand[l](x)
            x = self.enc_coeff_proj[l](a_list[l]) * self.enc_dw[l](x)
            x = self.enc_shrink[l](x)
            x = self.enc_mix[l](x)
            skips.append(x)
            if l < L - 1:
                x = self.enc_downs[l](x)
        if self.implicit:
            x = self.implicit_step(x)
        for j in range(L - 1):
            l = L - 2 - j
            x = self.dec_ups[j](x)
            skip = skips[l]
            if x.shape[-2:] != skip.shape[-2:]:
                dH = skip.size(2) - x.size(2)
                dW = skip.size(3) - x.size(3)
                x = F.pad(x, [dW // 2, dW - dW // 2, dH // 2, dH - dH // 2])
            x = torch.cat([skip, x], dim=1)
            x = a_list[l] * self.dec_conv1[j](x)
            x = self.dec_conv2[j](x)
        return x


class EncoderSolverPrecond(nn.Module):
    def __init__(self, num_channel_u=32, num_levels=4, implicit=True):
        super().__init__()
        ch = num_channel_u
        self.coeff_net = _ES_coeff_net(num_levels, ch)
        self.lifting_rhs = nn.Conv2d(2, ch, kernel_size=1, bias=False)
        self.solver = _ES_linear_solver(num_levels, ch, implicit)
        self.proj_u = nn.Conv2d(ch, 2, kernel_size=1, bias=False)

    def setup(self, coeff):
        a_list = self.coeff_net(coeff)
        return a_list, None

    def forward(self, rhs, a_list, _s_list):
        x = self.lifting_rhs(rhs)
        x = self.solver(x, a_list)
        return self.proj_u(x)


class EncoderSolver_BS(MPMG_BS):
    def __init__(self, loss_type, num_channel_u=32, num_levels=4, implicit=True, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.precond = EncoderSolverPrecond(
            num_channel_u=num_channel_u,
            num_levels=num_levels,
            implicit=implicit,
        ).to(self.device)

    def forward(self, u):
        if self.loss_type == 1:
            residual = self.b - self.A_op(u)
        else:
            residual = self.G(self.V * u + self.b / self.dx ** 2) - u
        residual = torch.cat([residual.real, residual.imag], dim=1).to(torch.float32)
        update = (self.precond(residual, self.a_list, self.s_list)
                  .permute(0, 2, 3, 1).unsqueeze(1).contiguous())
        u = u + torch.view_as_complex(update).to(self.complex_dtype)
        return u


