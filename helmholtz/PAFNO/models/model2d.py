import torch
import torch.nn as nn
import torch.nn.functional as F
from .helm_op import helmholtz2d_op

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

class setup_model(nn.Module):

    def __init__(self, a_channels_in, a_channels_list, u_channels_list, num_levels):
        super().__init__()
        self.num_levels = num_levels
        self.a_channels_in = a_channels_in
        self.a_channels_list = a_channels_list
        self.u_channels_list = u_channels_list

        self.lifting_a = nn.Conv2d(a_channels_in, a_channels_list[0], kernel_size=3, stride=1, padding=1, bias=True)

        self.f_s_models = nn.ModuleList()
        self.f_a_models = nn.ModuleList()

        # projection layers to match u_channel_list
        self.proj_a_to_u_channel = nn.ModuleList()          # for a_list

        for i in range(num_levels):
            self.f_s_models.append(
                nn.Sequential(
                    nn.GELU(),
                    nn.Conv2d(a_channels_list[i], a_channels_list[i], kernel_size=3, stride=1, padding=1, bias=True),
                    # nn.GELU(),
                    # nn.Conv2d(a_channels_list[i], a_channels_list[i], kernel_size=3, stride=1, padding=1, bias=True)
                )
            )
            if i < num_levels - 1:
                self.f_a_models.append(
                    nn.Sequential(
                        nn.GELU(),
                        nn.Conv2d(a_channels_list[i], a_channels_list[i+1], kernel_size=3, stride=2, padding=1, bias=True)
                    )
                )


            self.proj_a_to_u_channel.append(
                nn.Sequential(
                    nn.GELU(),
                    nn.Conv2d(a_channels_list[i], u_channels_list[i], kernel_size=3, stride=1, padding=1, bias=True),
                )
            )
    
    def forward(self, a):

        a = self.lifting_a(a)
        a_list = [a]
        s_list = []
        
        for i in range(self.num_levels):
            s_list.append(self.f_s_models[i](a_list[i]))
            if i < self.num_levels - 1:
                a_list.append(self.f_a_models[i](a_list[i]))

        for i in range(self.num_levels):
            a_list[i] = self.proj_a_to_u_channel[i](a_list[i]) 

        return a_list, s_list
 
class A_operator(nn.Module):
    '''
    Computes Au = A_operator(u,a)
    Shared over the levels.
    '''
    def __init__(self, u_channels, a_channels):
        super().__init__()
        # self.conv_a = nn.Conv2d(a_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.physconv = PhysConvLight(u_channels, u_channels)
        self.conv_I = nn.Conv2d(u_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, u, a):
        Au = self.conv_I(u) +  self.physconv(a, u)
        return Au

    
class mg_iteration_init(nn.Module):
    '''
    u = u + S(b), since u = 0
    '''
    def __init__(self, u_channels, a_channels):
        super().__init__()
        self.physconv = PhysConvLight(u_channels, a_channels)
        self.conv_S   = nn.Conv2d(a_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, out):
        (u, a, s, rhs) = out
        u = self.conv_S(self.physconv(s, rhs))
        # u = s * self.conv_res(rhs)
        return u, a, s, rhs

class mg_iteration(nn.Module):
    '''
      u = u + S(b - Au)
    '''
    def __init__(self, u_channels, a_channels, A_op):
        super().__init__()
        self.A = A_op
        self.physconv = PhysConvLight(u_channels, a_channels)
        self.conv_S   = nn.Conv2d(a_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, out):
        (u, a, s, rhs) = out
        res = rhs - self.A(u, a)
        u = u + self.conv_S(self.physconv(s, res))
        # u = u + self.physconv(s, res)
        return u, a, s, rhs
    
class mg_restriction(nn.Module):
    def __init__(self, u_channels_in, u_channels_out, A_op):
        super().__init__()
        self.R_rhs = nn.Conv2d(u_channels_in, u_channels_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.A     = A_op
    def forward(self, out):
        (u_old, a_old, rhs_old) = out
        res_old = rhs_old - self.A(u_old, a_old)
        rhs = self.R_rhs(res_old)        
        u = None                  
        return u, rhs
    
class mg_prolongation(nn.Module):
    '''
    Prolongates u from the coarser level.
    '''
    def __init__(self, u_channels_in, u_channels_out):
        super().__init__()
        self.P_u = nn.ConvTranspose2d(u_channels_in, u_channels_out, kernel_size=4, stride=2, padding=1, bias=False)
    def forward(self, u):
        u = self.P_u(u)
        return u
    

class mpmg_operator(nn.Module):
    def __init__(self, u_channels_list, a_channels_list, num_iteration=[1, 1, 1, 1, 1]):
        super().__init__()
        self.num_iteration = num_iteration 

        self.u_channels_list = u_channels_list
        self.a_channels_list = a_channels_list

 
        self.setup_model = setup_model(a_channels_in=2,
                                       a_channels_list=a_channels_list, 
                                       u_channels_list=u_channels_list,
                                       num_levels=len(num_iteration))
        # Setup the A operators, restriction, and prolongation layers
        self.A_list  = nn.ModuleList([])
        self.RT_list = nn.ModuleList([])
        self.R_list  = nn.ModuleList([])

        for j in range(len(num_iteration)):
            self.A_list.append(A_operator(u_channels=u_channels_list[j], a_channels=a_channels_list[j]))
            if j < len(num_iteration) - 1:
                self.R_list.append(mg_restriction(u_channels_in=u_channels_list[j], 
                                                  u_channels_out=u_channels_list[j+1], 
                                                  A_op=self.A_list[j]))
                
                self.RT_list.append(mg_prolongation(u_channels_in=u_channels_list[j+1], 
                                                    u_channels_out=u_channels_list[j]))

        # Setup cycle layers for down and up sweeps
        self.Down_sweep_layers = nn.ModuleList([])
        self.Up_sweep_layers   = nn.ModuleList([])

        for l, num_iteration_l in enumerate(num_iteration):
            down_layer_i = []
            up_layer_i   = []
            # for down layers
            for li in range(num_iteration_l):
                if li == 0:
                    down_layer_i.append(mg_iteration_init(u_channels=self.u_channels_list[l],
                                                        a_channels=self.a_channels_list[l]))
                else:
                    down_layer_i.append(mg_iteration(u_channels=self.u_channels_list[l],
                                                    a_channels=self.a_channels_list[l],
                                                    A_op=self.A_list[l]))
                    
            self.Down_sweep_layers.append(nn.Sequential(*down_layer_i))

            # for up layers 
            if l < len(num_iteration) - 1:
                for li in range(num_iteration_l):
                    up_layer_i.append(mg_iteration(u_channels=self.u_channels_list[l],
                                                   a_channels=self.a_channels_list[l],
                                                   A_op=self.A_list[l]))
                self.Up_sweep_layers.append(nn.Sequential(*up_layer_i))

    def setup_coeff(self, a):
        a_list, s_list = self.setup_model(a)
        return a_list, s_list

    def forward(self, rhs, a_list, s_list):
        u = None
        out_list = [0] * len(self.num_iteration)

        # for l in range(len(self.num_iteration)):
        #     u, a, _, rhs, res = self.Down_sweep_layers[l]((u, a_list[l], s_list[l], rhs))
        #     out_list[l] = (u, rhs)
        #     if l < len(self.num_iteration) - 1:
        #         u, rhs = self.R_list[l]((u, a, rhs, res))
        for l in range(len(self.num_iteration)-1):
            u, a, _, rhs = self.Down_sweep_layers[l]((u, a_list[l], s_list[l], rhs))
            out_list[l] = (u, rhs)
            u, rhs = self.R_list[l]((u, a, rhs))
        l = len(self.num_iteration)-1
        u, _, _, rhs = self.Down_sweep_layers[l]((u, a_list[l], s_list[l], rhs))
        out_list[l] = (u, rhs)

        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, rhs = out_list[j][0], out_list[j][1]
            u = u + self.RT_list[j](out_list[j+1][0])
            u, _, _, rhs = self.Up_sweep_layers[j]((u, a_list[j], s_list[j], rhs))
            out_list[j] = (u, rhs)
        return out_list[0][0]
        '''plot basis'''
        # from utils import plot_basis
        # plot_basis(self,rhs,a_list,s_list)


class mpmg_operator_nopost(nn.Module):
    '''
    Multigrid operator.
    Input: rhs, a
    Returns: mg(rhs,a) approximates inv_A * rhs,
    using different u_channels and a_channels for each level.
    '''
    def __init__(self, u_channels_list, a_channels_list, num_iteration=[1, 1, 1, 1, 1]):
        super().__init__()
        self.num_iteration = num_iteration 
        assert len(u_channels_list) == len(num_iteration), "u_channels_list length must match number of levels"
        assert len(a_channels_list) == len(num_iteration), "a_channels_list length must match number of levels"
        self.u_channels_list = u_channels_list
        self.a_channels_list = a_channels_list

 
        self.setup_model = setup_model(a_channels_in=2,
                                       a_channels_list=a_channels_list, 
                                       u_channels_list=u_channels_list,
                                       num_levels=len(num_iteration))
        # Setup the A operators, restriction, and prolongation layers
        self.A_list  = nn.ModuleList([])
        self.RT_list = nn.ModuleList([])
        self.R_list  = nn.ModuleList([])

        for j in range(len(num_iteration)):
            self.A_list.append(A_operator(u_channels=u_channels_list[j], a_channels=a_channels_list[j]))
            if j < len(num_iteration) - 1:
                if j == 0:
                    self.R_list.append(mg_iteration(u_channels_in=u_channels_list[j], 
                                                           u_channels_out=u_channels_list[j+1], 
                                                           A_op=self.A_list[j]))
                else:
                    self.R_list.append(mg_restriction(u_channels_in=u_channels_list[j], 
                                                      u_channels_out=u_channels_list[j+1], 
                                                      A_op=self.A_list[j]))
                self.RT_list.append(mg_prolongation(u_channels_in=u_channels_list[j+1], 
                                                    u_channels_out=u_channels_list[j]))

        # Setup cycle layers for down and up sweeps
        self.Down_sweep_layers = nn.ModuleList([])
        # self.Up_sweep_layers   = nn.ModuleList([])

        for l, num_iteration_l in enumerate(num_iteration):
            down_layer_i = []
            # up_layer_i   = []
            # for down layers

            for li in range(num_iteration_l):
                if li == 0:
                    down_layer_i.append(mg_iteration_init(u_channels=self.u_channels_list[l],
                                                        a_channels=self.a_channels_list[l]))
                else:
                    down_layer_i.append(mg_iteration(u_channels=self.u_channels_list[l],
                                                    a_channels=self.a_channels_list[l],
                                                    A_op=self.A_list[l]))
                    
            self.Down_sweep_layers.append(nn.Sequential(*down_layer_i))

            # for up layers 
            # if l < len(num_iteration) - 1:
            #     for li in range(num_iteration_l):
            #         up_layer_i.append(mg_iteration(u_channels=self.u_channels_list[l],
            #                                        a_channels=self.a_channels_list[l],
            #                                        A_op=self.A_list[l]))
            #     self.Up_sweep_layers.append(nn.Sequential(*up_layer_i))

    def setup_coeff(self, a):
        a_list, s_list = self.setup_model(a)
        return a_list, s_list

    def forward(self, rhs, a_list, s_list):
        u = None
        out_list = [0] * len(self.num_iteration)
        for l in range(len(self.num_iteration)-1):
            u, a, _, rhs, res = self.Down_sweep_layers[l]((u, a_list[l], s_list[l], rhs))
            out_list[l] = (u, rhs)
        l = len(self.num_iteration)-1
        u, a, _, rhs, res = self.Down_sweep_layers[l]((u, a_list[l], s_list[l], rhs))
        out_list[l] = (u, rhs)
        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, rhs = out_list[j][0], out_list[j][1]
            u = u + self.RT_list[j](out_list[j+1][0])
            out_list[j] = (u, rhs)

        return out_list[0][0]
        '''plot coarse level'''
        from utils import plot_coarse_level
        plot_coarse_level(self,rhs, a_list, s_list)
    
class MPMG_precond(nn.Module):
    def __init__(self, 
                 u_channels_list=[2, 20, 30, 40, 50],
                 a_channels_list=[32, 32, 32, 32, 32],
                 mg_levels=[1, 2, 4, 8, 8]):
        super().__init__()
        torch.set_default_dtype(torch.float32)
        self.mg_levels = mg_levels
        self.lifting_rhs = nn.Conv2d(2, u_channels_list[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.mg = mpmg_operator(u_channels_list=u_channels_list, a_channels_list=a_channels_list, num_iteration=mg_levels)
        self.proj_u = nn.Conv2d(u_channels_list[0], 2, kernel_size=1, stride=1, padding=0, bias=False)

    def setup(self, coeff):
        '''
        coeff: [k(x), absorb], (batch, 2, N, N)
        '''
        a_list, s_list = self.mg.setup_coeff(coeff)
        return a_list, s_list
        
    def forward(self, rhs, a_list, s_list):
        rhs = self.lifting_rhs(rhs)   
        u = self.mg(rhs, a_list, s_list)
        u = self.proj_u(u)
        return u 

class MPMG_precond_nopost(nn.Module):
    def __init__(self, 
                 u_channels_list=[2, 20, 30, 40, 50],
                 a_channels_list=[32, 32, 32, 32, 32],
                 mg_levels=[1, 2, 4, 8, 8]):
        super().__init__()
        torch.set_default_dtype(torch.float32)
        self.mg_levels = mg_levels
        self.lifting_rhs = nn.Conv2d(2, u_channels_list[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.mg = mpmg_operator_nopost(u_channels_list=u_channels_list, a_channels_list=a_channels_list, num_iteration=mg_levels)
        self.proj_u = nn.Conv2d(u_channels_list[0], 2, kernel_size=1, stride=1, padding=0, bias=False)

    def setup(self, coeff):
        '''
        coeff: [k(x), absorb], (batch, 2, N, N)
        '''
        a_list, s_list = self.mg.setup_coeff(coeff)
        return a_list, s_list
        
    def forward(self, rhs, a_list, s_list):
        rhs = self.lifting_rhs(rhs)   
        u = self.mg(rhs, a_list, s_list)
        u = self.proj_u(u)
        return u 
class MPMG(nn.Module):
    def __init__(self, 
                 freq,
                 sos_mean,
                 sos_max,
                 sos_min,
                 N, 
                 dx, 
                 boundary_width=32,
                 abc_factor=1.,
                 helm_type='fdm_abc',
                 u_channels_list=[2, 20, 30, 40, 50],
                 a_channels_list=[32, 32, 32, 32, 32],
                 mg_levels=[1, 2, 4, 8, 8],
                 fdtype=torch.float32,
                 device='cuda'):
        super().__init__()
        self.float_dtype = fdtype
        self.complex_dtype = torch.complex128 if fdtype==torch.float64 else torch.complex64
        self.device = device
        # PDE params
        self.freq = freq # wavelength, the units depend on the length units of the spacial field
        self.omega = 2*torch.pi*freq
        self.dx = dx
        self.N = N
        self.sos_mean = sos_mean
        self.sos_max  = sos_max
        self.sos_min  = sos_min
        ## set grid
        self.boundary_width = boundary_width
        self.helm_type = helm_type
        self.abc_factor = abc_factor
        self.N_pad = self.N+self.boundary_width*2


        ## precond
        self.precond = MPMG_precond(u_channels_list,
                                    a_channels_list,
                                    mg_levels).to(device)#.to(torch.float64)

        ## helmholtz operator
        k0 = self.omega/self.sos_mean*self.dx
        self.helm_op = helmholtz2d_op(domain_size=self.N_pad,
                                        boundary_size=boundary_width,
                                        dx=dx,
                                        k0=k0,
                                        abc_factor=abc_factor,
                                        float_type=self.float_dtype,
                                        kind=helm_type).to(device)
        # self.absorb = self.helm_op.absorb.repeat(10, 1,1,1).to(torch.float32)

    def setup(self, sos, src=None):
        ## pad kh, gen coeff
        k= self.omega/sos#*self.dx
        self.k = F.pad(k, (self.boundary_width,)*4, mode='replicate')

        coeff1 = 2*(self.k - self.omega/self.sos_mean)/(self.omega/self.sos_min - self.omega/self.sos_max)
        if self.helm_type == 'fft_pbl':
            coeff2 = self.helm_op.absorb.imag.repeat(coeff1.size(0), 1, 1, 1).to(torch.float32)*self.abc_factor
        else:
            coeff2 = self.helm_op.absorb.repeat(coeff1.size(0), 1, 1, 1).to(torch.float32)/self.abc_factor
        # import matplotlib.pyplot as plt
        # print(coeff1[0,0].max())
        # plt.imshow(coeff1[0,0].cpu())
        # plt.colorbar()
        # plt.savefig('test.png')
        # exit()
        # test fp64
        # coeff = torch.cat([coeff1,coeff2], dim=1).to(torch.float64)
        coeff = torch.cat([coeff1,coeff2], dim=1)
        ## setup precond
        self.a_list, self.s_list = self.precond.setup(coeff)
        ## pad rhs
        if src != None:
            self.b = F.pad(src, (self.boundary_width,)*4, mode='constant', value=0)
            
    def A_op(self, u):
        return self.helm_op(u,self.k)

    def forward(self, u):
        ## iterate
        # residual = b - Au
        residual = self.b - self.A_op(u)
        # if self.helm_type == 'fft_pbl':
        #      residual = self.G(residual/(self.dx**2))
        # u = u + P*(b-Au)
        residual = torch.cat([residual.real, residual.imag], dim=1).to(torch.float32)
        update = self.precond(residual,self.a_list, self.s_list).permute(0,2,3,1).unsqueeze(1).contiguous()
        u = u + torch.view_as_complex(update).to(self.complex_dtype)
        
        return u
class MPMG_nopost(nn.Module):
    def __init__(self, 
                 freq,
                 sos_mean,
                 sos_max,
                 sos_min,
                 N, 
                 dx, 
                 boundary_width=32,
                 abc_factor=1.,
                 helm_type='fdm_abc',
                 u_channels_list=[2, 20, 30, 40, 50],
                 a_channels_list=[32, 32, 32, 32, 32],
                 mg_levels=[1, 2, 4, 8, 8],
                 fdtype=torch.float32,
                 device='cuda'):
        super().__init__()
        self.float_dtype = fdtype
        self.complex_dtype = torch.complex128 if fdtype==torch.float64 else torch.complex64
        self.device = device
        # PDE params
        self.freq = freq # wavelength, the units depend on the length units of the spacial field
        self.omega = 2*torch.pi*freq
        self.dx = dx
        self.N = N
        self.sos_mean = sos_mean
        self.sos_max  = sos_max
        self.sos_min  = sos_min
        ## set grid
        self.boundary_width = boundary_width
        self.helm_type = helm_type
        self.abc_factor = abc_factor
        self.N_pad = self.N+self.boundary_width*2


        ## precond
        self.precond = MPMG_precond_nopost(u_channels_list,
                                            a_channels_list,
                                            mg_levels).to(device)#.to(torch.float64)

        ## helmholtz operator
        k0 = self.omega/self.sos_mean*self.dx
        self.helm_op = helmholtz2d_op(domain_size=self.N_pad,
                                        boundary_size=boundary_width,
                                        dx=dx,
                                        k0=k0,
                                        abc_factor=abc_factor,
                                        float_type=self.float_dtype,
                                        kind=helm_type).to(device)
        # self.absorb = self.helm_op.absorb.repeat(10, 1,1,1).to(torch.float32)

    def setup(self, sos, src=None):
        ## pad kh, gen coeff
        k= self.omega/sos#*self.dx
        self.k = F.pad(k, (self.boundary_width,)*4, mode='replicate')

        coeff1 = 2*(self.k - self.omega/self.sos_mean)/(self.omega/self.sos_min - self.omega/self.sos_max)
        if self.helm_type == 'fft_pbl':
            coeff2 = self.helm_op.absorb.imag.repeat(coeff1.size(0), 1, 1, 1).to(torch.float32)*self.abc_factor
        else:
            coeff2 = self.helm_op.absorb.repeat(coeff1.size(0), 1, 1, 1).to(torch.float32)/self.abc_factor
        # import matplotlib.pyplot as plt
        # print(coeff1[0,0].max())
        # plt.imshow(coeff1[0,0].cpu())
        # plt.colorbar()
        # plt.savefig('test.png')
        # exit()
        # test fp64
        # coeff = torch.cat([coeff1,coeff2], dim=1).to(torch.float64)
        coeff = torch.cat([coeff1,coeff2], dim=1)
        ## setup precond
        self.a_list, self.s_list = self.precond.setup(coeff)
        ## pad rhs
        if src != None:
            self.b = F.pad(src, (self.boundary_width,)*4, mode='constant', value=0)
            
    def A_op(self, u):
        return self.helm_op(u,self.k)

    def forward(self, u):
        ## iterate
        # residual = b - Au
        residual = self.b - self.A_op(u)
        # if self.helm_type == 'fft_pbl':
        #      residual = self.G(residual/(self.dx**2))
        # u = u + P*(b-Au)
        residual = torch.cat([residual.real, residual.imag], dim=1).to(torch.float32)
        update = self.precond(residual,self.a_list, self.s_list).permute(0,2,3,1).unsqueeze(1).contiguous()
        u = u + torch.view_as_complex(update).to(self.complex_dtype)
        
        return u
#### MPMG preconditoned Born Series
class MPMG_BS(nn.Module):
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
    def __init__(self,loss_type, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.precond = Fourier_precond(num_features=25, wavenumber=[45,45,45],N_pad = self.N_pad).to(self.device)
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

def trainable_params_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024**2)


if __name__ == "__main__":
    import time
     
    # MG_precond_5 now accepts u_channels_list and a_channels_list to specify channels for each level.
    model = MPMG_precond(u_channels_list=[5, 15, 20, 25, 30],
                       a_channels_list=[10, 20, 30, 32, 32],
                       mg_levels=[1, 2, 4, 8, 8]).to('cuda')
    # print(model)
    a  = torch.rand(1, 2, 576, 576).to('cuda')
    x1 = torch.rand(1, 2, 576, 576).to('cuda')
    x2 = torch.rand(1, 2, 576, 576).to('cuda')
    a_list, s_list = model.setup(a)
    out = model(x1, a_list, s_list)
    t1 = time.perf_counter()
    model.setup(a)
    t2 = time.perf_counter()
    print(f"model setup time used: {t2 - t1:.4e}s")

    t1 = time.perf_counter()
    for i in range(6):
        out = model(x1, a_list, s_list)
    t2 = time.perf_counter()
    print(f"model solve time used: {t2 - t1:.4e}s")
    print(f"model param size: {trainable_params_size(model)}MB")

    # Linear check
    print("linear check: ", torch.linalg.norm(model(x1, a_list, s_list) + model(x2, a_list, s_list) - model(x1 + x2, a_list, s_list)) / torch.linalg.norm(model(x1 + x2, a_list, s_list)))
    # print(model(x1).dtype)




    # src = torch.rand(1, 1, 480, 480).to('cuda')

    # model.setup(sos,src)
    # u = torch.zeros_like(model.kh,dtype=model.complex_dtype)
    # for i in range(5):
    #     u = model(u)
    # print(u.shape)
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    # start_event.record()
    # model.setup(sos,src)
    # u = torch.zeros_like(model.kh,dtype=model.complex_dtype)
    # for i in range(6):
    #     u = model(u)
    # end_event.record()
    # torch.cuda.synchronize()
    # total_time = start_event.elapsed_time(end_event) / 1000.0
    # print(f'time: {total_time:.4f}s')