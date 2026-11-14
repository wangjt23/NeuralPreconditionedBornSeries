import torch
import torch.nn as nn
import torch.nn.functional as F

class coeff_model(nn.Module):
    """
    Pre-process the coefficient 'a' and share it over several levels.
    Now supports projecting each level to user-defined u_channel_list.
    """

    def __init__(self, a_channels_in, a_channels_list, u_channels_list, num_levels):
        super().__init__()
        self.num_levels = num_levels
        self.a_channels_in = a_channels_in

        self.lifting_a = nn.Conv2d(a_channels_in, a_channels_list[0], kernel_size=3, stride=1, padding=1, bias=True)

        # per‑level modules
        self.inv_a_models = nn.ModuleList()
        self.restrict_a_models = nn.ModuleList()

        # projection layers to match u_channel_list
        self.proj_a_to_u = nn.ModuleList()          # for a_list
        self.proj_inv_a_to_u = nn.ModuleList()      # for inv_a_list

        for i in range(num_levels):
            self.inv_a_models.append(
                nn.Sequential(
                    nn.Conv2d(a_channels_list[i], a_channels_list[i], kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(a_channels_list[i]),
                    nn.GELU(),
                    nn.Conv2d(a_channels_list[i], a_channels_list[i], kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(a_channels_list[i]),
                    nn.GELU(),
                )
            )
            self.proj_a_to_u.append(
                nn.Sequential(
                    nn.Conv2d(a_channels_list[i], u_channels_list[i], kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(u_channels_list[i]),
                    nn.GELU()
                )
            )
            
            self.proj_inv_a_to_u.append(
                nn.Sequential(
                    nn.Conv2d(a_channels_list[i], u_channels_list[i], kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(u_channels_list[i]),
                    nn.GELU(),
                )
            )
            
            if i < num_levels - 1:
                self.restrict_a_models.append(
                    nn.Sequential(
                        nn.Conv2d(a_channels_list[i], a_channels_list[i+1], kernel_size=3, stride=2, padding=1, bias=True),
                        nn.BatchNorm2d(a_channels_list[i+1]),
                        nn.GELU(),
                    )
                )

    def forward(self, a):
        """
        Returns
        -------
        a_list, inv_a_list 
        """
        a = self.lifting_a(a)
        a_list = [a]
        inv_a_list = []
        
        for i in range(self.num_levels):
            inv_a_list.append(self.inv_a_models[i](a_list[i]))
            if i < self.num_levels - 1:
                a_list.append(self.restrict_a_models[i](a_list[i]))

        for i in range(self.num_levels):
            a_list[i] = self.proj_a_to_u[i](a_list[i]) 
            inv_a_list[i] = self.proj_inv_a_to_u[i](inv_a_list[i])


        return a_list, inv_a_list

    
class A_operator(nn.Module):
    '''
    Computes Au = A_operator(u,a)
    Shared over the levels.
    '''
    def __init__(self, u_channels, a_channels):
        super().__init__()
        # self.conv_a = nn.Conv2d(a_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_u = nn.Conv2d(u_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_I = nn.Conv2d(u_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, u, a):
        Au = self.conv_I(u) +  a * self.conv_u(u)
        return Au

    
class mg_iteration(nn.Module):
    '''
    Performs a single multigrid iteration:
    u = u + B(b - Au)
    where B(b-Au) = linear_conv(nonlinear_conv(a)*linear_conv(b-Au))
    '''
    def __init__(self, u_channels, a_channels, A_op):
        super().__init__()
        self.A = A_op
        self.conv_res = nn.Conv2d(u_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, out):
        (u, a, inv_a, rhs, res) = out
        # u = u + self.conv_B(inv_a * self.conv_res(rhs))
        # u = u + inv_a * self.conv_res(res)
        u = u + inv_a * self.conv_res(res)
        res = rhs - self.A(u, a)
        return u, a, inv_a, rhs, res
    
class mg_restriction(nn.Module):
    '''
    Restricts rhs and initializes u to zeros at the coarser level.
    '''
    def __init__(self, u_channels_in, u_channels_out, A_op):
        super().__init__()
        self.R_rhs = nn.Conv2d(u_channels_in, u_channels_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.A     = A_op
    def forward(self, rhs):
        rhs = self.R_rhs(rhs)        
        u = torch.zeros_like(rhs)                  
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
    

class mg_operator(nn.Module):
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


        self.coeff_model = coeff_model(a_channels_in=3,
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
        a_list, inv_a_list = self.coeff_model(a)
        return a_list, inv_a_list

    def forward(self, rhs, a_list, inv_a_list):
        u = torch.zeros_like(rhs)
        out_list = [0] * len(self.num_iteration)

        for l in range(len(self.num_iteration)):
            u, _, _, rhs, res = self.Down_sweep_layers[l]((u, a_list[l], inv_a_list[l], rhs, rhs))
            out_list[l] = (u, rhs)
            if l < len(self.num_iteration) - 1:
                u, rhs = self.R_list[l](res)

        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, rhs = out_list[j][0], out_list[j][1]
            u = u + self.RT_list[j](out_list[j+1][0])
            u, _, _, rhs, _ = self.Up_sweep_layers[j]((u, a_list[j], inv_a_list[j], rhs, rhs))
            out_list[j] = (u, rhs)
            
        return out_list[0][0]
    
class MG_precond(nn.Module):
    def __init__(self, 
                M,
                u_channels_list=[2, 20, 30, 40, 50],
                a_channels_list=[32, 32, 32, 32, 32],
                mg_levels=[1, 2, 4, 8, 8],
                input_channels=None,
                output_channels=None):
        super().__init__()
        torch.set_default_dtype(torch.float32)
        self.mg_levels = mg_levels
        self.input_channels = 8 * M if input_channels is None else int(input_channels)
        self.output_channels = self.input_channels if output_channels is None else int(output_channels)
        self.lifting_rhs = nn.Conv2d(self.input_channels, u_channels_list[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.mg = mg_operator(u_channels_list=u_channels_list, a_channels_list=a_channels_list, num_iteration=mg_levels)
        self.proj_u = nn.Conv2d(u_channels_list[0], self.output_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def setup(self, coeff):
        '''
        coeff: [k(x), absorb], (batch, 2, N, N)
        '''
        a_list, inv_a_list = self.mg.setup_coeff(coeff)
        return a_list, inv_a_list
        
    def forward(self, rhs, a_list, inv_a_list):
        rhs = self.lifting_rhs(rhs)   
        u = self.mg(rhs, a_list, inv_a_list)
        u = self.proj_u(u)
        return u 
