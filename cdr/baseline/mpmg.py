# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class coeff_model(nn.Module):
#     """
#     Pre-process the coefficient 'a' and share it over several levels.
#     Now supports projecting each level to user-defined u_channel_list.
#     """

#     def __init__(self, a_channels_in, a_channels_list, u_channels_list, num_levels):
#         super().__init__()
#         self.num_levels = num_levels
#         self.a_channels_in = a_channels_in
#         self.a_channels_list = a_channels_list
#         self.u_channels_list = u_channels_list

#         self.lifting_a = nn.Conv2d(a_channels_in, a_channels_list[0], kernel_size=3, stride=1, padding=1, bias=True)

#         # per‑level modules
#         self.inv_a_models = nn.ModuleList()
#         self.restrict_a_models = nn.ModuleList()

#         # projection layers to match u_channel_list
#         self.proj_a_to_u = nn.ModuleList()          # for a_list
#         self.proj_inv_a_to_u = nn.ModuleList()      # for inv_a_list

#         for i in range(num_levels):
#             self.inv_a_models.append(
#                 nn.Sequential(
#                     nn.GELU(),
#                     nn.Conv2d(a_channels_list[i], a_channels_list[i], kernel_size=3, stride=1, padding=1, bias=True),
#                     nn.GELU(),
#                     nn.Conv2d(a_channels_list[i], a_channels_list[i], kernel_size=3, stride=1, padding=1, bias=True)
#                 )
#             )
#             if i < num_levels - 1:
#                 self.restrict_a_models.append(
#                     nn.Sequential(
#                         nn.GELU(),
#                         nn.Conv2d(a_channels_list[i], a_channels_list[i+1], kernel_size=3, stride=2, padding=1, bias=True)
#                     )
#                 )


#             self.proj_a_to_u.append(
#                 nn.Sequential(
#                     nn.GELU(),
#                     nn.Conv2d(a_channels_list[i], u_channels_list[i], kernel_size=3, stride=1, padding=1, bias=True),
#                 )
#             )
#             # self.proj_inv_a_to_u.append(
#             #     nn.Sequential(
#             #         nn.GELU(),
#             #         nn.Conv2d(a_channels_list[i], u_channels_list[i], kernel_size=3, stride=1, padding=1, bias=True)
#             #     )
#             # )

#     def forward(self, a):
#         """
#         Returns
#         -------
#         a_list, inv_a_list 
#         """
#         a = self.lifting_a(a)
#         a_list = [a]
#         inv_a_list = []
        
#         for i in range(self.num_levels):
#             inv_a_list.append(self.inv_a_models[i](a_list[i]))
#             if i < self.num_levels - 1:
#                 a_list.append(self.restrict_a_models[i](a_list[i]))

#         for i in range(self.num_levels):
#             a_list[i]     = self.proj_a_to_u[i](a_list[i]) 
#             # inv_a_list[i] = self.proj_inv_a_to_u[i](inv_a_list[i])


#         return a_list, inv_a_list
 
    
# class A_operator(nn.Module):
#     '''
#     Computes Au = A_operator(u,a)
#     Shared over the levels.
#     '''
#     def __init__(self, u_channels, a_channels):
#         super().__init__()
#         # self.conv_a = nn.Conv2d(a_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv_u = nn.Conv2d(u_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv_I = nn.Conv2d(u_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
#     def forward(self, u, a):
#         Au = (self.conv_I(u) +  a * self.conv_u(u))
#         return Au
    
# class mg_iteration_init(nn.Module):
#     '''
#     Performs a single multigrid iteration:
#       u = u + B(b)
#     where B(b) = linear_conv(nonlinear_conv(a)*linear_conv(b-Au))
#     '''
#     def __init__(self, u_channels, a_channels):
#         super().__init__()
#         self.conv_res = nn.Conv2d(u_channels, a_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv_B   = nn.Conv2d(a_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
#     def forward(self, out):
#         (u, a, inv_a, rhs) = out
#         u = self.conv_B(inv_a * self.conv_res(rhs))
#         # u = inv_a * self.conv_res(rhs)
#         return u, a, inv_a, rhs, None
    
# class mg_iteration(nn.Module):
#     '''
#     Performs a single multigrid iteration:
#       u = u + B(b - Au)
#     where B(b-Au) = linear_conv(nonlinear_conv(a)*linear_conv(b-Au))
#     '''
#     def __init__(self, u_channels, a_channels, A_op):
#         super().__init__()
#         self.A = A_op
#         self.conv_res = nn.Conv2d(u_channels, a_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv_B   = nn.Conv2d(a_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
#     def forward(self, out):
#         (u, a, inv_a, rhs, _) = out
#         res = rhs - self.A(u, a)
#         u = u + self.conv_B(inv_a * self.conv_res(res))
#         # u = u + inv_a * self.conv_res(res)
#         return u, a, inv_a, rhs, res
    
# class mg_restriction_init(nn.Module):
#     '''
#     Restricts rhs and initializes u to zeros at the coarser level.
#     '''
#     def __init__(self, u_channels_in, u_channels_out, A_op):
#         super().__init__()
#         self.R_rhs = nn.Conv2d(u_channels_in, u_channels_out, kernel_size=3, stride=2, padding=1, bias=False)
#         self.A     = A_op
#     def forward(self, out):
#         (u_old, a_old, rhs_old, _) = out
#         res_old = rhs_old - self.A(u_old, a_old)
#         rhs = self.R_rhs(res_old)        
#         u = None                  
#         return u, rhs
    
# class mg_restriction(nn.Module):
#     '''
#     Restricts rhs and initializes u to zeros at the coarser level.
#     '''
#     def __init__(self, u_channels_in, u_channels_out, A_op):
#         super().__init__()
#         self.R_rhs = nn.Conv2d(u_channels_in, u_channels_out, kernel_size=3, stride=2, padding=1, bias=False)
#         self.A     = A_op
#     def forward(self, out):
#         (_, _, _, res_old) = out
#         rhs = self.R_rhs(res_old)        
#         u = None                  
#         return u, rhs
    
# class mg_prolongation(nn.Module):
#     '''
#     Prolongates u from the coarser level.
#     '''
#     def __init__(self, u_channels_in, u_channels_out):
#         super().__init__()
#         self.P_u = nn.ConvTranspose2d(u_channels_in, u_channels_out, kernel_size=4, stride=2, padding=1, bias=False)
#     def forward(self, u):
#         u = self.P_u(u)
#         return u
    

# class mpmg_operator(nn.Module):
#     '''
#     Multigrid operator.
#     Input: rhs, a
#     Returns: mg(rhs,a) approximates inv_A * rhs,
#     using different u_channels and a_channels for each level.
#     '''
#     def __init__(self, u_channels_list, a_channels_list, num_iteration=[1, 1, 1, 1, 1]):
#         super().__init__()
#         self.num_iteration = num_iteration 
#         assert len(u_channels_list) == len(num_iteration), "u_channels_list length must match number of levels"
#         assert len(a_channels_list) == len(num_iteration), "a_channels_list length must match number of levels"
#         self.u_channels_list = u_channels_list
#         self.a_channels_list = a_channels_list

 
#         self.coeff_model = coeff_model(a_channels_in=3,
#                                        a_channels_list=a_channels_list, 
#                                        u_channels_list=u_channels_list,
#                                        num_levels=len(num_iteration))
#         # Setup the A operators, restriction, and prolongation layers
#         self.A_list  = nn.ModuleList([])
#         self.RT_list = nn.ModuleList([])
#         self.R_list  = nn.ModuleList([])

#         for j in range(len(num_iteration)):
#             self.A_list.append(A_operator(u_channels=u_channels_list[j], a_channels=a_channels_list[j]))
#             if j < len(num_iteration) - 1:
#                 if j == 0:
#                     self.R_list.append(mg_restriction_init(u_channels_in=u_channels_list[j], 
#                                                            u_channels_out=u_channels_list[j+1], 
#                                                            A_op=self.A_list[j]))
#                 else:
#                     self.R_list.append(mg_restriction(u_channels_in=u_channels_list[j], 
#                                                       u_channels_out=u_channels_list[j+1], 
#                                                       A_op=self.A_list[j]))
#                 self.RT_list.append(mg_prolongation(u_channels_in=u_channels_list[j+1], 
#                                                     u_channels_out=u_channels_list[j]))

#         # Setup cycle layers for down and up sweeps
#         self.Down_sweep_layers = nn.ModuleList([])
#         self.Up_sweep_layers   = nn.ModuleList([])

#         for l, num_iteration_l in enumerate(num_iteration):
#             down_layer_i = []
#             up_layer_i   = []
#             # for down layers

#             for li in range(num_iteration_l):
#                 if li == 0:
#                     down_layer_i.append(mg_iteration_init(u_channels=self.u_channels_list[l],
#                                                         a_channels=self.a_channels_list[l]))
#                 else:
#                     down_layer_i.append(mg_iteration(u_channels=self.u_channels_list[l],
#                                                     a_channels=self.a_channels_list[l],
#                                                     A_op=self.A_list[l]))
                    
#             self.Down_sweep_layers.append(nn.Sequential(*down_layer_i))

#             # for up layers 
#             if l < len(num_iteration) - 1:
#                 for li in range(num_iteration_l):
#                     up_layer_i.append(mg_iteration(u_channels=self.u_channels_list[l],
#                                                    a_channels=self.a_channels_list[l],
#                                                    A_op=self.A_list[l]))
#                 self.Up_sweep_layers.append(nn.Sequential(*up_layer_i))

#     def setup_coeff(self, a):
#         a_list, inv_a_list = self.coeff_model(a)
#         return a_list, inv_a_list

#     def forward(self, rhs, a_list, inv_a_list):
#         u = None
#         out_list = [0] * len(self.num_iteration)

#         for l in range(len(self.num_iteration)):
#             u, a, _, rhs, res = self.Down_sweep_layers[l]((u, a_list[l], inv_a_list[l], rhs))
#             out_list[l] = (u, rhs)
#             if l < len(self.num_iteration) - 1:
#                 u, rhs = self.R_list[l]((u, a, rhs, res))

#         for j in range(len(self.num_iteration) - 2, -1, -1):
#             u, rhs = out_list[j][0], out_list[j][1]
#             u = u + self.RT_list[j](out_list[j+1][0])
#             u, _, _, rhs, _ = self.Up_sweep_layers[j]((u, a_list[j], inv_a_list[j], rhs, None))
#             out_list[j] = (u, rhs)
            
#         return out_list[0][0]
    
# class MPMG_precond(nn.Module):
#     def __init__(self, 
#                  u_channels_list=[2, 20, 30, 40, 50],
#                  a_channels_list=[32, 32, 32, 32, 32],
#                  mg_levels=[1, 2, 4, 8, 8]):
#         super().__init__()
#         torch.set_default_dtype(torch.float32)
#         self.mg_levels = mg_levels
#         self.lifting_rhs = nn.Conv2d(1, u_channels_list[0], kernel_size=1, stride=1, padding=0, bias=True)
#         self.mg = mpmg_operator(u_channels_list=u_channels_list, a_channels_list=a_channels_list, num_iteration=mg_levels)
#         self.proj_u = nn.Conv2d(u_channels_list[0], 1, kernel_size=1, stride=1, padding=0, bias=True)

#     def setup(self, coeff):
#         '''
#         coeff: [k(x), absorb], (batch, 2, N, N)
#         '''
#         a_list, inv_a_list = self.mg.setup_coeff(coeff)
#         return a_list, inv_a_list
        
#     def forward(self, rhs, a_list, inv_a_list):
#         rhs = self.lifting_rhs(rhs)   
#         u = self.mg(rhs, a_list, inv_a_list)
#         u = self.proj_u(u)
#         return u 


# class MPMG(nn.Module):
#     def __init__(self, 
#                  u_channels_list=[2, 20, 30, 40, 50],
#                  a_channels_list=[32, 32, 32, 32, 32],
#                  mg_levels=[1, 2, 4, 8, 8],
#                  fdtype=torch.float32,
#                  device='cuda'):
#         super().__init__()
#         self.float_dtype = fdtype
#         self.device = device


#         ## precond
#         self.precond = MPMG_precond(u_channels_list,
#                                     a_channels_list,
#                                     mg_levels).to(device)#.to(torch.float64)

#     def setup(self, sos):
#         grid = self.get_grid(sos.shape, sos.device)
#         coeff = torch.cat([sos,grid], dim=1)
#         ## setup precond
#         self.a_list, self.inv_a_list = self.precond.setup(coeff)

            
#     def get_grid(self, shape, device):
#         batchsize, _, size_x, size_y = shape
        
#         # Create a 1D tensor for x coordinates
#         gridx_1d = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
#         # Reshape to (1, size_x, 1) and expand to (batchsize, size_x, size_y)
#         # This is a memory-efficient operation that reuses the data from the original tensor.
#         gridx = gridx_1d.view(1, size_x, 1).expand(batchsize, size_x, size_y)
        
#         # Create a 1D tensor for y coordinates
#         gridy_1d = torch.linspace(0, 1, size_y, dtype=torch.float, device=device)
#         # Reshape to (1, 1, size_y) and expand to (batchsize, size_x, size_y)
#         gridy = gridy_1d.view(1, 1, size_y).expand(batchsize, size_x, size_y)
#         return torch.stack([gridx, gridy], dim=1)
#     @staticmethod
#     def _pad_to_32(x):
#         _, _, h, w = x.shape
#         nh = math.ceil(h / 32) * 32
#         nw = math.ceil(w / 32) * 32
#         ph, pw = nh - h, nw - w
#         top, bottom = ph // 2, ph - ph // 2
#         left, right = pw // 2, pw - pw // 2
#         return F.pad(x, (left, right, top, bottom)), (left, right, top, bottom)

#     @staticmethod
#     def _unpad(x, pad):
#         left, right, top, bottom = pad
#         if (left + right + top + bottom) == 0:
#             return x
#         return x[:, :, top: x.size(2)-bottom, left: x.size(3)-right]
#     def forward(self, sos):
#         ## iterate
#         sos, pad = self._pad_to_32(sos)
#         self.setup(sos)
#         u = self.precond(sos,self.a_list, self.inv_a_list)
#         u = self._unpad(u, pad)
#         return u

# if __name__ == "__main__":
#     device = 'cuda'
#     model = MPMG(u_channels_list=[5,10,15,20,32],
#                  a_channels_list=[10,15,20,30,32],
#                  mg_levels=[1,2,4,8,8],
#                  fdtype=torch.float32,
#                  device=device)
#     x = torch.rand(10,1,70,70).to(device)

#     y = model(x)

#     print(y.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class coeff_model(nn.Module):
    """
    Pre-process the coefficient 'a' and share it over several levels.
    优化:
    1. 在GELU激活函数前加入InstanceNorm2d，增强对不同方差输入的适应性。
    2. 加深了inv_a_models，增强其拟合复杂系数的能力。
    3. 恢复了proj_inv_a_to_u的计算，让a_list和inv_a_list的通道数保持一致，并进行独立投影。
    """

    def __init__(self, a_channels_in, a_channels_list, u_channels_list, num_levels):
        super().__init__()
        self.num_levels = num_levels
        self.a_channels_in = a_channels_in
        self.a_channels_list = a_channels_list
        self.u_channels_list = u_channels_list

        self.lifting_a = nn.Conv2d(a_channels_in, a_channels_list[0], kernel_size=3, stride=1, padding=1, bias=True)

        self.inv_a_models = nn.ModuleList()
        self.restrict_a_models = nn.ModuleList()
        self.proj_a_to_u = nn.ModuleList()
        self.proj_inv_a_to_u = nn.ModuleList()

        for i in range(num_levels):
            # 增强 inv_a_models 的能力，并加入归一化
            self.inv_a_models.append(
                nn.Sequential(
                    nn.Conv2d(a_channels_list[i], a_channels_list[i], kernel_size=3, stride=1, padding=1, bias=False),
                    nn.InstanceNorm2d(a_channels_list[i]),
                    nn.GELU(),
                    nn.Conv2d(a_channels_list[i], a_channels_list[i], kernel_size=3, stride=1, padding=1, bias=True),
                )
            )
            if i < num_levels - 1:
                # 在下采样前加入归一化和激活
                self.restrict_a_models.append(
                    nn.Sequential(
                        nn.InstanceNorm2d(a_channels_list[i]),
                        nn.GELU(),
                        nn.Conv2d(a_channels_list[i], a_channels_list[i+1], kernel_size=3, stride=2, padding=1, bias=True)
                    )
                )

            # 投影层，同样加入归一化
            self.proj_a_to_u.append(
                nn.Sequential(
                    nn.InstanceNorm2d(a_channels_list[i]),
                    nn.GELU(),
                    nn.Conv2d(a_channels_list[i], u_channels_list[i], kernel_size=3, stride=1, padding=1, bias=True),
                )
            )
            self.proj_inv_a_to_u.append(
                nn.Sequential(
                    nn.InstanceNorm2d(a_channels_list[i]),
                    nn.GELU(),
                    nn.Conv2d(a_channels_list[i], u_channels_list[i], kernel_size=3, stride=1, padding=1, bias=True)
                )
            )

    def forward(self, a):
        a = self.lifting_a(a)
        a_list = [a]
        
        # 先生成所有层级的 a_list
        for i in range(self.num_levels - 1):
            a_list.append(self.restrict_a_models[i](a_list[i]))

        # 基于 a_list 计算 inv_a_list
        inv_a_list = [self.inv_a_models[i](a_list[i]) for i in range(self.num_levels)]

        # 独立地将 a_list 和 inv_a_list 投影到 u 的通道维度
        a_list_proj = [self.proj_a_to_u[i](a_list[i]) for i in range(self.num_levels)]
        inv_a_list_proj = [self.proj_inv_a_to_u[i](inv_a_list[i]) for i in range(self.num_levels)]

        return a_list_proj, inv_a_list_proj

class A_operator(nn.Module):
    '''
    Computes Au = A_operator(u,a)
    Shared over the levels.
    优化: 此处结构合理，无需改动。
    '''
    def __init__(self, u_channels, a_channels):
        super().__init__()
        # 注意: 这里的 a_channels 实际上是 u_channels, 因为 a 已经被 proj_a_to_u 投影了
        self.conv_u = nn.Conv2d(u_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_I = nn.Conv2d(u_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, u, a):
        # a 已经被投影到和 u 相同的通道数
        return self.conv_I(u) + a * self.conv_u(u)

class mg_iteration(nn.Module):
    '''
    Performs a single multigrid iteration:
      u = u + B(b - Au)
    优化:
    1. 合并了 mg_iteration_init 和 mg_iteration，通过 is_init_iter 标志位区分。
    2. 增加了 InstanceNorm2d 来稳定平滑操作。
    '''
    def __init__(self, u_channels, A_op):
        super().__init__()
        self.A = A_op
        # u_channels 和 inv_a 的通道数相同
        self.conv_res = nn.Conv2d(u_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_res = nn.InstanceNorm2d(u_channels)
        self.conv_B   = nn.Conv2d(u_channels, u_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_B   = nn.InstanceNorm2d(u_channels)

    def forward(self, out, is_init_iter=False):
        (u, a, inv_a, rhs, prev_res) = out
        
        if is_init_iter:
            # 第一次迭代，没有 u，直接处理 rhs
            res = rhs
            # 将 u 初始化为 0
            u = torch.zeros_like(rhs)
        else:
            # 后续迭代，计算残差
            res = rhs - self.A(u, a)

        # 平滑操作，B(res)
        delta_u = self.norm_res(self.conv_res(res))
        delta_u = inv_a * delta_u
        delta_u = self.norm_B(self.conv_B(delta_u))
        
        u = u + delta_u
        
        return u, a, inv_a, rhs, res


class mg_restriction(nn.Module):
    '''
    Restricts residual to the coarser level.
    优化: 合并了 mg_restriction_init 和 mg_restriction。
    '''
    def __init__(self, u_channels_in, u_channels_out):
        super().__init__()
        self.R_rhs = nn.Conv2d(u_channels_in, u_channels_out, kernel_size=3, stride=2, padding=1, bias=False)
    
    def forward(self, res):
        rhs_coarse = self.R_rhs(res)
        u_coarse = torch.zeros_like(rhs_coarse) # 在粗层级将 u 初始化为 0
        return u_coarse, rhs_coarse


class mg_prolongation(nn.Module):
    '''
    Prolongates u from the coarser level.
    结构合理，无需改动。
    '''
    def __init__(self, u_channels_in, u_channels_out):
        super().__init__()
        self.P_u = nn.ConvTranspose2d(u_channels_in, u_channels_out, kernel_size=4, stride=2, padding=1, bias=False)
    def forward(self, u):
        return self.P_u(u)


class mpmg_operator(nn.Module):
    '''
    Multigrid operator.
    优化:
    1. 取消了平滑迭代中的参数共享，为每次迭代创建独立的 mg_iteration 模块，以增加模型参数量和容量。
    2. 调整了 forward 逻辑以适应新的模块结构。
    '''
    def __init__(self, u_channels_list, a_channels_list, num_iteration=[1, 1, 1, 1, 1]):
        super().__init__()
        self.num_levels = len(num_iteration)
        self.num_iteration = num_iteration
        assert len(u_channels_list) == self.num_levels, "u_channels_list length must match number of levels"
        assert len(a_channels_list) == self.num_levels, "a_channels_list length must match number of levels"

        self.coeff_model = coeff_model(a_channels_in=3,
                                       a_channels_list=a_channels_list,
                                       u_channels_list=u_channels_list,
                                       num_levels=self.num_levels)

        self.A_list = nn.ModuleList([A_operator(u_channels=u_channels_list[j], a_channels=u_channels_list[j]) for j in range(self.num_levels)])
        self.R_list = nn.ModuleList([mg_restriction(u_channels_in=u_channels_list[j], u_channels_out=u_channels_list[j+1]) for j in range(self.num_levels - 1)])
        self.P_list = nn.ModuleList([mg_prolongation(u_channels_in=u_channels_list[j+1], u_channels_out=u_channels_list[j]) for j in range(self.num_levels - 1)])

        # -- 主要改动点: 为每次迭代创建独立的平滑器 --
        self.Down_sweep_layers = nn.ModuleList()
        self.Up_sweep_layers = nn.ModuleList()

        for l, num_iter in enumerate(num_iteration):
            # 为第 l 层的向下扫描创建 num_iter 个独立的平滑器
            down_smoothers_l = nn.ModuleList(
                [mg_iteration(u_channels=u_channels_list[l], A_op=self.A_list[l]) for _ in range(num_iter)]
            )
            self.Down_sweep_layers.append(down_smoothers_l)

            # 为第 l 层的向上扫描创建 num_iter 个独立的平滑器
            if l < self.num_levels - 1:
                up_smoothers_l = nn.ModuleList(
                    [mg_iteration(u_channels=u_channels_list[l], A_op=self.A_list[l]) for _ in range(num_iter)]
                )
                self.Up_sweep_layers.append(up_smoothers_l)

    def setup_coeff(self, a):
        return self.coeff_model(a)

    def forward(self, rhs, a_list, inv_a_list):
        u_list = [None] * self.num_levels
        rhs_list = [None] * self.num_levels
        
        # V-cycle: Down sweep
        u = None
        current_rhs = rhs
        
        for l in range(self.num_levels):
            rhs_list[l] = current_rhs
            level_down_smoothers = self.Down_sweep_layers[l]
            
            # 预平滑 (pre-smoothing)
            # -- 主要改动点: 顺序调用独立的平滑器 --
            res = None
            # 第一次迭代
            u, _, _, _, res = level_down_smoothers[0]((u, a_list[l], inv_a_list[l], current_rhs, None), is_init_iter=True)
            # 后续迭代
            for i in range(1, len(level_down_smoothers)):
                u, _, _, _, res = level_down_smoothers[i]((u, a_list[l], inv_a_list[l], current_rhs, res))
            
            u_list[l] = u
            
            if l < self.num_levels - 1:
                # 限制残差到更粗的层级作为新的rhs
                u, current_rhs = self.R_list[l](res)

        # V-cycle: Up sweep
        for l in range(self.num_levels - 2, -1, -1):
            # 从粗层级获取误差修正项并插值
            correction = self.P_list[l](u_list[l+1])
            u = u_list[l] + correction
            
            current_rhs = rhs_list[l]
            level_up_smoothers = self.Up_sweep_layers[l]
            
            # 后平滑 (post-smoothing)
            # -- 主要改动点: 顺序调用独立的平滑器 --
            for smoother in level_up_smoothers:
                 u, _, _, _, _ = smoother((u, a_list[l], inv_a_list[l], current_rhs, None))
            
            u_list[l] = u
            
        return u_list[0]

class MPMG_precond(nn.Module):
    def __init__(self,
                 u_channels_list=[2, 20, 30, 40, 50],
                 a_channels_list=[32, 32, 32, 32, 32],
                 mg_levels=[1, 2, 4, 8, 8]):
        super().__init__()
        torch.set_default_dtype(torch.float32)
        self.mg_levels = mg_levels
        self.lifting_rhs = nn.Conv2d(1, u_channels_list[0], kernel_size=1, stride=1, padding=0, bias=True)
        self.mg = mpmg_operator(u_channels_list=u_channels_list, a_channels_list=a_channels_list, num_iteration=mg_levels)
        self.proj_u = nn.Conv2d(u_channels_list[0], 1, kernel_size=1, stride=1, padding=0, bias=True)

    def setup(self, coeff):
        return self.mg.setup_coeff(coeff)
        
    def forward(self, rhs, a_list, inv_a_list):
        rhs = self.lifting_rhs(rhs)
        u = self.mg(rhs, a_list, inv_a_list)
        u = self.proj_u(u)
        return u


class MPMG(nn.Module):
    def __init__(self,
                 u_channels_list=[2, 20, 30, 40, 50],
                 a_channels_list=[32, 32, 32, 32, 32],
                 mg_levels=[1, 2, 4, 8, 8],
                 fdtype=torch.float32,
                 device='cuda'):
        super().__init__()
        self.float_dtype = fdtype
        self.device = device
        self.precond = MPMG_precond(u_channels_list,
                                    a_channels_list,
                                    mg_levels).to(device)

    def setup(self, sos):
        grid = self.get_grid(sos.shape, sos.device)
        coeff = torch.cat([sos, grid], dim=1).to(self.float_dtype)
        self.a_list, self.inv_a_list = self.precond.setup(coeff)

    def get_grid(self, shape, device):
        batchsize, _, size_x, size_y = shape
        gridx = torch.linspace(0, 1, size_x, dtype=self.float_dtype, device=device).reshape(1, 1, size_x, 1).expand(batchsize, 1, size_x, size_y)
        gridy = torch.linspace(0, 1, size_y, dtype=self.float_dtype, device=device).reshape(1, 1, 1, size_y).expand(batchsize, 1, size_x, size_y)
        return torch.cat([gridx, gridy], dim=1)

    @staticmethod
    def _pad_to_32(x):
        _, _, h, w = x.shape
        nh = math.ceil(h / 32) * 32
        nw = math.ceil(w / 32) * 32
        if nh == h and nw == w:
            return x, (0, 0, 0, 0)
        ph, pw = nh - h, nw - w
        top, bottom = ph // 2, ph - ph // 2
        left, right = pw // 2, pw - pw // 2
        return F.pad(x, (left, right, top, bottom)), (left, right, top, bottom)

    @staticmethod
    def _unpad(x, pad):
        left, right, top, bottom = pad
        if sum(pad) == 0:
            return x
        return x[:, :, top: x.size(2)-bottom, left: x.size(3)-right]

    def forward(self, sos):
        sos_padded, pad = self._pad_to_32(sos)
        self.setup(sos_padded)
        u = self.precond(sos_padded, self.a_list, self.inv_a_list)
        u = self._unpad(u, pad)
        return u

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MPMG(u_channels_list=[5,10,15,20,32],
                 a_channels_list=[10,15,20,30,32],
                 mg_levels=[1,2,4,8,8],
                 fdtype=torch.float32,
                 device=device)
    
    # 确保模型在正确的设备上
    model.to(device)
    
    x = torch.rand(10,1,70,70).to(device)

    y = model(x)

    print(f"Output shape: {y.shape}")
    print(f"Input was on {x.device}, output is on {y.device}")
