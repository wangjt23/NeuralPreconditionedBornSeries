import torch
import torch.nn as nn
from .laplace_fft import LaplacianFFTwithPML2D,LaplacianFFTwithPML3D,LaplacianComplexFFT2D,LaplacianComplexFFT3D


class helmholtz2d_op(nn.Module):
    '''
    Args:
        domain_size: int, size of computational domain + boundary domain (square domain)
        boundary_size: int, size of boundary domain
        dx: float, grid spacing
        abc_factor: float,  strength factor of absorbing layer
        float_type: torch.float32 or torch.float64 
        kind: str, 'fdm_abc', 'fft_pml', 'fdm_pml' and 'fft_abc'
    

    '''
    def __init__(self, domain_size, boundary_size, dx, abc_factor, k0=None, float_type=torch.float64 , kind='fdm_abc'):
        super().__init__()
        self.complex_type = torch.complex128 if float_type==torch.float64 else torch.complex64
        self.float_type = float_type
        self.kind = kind
        self.dx = dx
        self.k0 = k0
        if kind == 'fdm_abc':
            self.lap_op = nn.Conv2d(2,2,kernel_size=3,stride=1,padding=1,groups=2,bias=False)
            kernel = torch.tensor([[[[0.,  1., 0.],
                                     [1., -4., 1.],
                                     [0.,  1., 0]]]],dtype=float_type).repeat(2,1,1,1)
            self.lap_op.weight = nn.Parameter(kernel)
            self.lap_op.weight.requires_grad = False
            # self.absorb = nn.Parameter(self.set_abc_layer(domain_size, boundary_size, abc_factor),
            #                            requires_grad=False)
            self.register_buffer('absorb', self.set_abc_layer(domain_size, boundary_size, abc_factor),persistent=False)
            self.forward = self.fdm_abc_forward
        elif kind == 'fdm_pml':
            self.lap_op1 = nn.Conv2d(4,4,kernel_size=3,stride=1,padding=1,groups=4,bias=False)
            kernel1 = torch.tensor([[[[0.,  0., 0.],
                                      [1., -2., 1.],
                                      [0.,  0., 0.]]],

                                    [[[0.,  0., 0.],
                                      [1., -2., 1.],
                                      [0.,  0., 0.]]],

                                    [[[0.,  1., 0.],
                                      [0., -2., 0.],
                                      [0.,  1., 0.]]],

                                    [[[0.,  1., 0.],
                                      [0., -2., 0.],
                                      [0.,  1., 0.]]],
                                    ],dtype=float_type) / (self.dx**2)
            self.lap_op1.weight = nn.Parameter(kernel1)
            self.lap_op1.weight.requires_grad = False
            self.lap_op2 = nn.Conv2d(4,4,kernel_size=3,stride=1,padding=1,groups=4,bias=False)
            kernel2 = torch.tensor([[[[0.,  0., 0.],
                                      [-1., 0., 1.],
                                      [0.,  0., 0.]]],

                                    [[[0.,  0., 0.],
                                      [-1., 0., 1.],
                                      [0.,  0., 0.]]],

                                    [[[0., -1., 0.],
                                      [0.,  0., 0.],
                                      [0.,  1., 0.]]],

                                    [[[0., -1., 0.],
                                      [0.,  0., 0.],
                                      [0.,  1., 0.]]],
                                    ],dtype=float_type) / (2 * self.dx)
            self.lap_op2.weight = nn.Parameter(kernel2)
            self.lap_op2.weight.requires_grad = False
            absorb, coeff1, coeff2 = self.set_pml_utils(domain_size,PMLsize=boundary_size,sigma_max=abc_factor)
            self.register_buffer('absorb', absorb.unsqueeze(0).unsqueeze(0),persistent=False)
            self.register_buffer('coeff1', coeff1.unsqueeze(0),persistent=False)
            self.register_buffer('coeff2', coeff2.unsqueeze(0),persistent=False)
            self.forward = self.fdm_pml_forward


        elif kind == 'fft_abc':
            self.lap_op = LaplacianComplexFFT2D(domain_size=domain_size, dx=dx, complex_dtype=self.complex_type)
            # self.absorb = nn.Parameter(self.set_abc_layer(domain_size, boundary_size, abc_factor),
            #                            requires_grad=False)
            self.register_buffer('absorb', self.set_abc_layer(domain_size, boundary_size, abc_factor),persistent=False)
            self.forward = self.fft_abc_forward
        elif kind == 'fft_pbl':
            self.lap_op = LaplacianComplexFFT2D(domain_size=domain_size, dx=dx, complex_dtype=self.complex_type)
            # self.absorb = nn.Parameter(self.set_abc_layer(domain_size, boundary_size, abc_factor),
            #                            requires_grad=False)
            self.register_buffer('absorb', self.set_pbl_layer(domain_size, boundary_size),persistent=False)
            self.forward = self.fft_pbl_forward
        elif kind == 'fft_pml':
            self.lap_op = LaplacianFFTwithPML2D(domain_size=domain_size,PMLsize=boundary_size,dx=dx,
                                                sigma_max=abc_factor,complex_dtype=self.complex_type)
            # self.absorb = nn.Parameter(self.lap_op.sigma_2d.unsqueeze(0).unsqueeze(0),requires_grad=False)
            self.register_buffer('absorb', self.lap_op.sigma_2d.unsqueeze(0).unsqueeze(0),persistent=False)
            self.forward = self.fft_pml_forward
        else:
            raise NotImplementedError("no such method")
        
            
    def fdm_abc_forward(self, u, k):
        '''
        input:
            u: complex tensor, batch,1,N,N
            kh: float tensor, batch,1,N,N
        return:
            Au: complex tensor, batch,1,N,N
        '''
        # u = torch.view_as_real(u.squeeze(1)).permute(0,3,1,2)#.contiguous()
        u = torch.cat([u.real, u.imag], dim=1)#.to(torch.float32)
        lap_u = self.lap_op(u)
        kh2 = torch.pow(k*self.dx, 2)
        Au = -lap_u - kh2 * u
        gamma_i_u_imag = self.absorb * u[:, 0:1, :, :]
        gamma_i_u_real =-self.absorb * u[:, 1:2, :, :]
        
        Au[:, 0:1, :, :] += gamma_i_u_real*kh2
        Au[:, 1:2, :, :] += gamma_i_u_imag*kh2
        Au = torch.view_as_complex(Au.permute(0,2,3,1).unsqueeze(1).contiguous())
        return Au
    
    def fdm_pml_forward(self, u, k):
        u_ = torch.cat([u.real, u.imag, u.real, u.imag], dim=1)
        lap1 = torch.view_as_complex(self.lap_op1(u_).reshape(u.size(0),2,2,u.size(2),u.size(3)).movedim(2, -1).contiguous())
        lap2 = torch.view_as_complex(self.lap_op2(u_).reshape(u.size(0),2,2,u.size(2),u.size(3)).movedim(2, -1).contiguous())
        lap_pml_u = (self.coeff1*lap1 + self.coeff2*lap2).sum(dim=1, keepdim=True)
        k2 = torch.pow(k, 2)
        Au = (-lap_pml_u - k2*u)*(self.dx**2)
        return Au

    def fft_abc_forward(self, u, k):
        '''
        input:
            u: complex tensor, batch,1,N,N
            kh: float tensor, batch,1,N,N
        return:
            Au: complex tensor, batch,1,N,N
        '''
        lap_u = self.lap_op(u)
        k2 = torch.pow(k, 2)
        Au = (-lap_u - (1-1.j*self.absorb)*k2*u)*(self.dx**2)

        return Au
    
    def fft_pbl_forward(self, u, k):
        '''
        input:
            u: complex tensor, batch,1,N,N
            kh: float tensor, batch,1,N,N
        return:
            Au: complex tensor, batch,1,N,N
        '''
        lap_u = self.lap_op(u)
        k2 = torch.pow(k, 2)
        Au = (-lap_u - (1+self.absorb)*k2*u)*(self.dx**2)

        return Au
    def fft_pml_forward(self, u, k):
        '''
        input:
            u: complex tensor, batch,1,N,N
            kh: float tensor, batch,1,N,N
        return:
            Au: complex tensor, batch,1,N,N
        '''
        lap_u = self.lap_op(u)
        k2 = torch.pow(k, 2)
        Au = (-lap_u - k2*u)*(self.dx**2)
        return Au

    def set_abc_layer(self, domain_size, width, value):
        abc_coord = torch.arange(width)
        # Sigma function (quadratic)
        layer_outer = value * (1 - abc_coord / width) ** 2
        layer_1d = torch.zeros((domain_size,),dtype=self.float_type)
        layer_1d[:width] = layer_outer
        layer_1d[-width:] = torch.flip(layer_outer, dims=[0])
        layer_x_2d, layer_y_2d = torch.meshgrid(layer_1d, layer_1d, indexing='xy')
        out = (layer_x_2d + layer_y_2d).unsqueeze(0).unsqueeze(0)
        torch.clamp_(out, max=value)
        return out
    def set_pml_utils(self, domain_size, PMLsize, sigma_max):
        pml_coord = torch.arange(PMLsize, dtype=self.float_type)
        # Sigma function (quadratic)
        sigma_outer = sigma_max * (1 - pml_coord / PMLsize) ** 2
        sigma_1d = torch.zeros((domain_size,), dtype=self.float_type)
        sigma_1d[:PMLsize] = sigma_outer
        sigma_1d[-PMLsize:] = torch.flip(sigma_outer, dims=[0])
        sigma_x_2d, sigma_y_2d = torch.meshgrid(sigma_1d, sigma_1d, indexing='xy')
        sigma_2d = torch.clamp(sigma_x_2d+sigma_y_2d,max=sigma_max)
        # sigma_2d = torch.nn.Parameter(sigma_2d, requires_grad=False)

        # Inverse gamma 
        # inv_gamma = 1 / (1 + j * sigma / omega0)
        # where omega0 = omega/mean(c) in heterogeneous medium
        inv_gamma_x = 1.0 / (torch.ones_like(sigma_x_2d) + (1j * sigma_x_2d))
        inv_gamma_y = 1.0 / (torch.ones_like(sigma_y_2d) + (1j * sigma_y_2d))
        # derivative of sigma
        # Dsigma/Dz = -2 * sigma_max / L_pml * (1 - z/L_pml)
        sigma_prime_outer = -2 * sigma_max / PMLsize * (1 - pml_coord / PMLsize)
        sigma_prime_1d = torch.zeros((domain_size,), dtype=self.float_type)
        sigma_prime_1d[:PMLsize] = sigma_prime_outer
        sigma_prime_1d[-PMLsize:] = -torch.flip(sigma_prime_outer, dims=[0])
        sigma_x_partial_2d, sigma_y_partial_2d = torch.meshgrid(sigma_prime_1d, sigma_prime_1d, indexing='xy')
        # derivative of Gamma
        # gamma_partial = (j / omega0) * sigma_partial
        gamma_x_partial = 1j * sigma_x_partial_2d
        gamma_y_partial = 1j * sigma_y_partial_2d
        # Coefficients for the modified Laplacian
        # L = ax dx' + bx dx'' + ay dy' + by dy''
        # ax = -gamma_x_prime * inv_gamma_x^3
        # bx = inv_gamma_x^2
        ax = -gamma_x_partial * (inv_gamma_x ** 3)
        bx = inv_gamma_x ** 2
        ay = -gamma_y_partial * (inv_gamma_y ** 3)
        by = inv_gamma_y ** 2
        coeff1 = torch.cat([bx.unsqueeze(0),by.unsqueeze(0)],dim=0)
        coeff2 = torch.cat([ax.unsqueeze(0),ay.unsqueeze(0)],dim=0)
        return sigma_2d, coeff1, coeff2
    
    def set_pbl_layer(self, domain_size, boundary_size):
        layer_outer = torch.arange(boundary_size, 0, -1)
        layer_1d = torch.zeros((domain_size,),dtype=self.float_type)
        layer_1d[:boundary_size] = layer_outer
        layer_1d[-boundary_size:] = torch.flip(layer_outer, dims=[0])
        boundary_dist = torch.sqrt(layer_1d**2 + (layer_1d.view(-1,1))**2)
        r = boundary_dist
        k0 = self.k0
        c = k0/2
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
        return f_boundary_curve.unsqueeze(0).unsqueeze(0)
    
class helmholtz3d_op(nn.Module):
    '''
    Args:
        domain_size: int, size of computational domain + boundary domain (square domain)
        boundary_size: int, size of boundary domain
        dx: float, grid spacing
        abc_factor: float,  strength factor of absorbing layer
        float_type: torch.float32 or torch.float64 
        kind: str, 'fft_pml', TODO: 'fdm_pml' 'fdm_abc' and 'fft_abc' 
    

    '''
    def __init__(self, domain_size, boundary_size, dx, abc_factor,float_type, kind='fft_pml'):
        super().__init__()
        self.complex_type = torch.complex128 if float_type==torch.float64 else torch.complex64
        self.float_type = float_type
        self.kind = kind
        self.dx = dx
        if kind == 'fft_pml':
            self.lap_op = LaplacianFFTwithPML3D(domain_size=domain_size,PMLsize=boundary_size,dx=dx,
                                                sigma_max=abc_factor,complex_dtype=self.complex_type)
            self.forward = self.fft_pml_forward
        elif kind == 'fdm_abc':
            # 3D 7-point Laplacian (no 1/dx^2 scaling; we incorporate dx in k)
            self.lap_op = nn.Conv3d(2, 2, kernel_size=3, stride=1, padding=1, groups=2, bias=False)
            k = torch.zeros((1, 1, 3, 3, 3), dtype=float_type)
            k[..., 1, 1, 1] = -6.0
            k[..., 0, 1, 1] = 1.0
            k[..., 2, 1, 1] = 1.0
            k[..., 1, 0, 1] = 1.0
            k[..., 1, 2, 1] = 1.0
            k[..., 1, 1, 0] = 1.0
            k[..., 1, 1, 2] = 1.0
            kernel = k.repeat(2, 1, 1, 1, 1)
            self.lap_op.weight = nn.Parameter(kernel)
            self.lap_op.weight.requires_grad = False

            # Absorbing boundary layer (quadratic profile), shape: (1,1,D,H,W)
            self.register_buffer('absorb', self.set_abc_layer_3d(domain_size, boundary_size, abc_factor))

            self.forward = self.fdm_abc_forward
        else:
            raise NotImplementedError("no such method")
            

    def fft_pml_forward(self, u, k):
        lap_u = self.lap_op(u)
        k2 = torch.pow(k, 2)
        Au = (-lap_u - k2*u)*(self.dx**2)
        return Au
    
    def fdm_abc_forward(self, u, k):
        '''
        input:
            u: complex tensor, shape (B, 1, D, H, W)
            k: real tensor, wavenumber field, shape (B, 1, D, H, W)
        return:
            Au: complex tensor, shape (B, 1, D, H, W)
        Notes:
            - Discrete Laplacian is unscaled by 1/dx^2; we instead use (kh*dx)^2.
            - ABC adds i * absorb * (kh*dx)^2 * u near the boundaries.
        '''
        # Split into real/imag channels for grouped conv
        u_ri = torch.cat([u.real, u.imag], dim=1)  # (B,2,D,H,W)

        lap_u = self.lap_op(u_ri)  # (B,2,D,H,W)

        kh2 = torch.pow(k * self.dx, 2)  # (B,1,D,H,W)

        Au = -lap_u - kh2 * u_ri  # (B,2,D,H,W)

        # Add i * absorb * kh2 * u  ->  real += (-absorb * imag) * kh2 ; imag += (absorb * real) * kh2
        gamma_i_u_imag = self.absorb * u_ri[:, 0:1, ...]   # absorb * Re(u)
        gamma_i_u_real = -self.absorb * u_ri[:, 1:2, ...]  # -absorb * Im(u)

        Au[:, 0:1, ...] += gamma_i_u_real * kh2
        Au[:, 1:2, ...] += gamma_i_u_imag * kh2

        # Recombine to complex
        Au = torch.view_as_complex(Au.permute(0, 2, 3, 4, 1).unsqueeze(1).contiguous())
        return Au

    def set_abc_layer_3d(self, domain_size, width, value):
        # Quadratic profile from boundary towards interior
        abc_coord = torch.arange(width, dtype=self.float_type)
        layer_outer = value * (1 - abc_coord / width) ** 2  # (width,)
        layer_1d = torch.zeros((domain_size,), dtype=self.float_type)
        layer_1d[:width] = layer_outer
        layer_1d[-width:] = torch.flip(layer_outer, dims=[0])

        # 3D separable construction
        lx, ly, lz = torch.meshgrid(layer_1d, layer_1d, layer_1d, indexing='ij')
        out = (lx + ly + lz).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        torch.clamp_(out, max=value)
        return out
    

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    domain_size=256
    boundary_size=16
    dx=0.01
    abc_factor=1.
    float_type = torch.float64
    complex_type = torch.complex128 if float_type==torch.float64 else torch.complex64
    # kind='fft_abc'
    device = 'cuda'
    helm_op = helmholtz2d_op(domain_size=domain_size,
                             boundary_size=boundary_size,
                             dx=dx,
                             k0=1.,
                             abc_factor=abc_factor,
                             float_type=float_type,
                             kind='fdm_abc').to(device)
    helm_op2 = helmholtz2d_op(domain_size=domain_size,
                            boundary_size=boundary_size,
                            dx=dx,
                            k0 = 1,
                            abc_factor=abc_factor,
                            float_type=float_type,
                            kind='fft_pbl').to(device)
    helm_op3 = helmholtz3d_op(domain_size=100,
                            boundary_size=boundary_size,
                            dx=dx,
                            # k0=1,
                            abc_factor=abc_factor,
                            float_type=float_type,
                            kind='fdm_abc').to(device)
    u = torch.sin(torch.arange(0,256**2*2).view(1,2,256,256)).to(dtype=float_type).to(device)
    u_complex = torch.view_as_complex(u.permute(0,2,3,1).unsqueeze(1).contiguous())

    u3d = torch.sin(torch.arange(0,100**3*2).view(1,2,100,100,100)).to(dtype=float_type).to(device)
    u3d_complex = torch.view_as_complex(u3d.permute(0,2,3,4,1).unsqueeze(1).contiguous())
    # print(u3d_complex.shape)
    k = torch.sin(torch.arange(0,256**2).view(1,1,256,256)/10).to(dtype=float_type).to(device)
    k3d = torch.sin(torch.arange(0,100**3).view(1,1,100,100,100)/10).to(dtype=float_type).to(device)
    # warm up
    for i in range(20):
        Au = helm_op(u_complex,k)
        Au2 = helm_op2(u_complex,k)
    s1 = time.perf_counter()
    for i in range(10):
        Au = helm_op(u_complex,k)
    s2 = time.perf_counter()
    for i in range(10):
        Au2 = helm_op2(u_complex,k)
    s3 = time.perf_counter()
    Au3 = helm_op3(u3d_complex,k3d)
    print(Au3.shape)
    # print(torch.norm(Au-Au2)/torch.norm(Au))
    print(f"fdm_abc: {s2-s1:.4e}s")
    print(f"fft_pml: {s3-s2:.4e}s")

    # print(torch.norm(Au2-Au)/torch.norm(Au2))
    # plt.imshow(Au.real[0,0].cpu())
    # plt.colorbar()
    # plt.savefig('test.png')
    # plt.figure()
    # plt.imshow(Au2.real[0,0].cpu())
    # plt.colorbar()
    # plt.savefig('test2.png')
    # plt.figure()
    # plt.imshow((Au-Au2).real[0,0].cpu())
    # plt.colorbar()
    # plt.savefig('test3.png')
    # print(Au.shape)