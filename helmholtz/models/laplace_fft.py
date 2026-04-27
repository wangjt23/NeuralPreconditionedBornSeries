'''
Implementations of FFT-based 2D and 3D Laplacian operator in PyTorch for both real and complex inputs, 
with optional support for Perfectly Matched Layer (PML) boundary conditions.

- Author: Juntao Wang
- Date: 2025-05-19
'''
import torch
from torch import nn

## 2D version
class LaplacianRealFFT2D(nn.Module):
    def __init__(self, 
                 domain_size: int, 
                 dx         : float,
                 float_dtype: torch.dtype = torch.float32):
        """
        Pure Laplace Operator with FFT discretization for real-valued input.
        Uses rfftn for efficiency.
        Args:
            domain_size (int): Grid size of each dimension (supports square domain).
            dx          (float): Grid spacing.
            float_dtype (torch.dtype, optional): The float type for input and computation.
                                                 Default is torch.float32.
        """
        super().__init__()
        self.domain_size = domain_size
        self.dx = dx
        self.float_dtype = float_dtype

        # Generate Laplacian kernel in Fourier space for rfft
        # kx for rfft covers only non-negative frequencies
        # The size will be domain_size // 2 + 1
        k_1d_x_rfft = 2*torch.pi*torch.fft.rfftfreq(self.domain_size, d=self.dx)
        # ky covers the full frequency spectrum
        k_1d_y      = 2*torch.pi*torch.fft.fftfreq(self.domain_size, d=self.dx)
        kx_2d, ky_2d = torch.meshgrid(k_1d_x_rfft, k_1d_y, indexing='xy')
        # Laplacian operator in Fourier space: -(kx^2 + ky^2)
        lap_fourier = -(kx_2d**2 + ky_2d**2)
        
        # (1, domain_size, domain_size // 2 + 1)
        self.lap_fourier_rfft = torch.nn.Parameter(lap_fourier.unsqueeze(0).to(self.float_dtype), 
                                                   requires_grad=False)

    def forward(self, u: torch.Tensor):
        """
        Apply Laplacian to a real-valued input tensor u.

        Args:
            u (float tensor)    : (batch_size, domain_size, domain_size).
        Returns:
            lap_u (float tensor): (batch_size, domain_size, domain_size).
        """
        if not u.is_floating_point() or u.is_complex():
            raise TypeError(f"Input tensor u must be real floating point, but got {u.dtype}")
        u_fft = torch.fft.rfftn(u, dim=(-2, -1), norm="backward")
        print(self.lap_fourier_rfft.shape)
        print(u_fft.shape)
        lap_u_fft = u_fft * self.lap_fourier_rfft
        # Output shape: (batch_size, domain_size, domain_size)
        lap_u = torch.fft.irfftn(lap_u_fft, 
                                 s=(self.domain_size, self.domain_size), 
                                 dim=(-2, -1), 
                                 norm="backward")
        return lap_u
    
class LaplacianComplexFFT2D(nn.Module):
    def __init__(self, 
                 domain_size: int, 
                 dx         : float,
                 complex_dtype: torch.dtype = torch.complex64):
        """
        Pure Laplace Operator with FFT discretization For complex input

        Args:
            domain_size   : int,   grid size of each dimension (only support square domain)
            dx            : float, Grid spacing.
            complex_dtype : torch.dtype, optional, The type used to calculate. Default is torch.complex64.
        """
        super().__init__()
        self.domain_size = domain_size
        self.complex_dtype = complex_dtype
        self.float_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
        self.dx = dx
        # generate laplacian in fourier space
        k_1d = 2*torch.pi*torch.fft.fftfreq(self.domain_size, d=self.dx)
        kx_2d, ky_2d = torch.meshgrid(k_1d, k_1d, indexing='xy')
        # F(Laplacian(u)) = F(d^2u/dx^2 + d^2u/dy^2) = -(kx^2 + ky^2) * F(u)
        lap_fourier = -(kx_2d**2 + ky_2d**2)
        self.lap_fourier = torch.nn.Parameter(lap_fourier.unsqueeze(0).to(self.float_dtype), 
                                              requires_grad=False)

    def forward(self, u: torch.Tensor):
        """
        Apply laplacian to u

        Args:
            u (complex or float tensor): (batch_size, domain_size, domain_size) 
        return:
            lap_u (complex tensor)     : (batch_size, domain_size, domain_size) 
        """
        u_fft = torch.fft.fftn(u, dim=(-2, -1), norm="backward")
        lap_u_fft = u_fft * self.lap_fourier
        lap_u = torch.fft.ifftn(lap_u_fft, dim=(-2, -1), norm="backward")
        return lap_u

class LaplacianFFTwithPML2D(nn.Module):
    def __init__(self, 
                 domain_size: int, 
                 PMLsize: int, 
                 dx: float,
                 sigma_max: float,
                 complex_dtype: torch.dtype = torch.complex64):
        """
        Laplace Operator with Perfectly Matched Layer be discrete by FFT

        Args:
            domain_size   : int,   grid size of each dimension. (only support square domain)
            PMLsize       : int,   the PML width each side.
            dx            : float, Grid spacing.
            sigma_max     : float, PML strength factor.
            complex_dtype : torch.dtype, optional, The type used to calculate. Default is torch.complex64.
        """
        super().__init__()
        self.domain_size = domain_size
        self.PMLsize = PMLsize
        self.sigma_max = sigma_max
        self.complex_dtype = complex_dtype
        self.dx = dx
        self.float_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
        self.init_variables()

    def init_variables(self):
        k_1d = 2 * torch.pi * torch.fft.fftfreq(self.domain_size, d=self.dx)
        kx_real_2d, ky_real_2d = torch.meshgrid(k_1d, k_1d, indexing='xy') 
        # F(du/dx) = ikx*F(u) Shape [1, domain_size, domain_size]
        self.kx_i = torch.nn.Parameter((1j * kx_real_2d).unsqueeze(0).to(self.complex_dtype), requires_grad=False)
        self.ky_i = torch.nn.Parameter((1j * ky_real_2d).unsqueeze(0).to(self.complex_dtype), requires_grad=False)
        # F(d2u/dx2) = -kx^2*F(u) Shape [1, domain_size, domain_size]
        self.kx_i_sq = torch.nn.Parameter((-kx_real_2d**2).unsqueeze(0).to(self.complex_dtype), requires_grad=False)
        self.ky_i_sq = torch.nn.Parameter((-ky_real_2d**2).unsqueeze(0).to(self.complex_dtype), requires_grad=False)
        # Gamma related coefficients (ax, bx, ay, by)
        # shape [1, domain_size, domain_size]
        ax, bx, ay, by = self.get_laplacian_coeffs()
        self.ax = torch.nn.Parameter(ax, requires_grad=False)
        self.bx = torch.nn.Parameter(bx, requires_grad=False)
        self.ay = torch.nn.Parameter(ay, requires_grad=False)
        self.by = torch.nn.Parameter(by, requires_grad=False)

    def get_laplacian_coeffs(self):
        pml_coord = torch.arange(self.PMLsize, dtype=self.float_dtype)
        # Sigma function (quadratic)
        sigma_outer = self.sigma_max * (1 - pml_coord / self.PMLsize) ** 2
        sigma_1d = torch.zeros((self.domain_size,), dtype=self.float_dtype)
        sigma_1d[:self.PMLsize] = sigma_outer
        sigma_1d[-self.PMLsize:] = torch.flip(sigma_outer, dims=[0])
        sigma_x_2d, sigma_y_2d = torch.meshgrid(sigma_1d, sigma_1d, indexing='xy')
        sigma_2d = torch.clamp(sigma_x_2d+sigma_y_2d,max=self.sigma_max)
        self.sigma_2d = torch.nn.Parameter(sigma_2d, requires_grad=False)
        # Inverse gamma 
        # inv_gamma = 1 / (1 + j * sigma / omega0)
        # where omega0 = omega/mean(c) in heterogeneous medium
        inv_gamma_x = 1.0 / (torch.ones_like(sigma_x_2d) + (1j * sigma_x_2d))
        inv_gamma_y = 1.0 / (torch.ones_like(sigma_y_2d) + (1j * sigma_y_2d))
        # derivative of sigma
        # Dsigma/Dz = -2 * sigma_max / L_pml * (1 - z/L_pml)
        sigma_prime_outer = -2 * self.sigma_max / self.PMLsize * (1 - pml_coord / self.PMLsize)
        sigma_prime_1d = torch.zeros((self.domain_size,), dtype=self.float_dtype)
        sigma_prime_1d[:self.PMLsize] = sigma_prime_outer
        sigma_prime_1d[-self.PMLsize:] = -torch.flip(sigma_prime_outer, dims=[0])
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
        # Unsqueeze for batch dimension
        return ax.unsqueeze(0), bx.unsqueeze(0), ay.unsqueeze(0), by.unsqueeze(0)

    def forward(self, u: torch.Tensor):
        """
        Apply laplacian with PML to u

        Args:
            u (complex or float tensor): (batch_size, domain_size, domain_size) 
        return:
            lappml_u (complex tensor)  : (batch_size, domain_size, domain_size) 
        """
        u_fft = torch.fft.fftn(u, dim=(-2, -1), norm="backward")
        dx_fft  = u_fft * self.kx_i
        dy_fft  = u_fft * self.ky_i
        ddx_fft = u_fft * self.kx_i_sq
        ddy_fft = u_fft * self.ky_i_sq
        derivatives_fft_stack = torch.stack([dx_fft, dy_fft, ddx_fft, ddy_fft], dim=0)

        derivatives = torch.fft.ifftn(derivatives_fft_stack, dim=(-2, -1), norm="backward")
        dx  = derivatives[0]
        dy  = derivatives[1]
        ddx = derivatives[2]
        ddy = derivatives[3]
        # expansion of laplace with PML
        return self.ax*dx+self.ay*dy+self.bx*ddx+self.by*ddy
## 3D version

class LaplacianRealFFT3D(nn.Module):
    def __init__(self,
                 domain_size: int,
                 dx         : float,
                 float_dtype: torch.dtype = torch.float32):
        """
        3D Pure Laplace Operator with FFT discretization for real-valued input.
        Uses rfftn for efficiency.

        Args:
            domain_size (int): Grid size of each dimension (supports cubic domain).
            dx          (float): Grid spacing.
            float_dtype (torch.dtype, optional): The float type for input and computation.
                                                 Default is torch.float32.
        """
        super().__init__()
        self.domain_size = domain_size
        self.dx = dx
        self.float_dtype = float_dtype

        # Generate Laplacian kernel in Fourier space for rfftn (3D)
        # Frequencies for the first two dimensions (transformed by fft)
        k_1d_dim1 = 2 * torch.pi * torch.fft.fftfreq(self.domain_size, d=self.dx)
        k_1d_dim2 = 2 * torch.pi * torch.fft.fftfreq(self.domain_size, d=self.dx)
        # Frequencies for the last dimension (transformed by rfft)
        # The size will be domain_size // 2 + 1
        k_1d_dim3_rfft = 2 * torch.pi * torch.fft.rfftfreq(self.domain_size, d=self.dx)

        # Create 3D grids of wavevectors
        # K_dim1_grid corresponds to frequencies for the first transformed dimension
        # K_dim2_grid corresponds to frequencies for the second transformed dimension
        # K_dim3_rfft_grid corresponds to frequencies for the third (rfft) transformed dimension
        K_dim1_grid, K_dim2_grid, K_dim3_rfft_grid = torch.meshgrid(
            k_1d_dim1, k_1d_dim2, k_1d_dim3_rfft, indexing='xy'
        )

        # Laplacian operator in Fourier space: -(k_dim1^2 + k_dim2^2 + k_dim3^2)
        lap_fourier = -(K_dim1_grid**2 + K_dim2_grid**2 + K_dim3_rfft_grid**2)

        # Shape: (1, domain_size, domain_size, domain_size // 2 + 1)
        self.lap_fourier_rfft = torch.nn.Parameter(lap_fourier.unsqueeze(0).to(self.float_dtype),
                                                   requires_grad=False)

    def forward(self, u: torch.Tensor):
        """
        Apply 3D Laplacian to a real-valued input tensor u.

        Args:
            u (float tensor)    : (batch_size, domain_size, domain_size, domain_size).
        Returns:
            lap_u (float tensor): (batch_size, domain_size, domain_size, domain_size).
        """
        if not u.is_floating_point() or u.is_complex():
            raise TypeError(f"Input tensor u must be real floating point, but got {u.dtype}")

        # rfftn over the last three dimensions
        u_fft = torch.fft.rfftn(u, dim=(-3, -2, -1), norm="backward")
        
        lap_u_fft = u_fft * self.lap_fourier_rfft
        
        # irfftn back to real space
        # Output shape: (batch_size, domain_size, domain_size, domain_size)
        lap_u = torch.fft.irfftn(lap_u_fft,
                                 s=(self.domain_size, self.domain_size, self.domain_size),
                                 dim=(-3, -2, -1),
                                 norm="backward")
        return lap_u
    

class LaplacianComplexFFT3D(nn.Module):
    def __init__(self,
                 domain_size: int,
                 dx         : float,
                 complex_dtype: torch.dtype = torch.complex64):
        """
        3D Pure Laplace Operator with FFT discretization for complex input.

        Args:
            domain_size   : int,   grid size of each dimension (only support cubic domain).
            dx            : float, Grid spacing.
            complex_dtype : torch.dtype, optional, The type used to calculate. Default is torch.complex64.
        """
        super().__init__()
        self.domain_size = domain_size
        self.complex_dtype = complex_dtype
        # Determine float_dtype based on complex_dtype for the kernel
        self.float_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
        self.dx = dx

        # Generate laplacian kernel in Fourier space for 3D
        k_1d = 2 * torch.pi * torch.fft.fftfreq(self.domain_size, d=self.dx)
        
        kx_3d, ky_3d, kz_3d = torch.meshgrid(k_1d, k_1d, k_1d, indexing='xy')
        
        # F(Laplacian(u)) = -(kx^2 + ky^2 + kz^2) * F(u)
        lap_fourier = -(kx_3d**2 + ky_3d**2 + kz_3d**2)
        
        # Store the kernel with float_dtype; it will be broadcasted to complex during multiplication
        # Shape: (1, domain_size, domain_size, domain_size)
        self.lap_fourier = torch.nn.Parameter(lap_fourier.unsqueeze(0).to(self.float_dtype),
                                              requires_grad=False)

    def forward(self, u: torch.Tensor):
        """
        Apply 3D Laplacian to u.

        Args:
            u (complex or float tensor): (batch_size, domain_size, domain_size, domain_size).
                                         If float, it will be converted to complex by fftn.
        return:
            lap_u (complex tensor)     : (batch_size, domain_size, domain_size, domain_size).
        """
        # fftn over the last three dimensions
        u_fft = torch.fft.fftn(u, dim=(-3, -2, -1), norm="backward")
        
        lap_u_fft = u_fft * self.lap_fourier # lap_fourier (float) is promoted to complex
        
        # ifftn back to spatial domain
        lap_u = torch.fft.ifftn(lap_u_fft, dim=(-3, -2, -1), norm="backward")
        
        return lap_u
    

class LaplacianFFTwithPML3D(nn.Module):
    def __init__(self,
                 domain_size: int,
                 PMLsize: int,
                 dx: float,
                 sigma_max: float,
                 complex_dtype: torch.dtype = torch.complex64):
        """
        3D Laplace Operator with Perfectly Matched Layer be discrete by FFT

        Args:
            domain_size   : int,   grid size of each dimension. (only support cubic domain)
            PMLsize       : int,   the PML width each side.
            dx            : float, Grid spacing.
            sigma_max     : float, PML strength factor.
            complex_dtype : torch.dtype, optional, The type used to calculate. Default is torch.complex64.
        """
        super().__init__()
        self.domain_size = domain_size
        self.PMLsize = PMLsize
        self.sigma_max = sigma_max
        self.complex_dtype = complex_dtype
        self.dx = dx
        self.float_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
        self.init_variables()

    def init_variables(self):
        k_1d = 2 * torch.pi * torch.fft.fftfreq(self.domain_size, d=self.dx)
        # For 3D, meshgrid needs three k_1d inputs
        kx_real_3d, ky_real_3d, kz_real_3d = torch.meshgrid(k_1d, k_1d, k_1d, indexing='xy')

        # F(du/dx) = ikx*F(u) Shape [1, domain_size, domain_size, domain_size]
        self.kx_i = torch.nn.Parameter((1j * kx_real_3d).unsqueeze(0).to(self.complex_dtype), requires_grad=False)
        self.ky_i = torch.nn.Parameter((1j * ky_real_3d).unsqueeze(0).to(self.complex_dtype), requires_grad=False)
        self.kz_i = torch.nn.Parameter((1j * kz_real_3d).unsqueeze(0).to(self.complex_dtype), requires_grad=False)

        # F(d2u/dx2) = -kx^2*F(u) Shape [1, domain_size, domain_size, domain_size]
        self.kx_i_sq = torch.nn.Parameter((-kx_real_3d**2).unsqueeze(0).to(self.complex_dtype), requires_grad=False)
        self.ky_i_sq = torch.nn.Parameter((-ky_real_3d**2).unsqueeze(0).to(self.complex_dtype), requires_grad=False)
        self.kz_i_sq = torch.nn.Parameter((-kz_real_3d**2).unsqueeze(0).to(self.complex_dtype), requires_grad=False)

        # Gamma related coefficients (ax, bx, ay, by, az, bz)
        # shape [1, domain_size, domain_size, domain_size]
        ax, bx, ay, by, az, bz = self.get_laplacian_coeffs()
        self.ax = torch.nn.Parameter(ax, requires_grad=False)
        self.bx = torch.nn.Parameter(bx, requires_grad=False)
        self.ay = torch.nn.Parameter(ay, requires_grad=False)
        self.by = torch.nn.Parameter(by, requires_grad=False)
        self.az = torch.nn.Parameter(az, requires_grad=False)
        self.bz = torch.nn.Parameter(bz, requires_grad=False)

    def get_laplacian_coeffs(self):
        pml_coord = torch.arange(self.PMLsize, dtype=self.float_dtype)
        # Sigma function (quadratic)
        sigma_outer = self.sigma_max * (1 - pml_coord / self.PMLsize) ** 2
        sigma_1d = torch.zeros((self.domain_size,), dtype=self.float_dtype)
        sigma_1d[:self.PMLsize] = sigma_outer
        sigma_1d[-self.PMLsize:] = torch.flip(sigma_outer, dims=[0])

        # For 3D
        sigma_x_3d, sigma_y_3d, sigma_z_3d = torch.meshgrid(sigma_1d, sigma_1d, sigma_1d, indexing='xy') 
        self.sigma_3d = torch.nn.Parameter(sigma_x_3d + sigma_y_3d + sigma_z_3d, requires_grad=False) # Total sigma

        # Inverse gamma
        inv_gamma_x = 1.0 / (torch.ones_like(sigma_x_3d) + (1j * sigma_x_3d))
        inv_gamma_y = 1.0 / (torch.ones_like(sigma_y_3d) + (1j * sigma_y_3d))
        inv_gamma_z = 1.0 / (torch.ones_like(sigma_z_3d) + (1j * sigma_z_3d))

        # derivative of sigma
        sigma_prime_outer = -2 * self.sigma_max / self.PMLsize * (1 - pml_coord / self.PMLsize)
        sigma_prime_1d = torch.zeros((self.domain_size,), dtype=self.float_dtype)
        sigma_prime_1d[:self.PMLsize] = sigma_prime_outer
        sigma_prime_1d[-self.PMLsize:] = -torch.flip(sigma_prime_outer, dims=[0]) # Sign flip for the other side

        # For 3D
        sigma_x_partial_3d, sigma_y_partial_3d, sigma_z_partial_3d = torch.meshgrid(sigma_prime_1d, sigma_prime_1d, sigma_prime_1d, indexing='xy')

        # derivative of Gamma
        gamma_x_partial = 1j * sigma_x_partial_3d
        gamma_y_partial = 1j * sigma_y_partial_3d
        gamma_z_partial = 1j * sigma_z_partial_3d

        # Coefficients for the modified Laplacian
        # L = ax dx' + bx dx'' + ay dy' + by dy'' + az dz' + bz dz''
        ax = -gamma_x_partial * (inv_gamma_x ** 3)
        bx = inv_gamma_x ** 2
        ay = -gamma_y_partial * (inv_gamma_y ** 3)
        by = inv_gamma_y ** 2
        az = -gamma_z_partial * (inv_gamma_z ** 3)
        bz = inv_gamma_z ** 2

        # Unsqueeze for batch dimension
        return (ax.unsqueeze(0).to(self.complex_dtype), bx.unsqueeze(0).to(self.complex_dtype),
                ay.unsqueeze(0).to(self.complex_dtype), by.unsqueeze(0).to(self.complex_dtype),
                az.unsqueeze(0).to(self.complex_dtype), bz.unsqueeze(0).to(self.complex_dtype))

    def forward(self, u: torch.Tensor):
        """
        Apply 3D laplacian with PML to u

        Args:
            u (complex or float tensor): (batch_size, domain_size, domain_size, domain_size)
        return:
            lappml_u (complex tensor)  : (batch_size, domain_size, domain_size, domain_size)
        """
        # FFT over the spatial dimensions (last 3 dimensions)
        u_fft = torch.fft.fftn(u, dim=(-3, -2, -1), norm="backward")

        dx_fft  = u_fft * self.kx_i
        dy_fft  = u_fft * self.ky_i
        dz_fft  = u_fft * self.kz_i
        ddx_fft = u_fft * self.kx_i_sq
        ddy_fft = u_fft * self.ky_i_sq
        ddz_fft = u_fft * self.kz_i_sq

        # Stack derivatives for efficient inverse FFT
        # Order: dx, dy, dz, ddx, ddy, ddz
        derivatives_fft_stack = torch.stack([dx_fft, dy_fft, dz_fft, ddx_fft, ddy_fft, ddz_fft], dim=0)

        # Inverse FFT over the spatial dimensions
        derivatives = torch.fft.ifftn(derivatives_fft_stack, dim=(-3, -2, -1), norm="backward")

        dx  = derivatives[0]
        dy  = derivatives[1]
        dz  = derivatives[2]
        ddx = derivatives[3]
        ddy = derivatives[4]
        ddz = derivatives[5]

        lappml_u = (self.ax * dx + self.ay * dy + self.az * dz +
                    self.bx * ddx + self.by * ddy + self.bz * ddz)
        
        return lappml_u


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    domain_size = 256
    PMLsize = 40
    k_wave_input_x = 5  # Wave number k1 for the input plane wave e^(i*(k1*x + k2*y))
    k_wave_input_y = 5   # Wave number k2 for the input plane wave
    
    sigma_max_example = 2
    dx = 2*torch.pi/domain_size 
    default_dtype = torch.complex128
    float_dtype_for_real_input = torch.float64 if default_dtype == torch.complex128 else torch.float32

    # Create coordinate grid
    coords_1d = torch.arange(domain_size, dtype=float_dtype_for_real_input) * dx
    x_2d, y_2d = torch.meshgrid(coords_1d, coords_1d, indexing='xy') 

    # Generate complex plane wave: u_c = e^(i * (k1*x + k2*y))
    complex_arg = 1j * (float(k_wave_input_x) * x_2d + float(k_wave_input_y) * y_2d)
    u_c_plane_wave = torch.exp(complex_arg.to(default_dtype)) 
    u_c = u_c_plane_wave.unsqueeze(0) # Add batch dim

    # Generate real part for real input: u_r = cos(k1*x + k2*y)
    real_arg = float(k_wave_input_x) * x_2d + float(k_wave_input_y) * y_2d
    u_r_plane_wave = torch.cos(real_arg.to(float_dtype_for_real_input)) 
    u_r = u_r_plane_wave.unsqueeze(0) # Add batch dim

    # Instantiate operators
    laplacianpml_op = LaplacianFFTwithPML2D(domain_size, PMLsize, dx, sigma_max_example, complex_dtype=default_dtype)
    laplacianreal_op = LaplacianRealFFT2D(domain_size, dx, float_dtype=float_dtype_for_real_input)
    laplaciancmpx_op = LaplacianComplexFFT2D(domain_size, dx, complex_dtype=default_dtype)

    # Apply operators
    lappml_u = laplacianpml_op(u_c)
    lapreal_u = laplacianreal_op(u_r)
    lapcmpx_u = laplaciancmpx_op(u_c)

    print("Shape of lapreal_u:", lapreal_u.shape)
    
    # Analytical Laplacian of e^(i(k1x+k2y)) is -(k1^2+k2^2) * e^(i(k1x+k2y))
    # For u_r = cos(k1x+k2y), Laplacian is -(k1^2+k2^2) * cos(k1x+k2y)
    analytical_factor = - (float(k_wave_input_x)**2 + float(k_wave_input_y)**2)
    
    analytical_lap_real = analytical_factor * u_r
    error_real_vs_analytical = torch.norm(lapreal_u[0].cpu() - analytical_lap_real[0].cpu()) / torch.norm(analytical_lap_real[0].cpu())
    print(f"Relative error for real Laplacian vs analytical (2D wave): {error_real_vs_analytical.item()}")
    
    analytical_lap_complex = analytical_factor * u_c
    error_complex_vs_analytical = torch.norm(lapcmpx_u[0].cpu() - analytical_lap_complex[0].cpu()) / torch.norm(analytical_lap_complex[0].cpu())
    print(f"Relative error for complex Laplacian vs analytical (2D wave): {error_complex_vs_analytical.item()}")

    # Visualization 
    plt.figure(figsize=(18, 6)) # Adjusted figure size
    

    plt.subplot(1, 3, 1)
    plt.imshow(lapcmpx_u[0].real.cpu().numpy())
    plt.title('Real part of LaplacianComplexFFT')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(lappml_u[0].real.cpu().numpy())
    plt.title('Real part of LaplacianPMLwithFFT')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(lapcmpx_u[0].real.cpu().numpy() - lapreal_u[0].cpu().numpy())
    plt.title('Diff: $Re(LapComplexFFT) - LapRealFFT$')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('test_LaplaceFFT.png')
    print("Plot saved to test_LaplaceFFT.png")

    print("\nComparing lapcmpx_u.real and lapreal_u (2D wave):")
    norm_lapreal_u = torch.norm(lapreal_u[0].cpu())
    norm_diff_cmpx_real = torch.norm(lapcmpx_u[0].real.cpu() - lapreal_u[0].cpu()) / norm_lapreal_u
    print(f"Relative norm of (lapcmpx_u.real - lapreal_u): {norm_diff_cmpx_real.item()}")
    
