from ._precision import preserve_default_dtype
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvergentBornSeries2d(nn.Module):
    @preserve_default_dtype
    def __init__(self,
                 freq=1,
                 N = [96,96],
                 dx=0.125,
                 boundary_width=[8,8],
                 boundary_type='pbl',
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
        # self.lamb = 1/self.freq # wavelength, the units depend on the length units of the spacial field
        self.omega0 = 2*torch.pi * self.freq
        self.dx = dx
        self.N = torch.tensor(N).to(int)
        
        ## set grid
        self.boundary_widths = torch.tensor(boundary_width,dtype=int)
        self.N_pad = self.N+self.boundary_widths*2
        # spatial coordinates 
        self.px_range = 2*torch.pi*torch.fft.fftfreq(self.N_pad[0],d=self.dx).view(-1,1)
        self.py_range = (2*torch.pi*torch.fft.fftfreq(self.N_pad[1],d=self.dx))
        self.p_dist2 = (self.px_range**2+self.py_range**2).to(self.device)

        ## boundary setting
        self.boundary_type = boundary_type

        self.Bl = torch.ceil((self.N_pad - self.N)/2).to(int)
        self.Br = torch.floor((self.N_pad - self.N)/2).to(int)
        self.roi = [[self.Bl[0],self.Br[0]+self.N[0]],
                    [self.Bl[1],self.Br[1]+self.N[1]]]
        
        x = torch.cat((torch.arange(self.Bl[0], 0, -1), torch.zeros(self.N[0]), torch.arange(1, self.Br[0] + 1))).view(-1,1)
        y = torch.cat((torch.arange(self.Bl[1], 0, -1), torch.zeros(self.N[1]), torch.arange(1, self.Br[1] + 1)))
        self.boundary_dist = torch.sqrt(x**2 + y**2).to(self.device)

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
        # u = u - (self.gamma32*(u - self.G(self.V*u+self.b/self.dx**2, self.g0_tilde)).to(torch.complex64)).to(torch.complex128)
        u = u - self.gamma*(u - self.G(self.V*u+self.b/self.dx**2, self.g0_tilde))
        return u
    
    def setup(self, sos, src=None):
        self.sos = sos
        self.setup_boundary()
        
        ## CBS params
        self.k0 = ((1/torch.mean(self.sos,dim=[-2,-1]))*self.omega0).unsqueeze(-1).unsqueeze(-1)# default
        # self.k0  = self.omega0/self.sos_mean # perform bad
        self.epsilon = torch.amax(torch.abs(self.inv_sos2*self.omega0**2 - self.k0**2), dim=[-1,-2]).unsqueeze(-1).unsqueeze(-1)
        self.epsilon = 1.4*self.epsilon
        # scatter potential
        self.V = self.inv_sos2*self.omega0**2 - self.k0**2 - 1.0j*self.epsilon
        # g0_tilde
        self.g0_tilde = 1/(self.p_dist2-self.k0**2 - 1.0j*self.epsilon)
        # b
        if src != None:
            self.b = F.pad(src, 
                            ((self.N_pad[1]-src.size(-1))//2, 
                            (self.N_pad[1]-src.size(-1))//2,
                            (self.N_pad[0]-src.size(-2))//2, 
                            (self.N_pad[0]-src.size(-2))//2), 
                            mode='constant', value=0)
        # precond
        self.gamma = 1.j/self.epsilon*self.V
        # ## test mix precision
        # self.gamma32 = self.gamma.to(torch.complex64)
        # self.g0_tilde32 = self.g0_tilde.to(torch.complex64)

    def G(self, u, g0_tilde):
        out = torch.fft.ifftn(g0_tilde * torch.fft.fftn(u,dim=[-1,-2],norm='backward'), dim=[-1,-2],norm='backward')
        return out

    def G_inv(self, u, g0_tilde):
        out = torch.fft.ifftn(1/g0_tilde * torch.fft.fftn(u,dim=[-1,-2],norm='backward'), dim=[-1,-2],norm='backward')
        return  out
    
    def A_op(self, u):
        Au = (self.G_inv(u, self.g0_tilde) - self.V*u)*(self.dx**2)
        return Au

    def cal_residual(self, u, rhs):
        return rhs - self.A_op(u)
    
    def setup_boundary(self): 
        # add padding
        self.sos = F.pad(self.sos, 
                        (self.Bl[1], self.Br[1], self.Bl[0], self.Br[0]), 
                        mode='replicate')
        self.inv_sos2 = 1/self.sos**2
        
        factor = torch.abs(torch.mean(self.inv_sos2,dim=[-1,-2])).unsqueeze(-1).unsqueeze(-1)


        if self.boundary_type == "abc":
            # abc
            absorb = self.set_abc_layer(self.inv_sos2.size(-1),torch.max(self.Br),0.5)

            self.inv_sos2 = self.inv_sos2 + self.inv_sos2*absorb*1.j
            
        elif self.boundary_type == "pbl":
            # pbl
            Bmax = torch.max(self.Br)
            # debug: why divide self.dx ?

            # k0 = (1/torch.mean(self.sos, dim=[-1,-2]))*2*torch.pi/(self.lamb/self.dx)# k0 in 1/pixels
            k0 = (1/torch.mean(self.sos, dim=[-1,-2]))*2*torch.pi*self.freq*self.dx # k0 in 1/pixels
            # k0 = (1/1500.)*2*torch.pi/(self.lamb/self.dx)
            # k0 = torch.sqrt(torch.mean(self.inv_sos2, dim=[-1,-2]))*2*torch.pi/(self.lamb)# wjt's 
            k0 = k0.unsqueeze(-1).unsqueeze(-1)
            # the shape of the boundary is determined by f_boundary_curve, a function
            # that takes a position (in pixels, 0=start of boundary) and returns
            # Delta inv_sos2 for the boundary. 

            absorb = self.set_pbl_layer(r=self.boundary_dist,Bmax=Bmax,c=k0/2,k0=k0)
            self.inv_sos2 = self.inv_sos2 + factor*absorb.to(self.device)
    def set_pbl_layer(self, r, Bmax, c, k0):
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
    


class ConvergentBornSeries3d(nn.Module):
    @preserve_default_dtype
    def __init__(self,
                 freq=1,
                 N = [32,32,32], # Changed to 3D default
                 dx=0.125,
                 boundary_width=[4,4,4], # Changed to 3D default
                 boundary_type='pbl', # Default that doesn't strictly need LapPML for V
                 precond = None,
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
        # self.lamb = 1/freq 
        self.omega0 = 2*torch.pi*freq
        self.dx = dx # Assuming isotropic

        self.N = torch.tensor(N, dtype=torch.int)
        
        ## set grid
        self.boundary_widths = torch.tensor(boundary_width,dtype=torch.int)
        self.N_pad = self.N + self.boundary_widths*2
        
        # spatial coordinates (N_pad is [Nz, Ny, Nx] or [Nd, Nh, Nw])
        self.pz_range = 2*torch.pi*torch.fft.fftfreq(self.N_pad[0],d=self.dx).view(-1,1,1)
        self.py_range = 2*torch.pi*torch.fft.fftfreq(self.N_pad[1],d=self.dx).view(1,-1,1)
        self.px_range = 2*torch.pi*torch.fft.fftfreq(self.N_pad[2],d=self.dx).view(1,1,-1)
        self.p_dist2 = (self.pz_range**2 + self.py_range**2 + self.px_range**2).to(self.device)

        ## boundary setting
        self.boundary_type = boundary_type

        self.Bl = torch.ceil((self.N_pad - self.N)/2).to(torch.int)
        self.Br = torch.floor((self.N_pad - self.N)/2).to(torch.int)
        self.roi = [[self.Bl[0],self.Br[0]+self.N[0]],
                    [self.Bl[1],self.Br[1]+self.N[1]],
                    [self.Bl[2],self.Br[2]+self.N[2]]] # Added 3rd dimension
        
        z_coords = torch.cat((torch.arange(self.Bl[0], 0, -1), torch.zeros(self.N[0]), torch.arange(1, self.Br[0] + 1))).view(-1,1,1)
        y_coords = torch.cat((torch.arange(self.Bl[1], 0, -1), torch.zeros(self.N[1]), torch.arange(1, self.Br[1] + 1))).view(1,-1,1)
        x_coords = torch.cat((torch.arange(self.Bl[2], 0, -1), torch.zeros(self.N[2]), torch.arange(1, self.Br[2] + 1))).view(1,1,-1)
        self.boundary_dist = torch.sqrt(z_coords**2 + y_coords**2 + x_coords**2).to(self.device)        


    def forward(self, u):
        # residual = self.b - self.A_op(u)

        # u = u + (self.gamma * self.G(u=residual/(self.dx**2), g0_tilde=self.g0_tilde)).to(self.complex_dtype)
        # fast cbs
        u = u - self.gamma*(u-self.G(self.V*u+self.b/self.dx**2, self.g0_tilde))

        return u
    
    def setup(self, sos, src=None):
        self.sos = sos # Expected shape (batch, 1, Nz, Ny, Nx) or (batch, 1, D, H, W)
        self.setup_boundary() # This will pad self.sos
        
        self.k0 = ((1/torch.mean(self.sos,dim=[-3,-2,-1]))*self.omega0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.epsilon = torch.amax(torch.abs(self.inv_sos2*self.omega0**2 - self.k0**2), dim=[-3,-2,-1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        self.V = self.inv_sos2*self.omega0**2 - self.k0**2 - 1.0j*self.epsilon

        self.g0_tilde = 1/(self.p_dist2 - self.k0**2 - 1.0j*self.epsilon)
        
        if src is not None:
            # Pad src to N_pad dimensions (B, C, D, H, W)
            # Padding order: (pad_W_left, pad_W_right, pad_H_left, pad_H_right, pad_D_left, pad_D_right)
            pad_d = (self.N_pad[0]-src.size(-3))//2
            pad_h = (self.N_pad[1]-src.size(-2))//2
            pad_w = (self.N_pad[2]-src.size(-1))//2
            self.b = F.pad(src, 
                            (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), 
                            mode='constant', value=0)
        else:
            self.b = torch.zeros((sos.shape[0], 1, *self.N_pad), dtype=self.complex_dtype, device=self.device)

        # precond setup
        self.gamma = 1.j/self.epsilon*self.V

    def G(self, u, g0_tilde):
        out = torch.fft.ifftn(g0_tilde * torch.fft.fftn(u,dim=[-3,-2,-1],norm='backward'), dim=[-3,-2,-1],norm='backward')
        return out

    def G_inv(self, u, g0_tilde):
        out = torch.fft.ifftn(1/g0_tilde * torch.fft.fftn(u,dim=[-3,-2,-1],norm='backward'), dim=[-3,-2,-1],norm='backward')
        return out
    
    def A_op(self, u):
        Au = (self.G_inv(u, self.g0_tilde) - self.V*u)*(self.dx**2)
        return Au

    def cal_residual(self, u, rhs):
        return rhs - self.A_op(u)
    
    def setup_boundary(self): 
        # Pad sos for 3D: (pad_W_left, pad_W_right, pad_H_left, pad_H_right, pad_D_left, pad_D_right)
        self.sos = F.pad(self.sos, 
                        (self.Bl[2], self.Br[2], self.Bl[1], self.Br[1], self.Bl[0], self.Br[0]), 
                        mode='replicate')
        self.inv_sos2 = 1/self.sos**2
        factor = torch.abs(torch.mean(self.inv_sos2,dim=[-3,-2,-1])).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        if self.boundary_type == "abc":
            # abc for 3D: domain_dims N_pad = [Nz,Ny,Nx], widths = [Bz,By,Bx]
            absorb = self.set_abc_layer(self.N_pad, self.boundary_widths, 1) 
            self.inv_sos2 = self.inv_sos2 + factor*absorb*1.j
        else: # pbl
            Bmax = torch.max(self.Br).float()
            # k0_val = (1/torch.mean(self.sos, dim=[-3,-2,-1]))*2*torch.pi/(self.lamb/self.dx)
            k0_val = (1/torch.mean(self.sos, dim=[-3,-2,-1]))*2*torch.pi * self.freq * self.dx
            k0 = k0_val.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            absorb = self.set_pbl_layer(r=self.boundary_dist,Bmax=Bmax,c=k0/2,k0=k0)
            self.inv_sos2 = self.inv_sos2 + factor*absorb

    def set_pbl_layer(self, r, Bmax, c, k0):
        absorb = 1 / k0**2 * (c**5 * r**3 * (4.0 + (2.0j * k0 - c) * r)) / \
            (24 + 24 * c * r + 12 * c**2 * r**2 + 4 * c**3 * r**3 + c**4 * r**4)

        return absorb

    def set_abc_layer(self, domain_dims, widths, value):
        # domain_dims: [Nz, Ny, Nx]
        # widths: [width_z, width_y, width_x]
        layers = []
        for i in range(3): # Iterate over Z, Y, X dimensions
            if widths[i] > 0:
                abc_coord = torch.arange(widths[i], device=self.device, dtype=self.float_dtype)
                layer_outer_profile = value * (1 - abc_coord / widths[i]) ** 2 # Quadratic profile
                
                current_dim_layer_1d = torch.zeros((domain_dims[i],), device=self.device, dtype=self.float_dtype)
                current_dim_layer_1d[:widths[i]] = torch.flip(layer_outer_profile, dims=[0]) # Corrected flip
                current_dim_layer_1d[-widths[i]:] = layer_outer_profile
                
                # Reshape for broadcasting
                shape = [1,1,1]
                shape[i] = domain_dims[i]
                layers.append(current_dim_layer_1d.view(*shape))
            else: # If width is zero for a dimension, add a zero tensor of compatible shape for broadcasting
                shape = [1,1,1]
                shape[i] = domain_dims[i]
                layers.append(torch.zeros(shape, device=self.device, dtype=self.float_dtype))

        # Combine layers by addition (broadcasting will handle shapes)
        # E.g. layer_z (Nz,1,1) + layer_y (1,Ny,1) + layer_x (1,1,Nx)
        combined_layer = layers[0] + layers[1] + layers[2]
        return combined_layer
