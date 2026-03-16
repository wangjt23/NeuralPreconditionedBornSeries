import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class ConvectionDiffusionCBS(torch.nn.Module):
    """
    A CBS-style iterative solver for the 2D steady-state
    convection-diffusion-reaction equation in conservative form:
    −∇⋅(κ∇u) + v⋅∇u + σu = f
    
    This solver uses periodic boundary conditions and a preconditioned
    Richardson iteration: u_k+1 = u_k - G(A u_k - f).
    """
    def __init__(self,
                 N=[128, 128],
                 dx=0.01,
                 kappa0=1.0,
                 sigma0=0.0,
                 v0=[0.0, 0.0],
                 eta=1e-4,
                 fdtype=torch.float32,
                 device="cpu"):
        super().__init__()
        self.N = torch.tensor(N, dtype=torch.int)
        self.dx = dx
        self.device = device
        self.fdtype = fdtype

        # --- L₀ Operator Parameters ---
        self.kappa0 = kappa0
        self.sigma0 = sigma0
        self.v0 = torch.tensor(v0, device=device, dtype=fdtype)
        self.eta = eta # Damping for stability

        # --- Fourier Grid Setup ---
        freq_x = 2 * torch.pi * torch.fft.fftfreq(N[0], d=dx)
        freq_y = 2 * torch.pi * torch.fft.fftfreq(N[1], d=dx)
        self.xi_x = freq_x.view(-1, 1).to(device)
        self.xi_y = freq_y.view(1, -1).to(device)

        self.xi2 = self.xi_x**2 + self.xi_y**2

        # --- G = L₀⁻¹ in Fourier Space ---
        # L̂₀(ξ) = κ₀|ξ|² + i(v₀⋅ξ) + (σ₀+η)
        L0_hat = (self.kappa0 * self.xi2 +
                  1j * (self.v0[0] * self.xi_x + self.v0[1] * self.xi_y) +
                  (self.sigma0 + self.eta))
        
        # Add epsilon to avoid division by zero at zero-frequency if sigma0+eta is zero
        # L0_hat[L0_hat.abs() < 1e-9] = 1.0
        self.G_hat = (1.0 / L0_hat).to(torch.complex64 if fdtype==torch.float32 else torch.complex128)

    def _spectral_derivative(self, u, order, axis):
        """Computes derivative using FFT."""
        u_hat = torch.fft.fftn(u, dim=[-2, -1])
        xi = self.xi_x if axis == 0 else self.xi_y
        deriv_hat = ((1j * xi)**order) * u_hat
        return torch.fft.ifftn(deriv_hat, dim=[-2, -1]).real

    def _spectral_laplacian(self, u):
        """Computes Laplacian using FFT."""
        u_hat = torch.fft.fftn(u, dim=[-2, -1])
        lap_hat = (-self.xi2) * u_hat
        return torch.fft.ifftn(lap_hat, dim=[-2, -1]).real
    
    # New helper method to compute the gradient of a field
    def _spectral_gradient(self, field):
        """Computes the gradient (∇field) of a 2D field using FFT."""
        grad_x = self._spectral_derivative(field, order=1, axis=0)
        grad_y = self._spectral_derivative(field, order=1, axis=1)
        return grad_x, grad_y

    def A_op(self, u):
        """
        Applies the full operator A u in conservative form.
        A u = -κ(x)Δu - (∇κ)⋅(∇u) + v(x)⋅∇u + σ(x)u
        """
        # Compute derivatives of the solution u
        grad_u_x = self._spectral_derivative(u, order=1, axis=0)
        grad_u_y = self._spectral_derivative(u, order=1, axis=1)
        lap_u = self._spectral_laplacian(u)

        # Compute gradient of the coefficient kappa
        grad_kappa_x, grad_kappa_y = self._spectral_gradient(self.kappa)

        # Assemble the operator terms
        # Term 1: -κΔu
        term1 = -self.kappa * lap_u
        # Term 2: -(∇κ)⋅(∇u)
        term2 = -(grad_kappa_x * grad_u_x + grad_kappa_y * grad_u_y)
        # Term 3: v⋅∇u
        term3 = self.v[..., 0] * grad_u_x + self.v[..., 1] * grad_u_y
        # Term 4: σu
        term4 = self.sigma * u
        
        Au = term1 + term2 + term3 + term4
        return Au

    def G_op(self, u):
        """Applies the Green's function G = L₀⁻¹. Returns a complex tensor."""
        u_hat = torch.fft.fftn(u.to(self.G_hat.dtype), dim=[-2,-1])
        return torch.fft.ifftn(self.G_hat * u_hat, dim=[-2,-1])

    def setup(self, kappa, v, sigma, f):
        """Sets up the problem with specific coefficients."""
        self.kappa = kappa.to(self.device, self.fdtype)
        self.v = v.to(self.device, self.fdtype)
        self.sigma = sigma.to(self.device, self.fdtype)
        self.f = f.to(self.device, self.fdtype)

    def forward(self, u, relaxation=1.0):
        """Performs one step of preconditioned Richardson: u_k+1 = u_k - w*G(A u_k - f)."""
        residual = self.A_op(u) - self.f
        residual = self.G_op(residual)
        
        # The update is complex, but the true solution u should be real.
        # We take the real part of the update.
        u_next = u - relaxation * residual.real
        return u_next

if __name__ == '__main__':
    from dataloader import ConvectionDiffusionDataset
    # Set default dtype to float64 for higher precision
    torch.set_default_dtype(torch.float64)
    fdtype = torch.float64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- Problem Setup ---
    N = [128, 128]
    dx = 1.0 / N[0]
    
    grid_x = torch.linspace(0, 1, N[0], device=device)
    grid_y = torch.linspace(0, 1, N[1], device=device)
    X, Y = torch.meshgrid(grid_x, grid_y, indexing='ij')

    # kappa = 1.0 + 0.5 * torch.sin(2 * torch.pi * X) * torch.cos(2 * torch.pi * Y)
    # v = torch.stack([ torch.cos(2 * torch.pi * Y) * 1.0, torch.sin(2 * torch.pi * Y) * 10.0], dim=-1)
    # sigma = 1.0 * (torch.exp(-((X-0.25)**2 + (Y-0.25)**2) / 0.05))*0+0.5
    # f = 1.0 * torch.exp(-((X-0.75)**2 + (Y-0.75)**2) / 0.01)
    # data = torch.load('./data/test/sample_0000.pt')
    TEST_PATH = 'data/test'
    test_dataset = ConvectionDiffusionDataset(TEST_PATH, data_num=200, device=device)
    num_samples = min(len(test_dataset), 1)
    num_iters = 50
    tol = 1e-6
    relaxation = 1.0
    iter_counts = []
    iter_his = None

    print("Starting iterative solve...")
    for sample_idx in range(num_samples):
        data = test_dataset[sample_idx].unsqueeze(0).to(device)
        kappa = data[:, 0:1, ...]
        v = data[:, 1:3, ...].permute(0, 2, 3, 1)
        f = data[:, 3:4, ...]
        sigma = torch.ones_like(f)

        solver = ConvectionDiffusionCBS(
            N=N, dx=dx,
            kappa0=kappa.mean().item(),
            v0=[v[..., 0].mean().item(), v[..., 1].mean().item()],
            sigma0=sigma.mean().item(),
            eta=1.0,
            fdtype=fdtype,
            device=device,
        ).to(device)

        solver.setup(kappa, v, sigma, f)

        u = torch.zeros_like(f)
        iter_his = [1.0]
        for i in range(num_iters):
            u_prev = u
            u = solver(u, relaxation=relaxation)

            residual = solver.f - solver.A_op(u)
            rel_residual_norm = torch.linalg.norm(residual) / torch.linalg.norm(solver.f)
            iter_his.append(rel_residual_norm.item())
            change = torch.linalg.norm(u - u_prev) / torch.linalg.norm(u)
            print(f"Sample {sample_idx:3d} | Iter {i+1:4d}, Rel Change: {change:.4e}, Rel Residual: {rel_residual_norm:.4e}")

            if rel_residual_norm < tol:
                break
        print("-"*10)
        iter_counts.append(i + 1)
    
    avg_iters = float(np.mean(iter_counts)) if iter_counts else 0.0
    print(f"Average iterations over {num_samples} samples: {avg_iters:.2f}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Convection-Diffusion CBS-style Solver Results', fontsize=16)

    # Plot coefficients
    im = axes[0, 0].imshow(kappa.squeeze().cpu().numpy(), origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[0, 0].set_title("κ(x) - Diffusion")
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)

    x_1d, y_1d = grid_x.cpu().numpy(), grid_y.cpu().numpy()
    u_np, v_np = v[..., 0].cpu().numpy().squeeze(), v[..., 1].cpu().numpy().squeeze()
    axes[0, 1].streamplot(y_1d, x_1d, v_np.T, u_np.T, color='black', density=1.5)
    axes[0, 1].set_title("v(x) - Velocity Field")
    axes[0, 1].set_xlabel("Y-axis"); axes[0, 1].set_ylabel("X-axis")
    axes[0, 1].set_aspect('equal'); axes[0, 1].set_xlim(0, 1); axes[0, 1].set_ylim(0, 1)

    im = axes[0, 2].imshow(sigma.squeeze().cpu().numpy(), origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[0, 2].set_title("σ(x) - Reaction/Absorption")
    plt.colorbar(im, ax=axes[0, 2], shrink=0.8)
    
    # Plot results
    im = axes[1, 0].imshow(f.squeeze().cpu().numpy(), origin='lower', extent=[0,1,0,1], cmap='bwr')
    axes[1, 0].set_title("f(x) - Source Term")
    plt.colorbar(im, ax=axes[1, 0], shrink=0.8)

    im = axes[1, 1].imshow(u.squeeze().cpu().numpy(), origin='lower', extent=[0,1,0,1], cmap='jet')
    axes[1, 1].set_title(f"Solution u(x) (Iter {i+1})")
    plt.colorbar(im, ax=axes[1, 1], shrink=0.8)

    final_residual = (solver.f - solver.A_op(u))/torch.norm(solver.f)

    im = axes[1, 2].imshow(final_residual.squeeze().cpu().numpy(), origin='lower', extent=[0,1,0,1], cmap='bwr')
    axes[1, 2].set_title("(f - Au)")
    plt.colorbar(im, ax=axes[1, 2], shrink=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('conv_diff_cbs_improved.png', dpi=150)
    print("\nSaved plot to conv_diff_cbs_improved.png")
    torch.save(iter_his,'iter_his_cbs.pt')