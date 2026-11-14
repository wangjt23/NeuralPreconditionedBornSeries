import torch
import torch.nn as nn


class PseudoSpectralRTE2d(nn.Module):
    """
    Experimental periodic pseudo-spectral DOM RTE operator.

    State shape is [batch, 4*M, I, J]. This is deliberately separate from
    the ATFPS inflow-basis operator, whose state shape is [batch, 8*M, I, J].
    """

    def __init__(self, mesh_size=16, domain=(0.0, 1.0, 0.0, 1.0), fdtype=torch.float32, device="cpu"):
        super().__init__()
        self.mesh_size = int(mesh_size)
        self.domain = domain
        self.float_dtype = fdtype
        self.complex_dtype = torch.complex128 if fdtype == torch.float64 else torch.complex64
        self.device = torch.device(device)
        self._is_setup = False

    def setup(self, coeff, ct, st, omega, Kappa):
        if coeff.ndim == 3:
            coeff = coeff.unsqueeze(0)
        coeff = coeff.to(self.device, dtype=self.float_dtype)
        self.coeff = coeff
        self.batch_size, _, self.I, self.J = coeff.shape
        self.mesh_size = self.I

        self.ct = ct.reshape(-1).to(self.device, dtype=self.float_dtype)
        self.st = st.reshape(-1).to(self.device, dtype=self.float_dtype)
        self.omega = omega.reshape(-1).to(self.device, dtype=self.float_dtype)
        self.Kappa = Kappa.to(self.device, dtype=self.float_dtype)
        self.KW = self.Kappa @ torch.diag(self.omega)
        self.num_angles = self.ct.numel()

        self.sigma_t = coeff[:, 0]
        self.sigma_a = coeff[:, 1]
        self.epsilon = torch.exp(coeff[:, 2])
        self._setup_reference()
        self._is_setup = True
        return self

    def _wave_numbers(self):
        xl, xr, yl, yr = self.domain
        hx = (xr - xl) / self.I
        hy = (yr - yl) / self.J
        kx = 2.0 * torch.pi * torch.fft.fftfreq(self.I, d=hx, device=self.device)
        ky = 2.0 * torch.pi * torch.fft.fftfreq(self.J, d=hy, device=self.device)
        return kx.to(self.float_dtype).view(self.I, 1), ky.to(self.float_dtype).view(1, self.J)

    def _setup_reference(self):
        sigma_t0 = self.sigma_t.mean(dim=(-2, -1))
        sigma_a0 = self.sigma_a.mean(dim=(-2, -1))
        epsilon0 = self.epsilon.mean(dim=(-2, -1))
        self.ref_absorb = sigma_t0 / epsilon0
        self.ref_scatter = sigma_t0 / epsilon0 - epsilon0 * sigma_a0

        eye = torch.eye(self.num_angles, device=self.device, dtype=self.float_dtype)
        collision0 = self.ref_absorb[:, None, None] * eye - self.ref_scatter[:, None, None] * self.KW
        self.collision0 = collision0.to(self.complex_dtype)

        kx, ky = self._wave_numbers()
        kx_grid = kx.expand(self.I, self.J)
        ky_grid = ky.expand(self.I, self.J)
        phase = self.ct.view(1, 1, 1, -1) * kx_grid.view(1, self.I, self.J, 1)
        phase = phase + self.st.view(1, 1, 1, -1) * ky_grid.view(1, self.I, self.J, 1)
        phase = phase.expand(self.batch_size, -1, -1, -1)
        transport = torch.diag_embed(1j * phase.to(self.complex_dtype))
        self.reference_symbol = transport + self.collision0[:, None, None, :, :]

    def _spectral_streaming(self, u):
        kx, ky = self._wave_numbers()
        u_hat = torch.fft.fftn(u, dim=(-2, -1))
        dx = torch.fft.ifftn(1j * kx.view(1, 1, self.I, 1) * u_hat, dim=(-2, -1))
        dy = torch.fft.ifftn(1j * ky.view(1, 1, 1, self.J) * u_hat, dim=(-2, -1))
        return self.ct.view(1, -1, 1, 1) * dx + self.st.view(1, -1, 1, 1) * dy

    def _variable_collision(self, u):
        scatter = torch.einsum("mn,bnij->bmij", self.KW.to(u.dtype), u)
        absorb = (self.sigma_t / self.epsilon).to(u.dtype).unsqueeze(1)
        scatter_weight = (self.sigma_t / self.epsilon - self.epsilon * self.sigma_a).to(u.dtype).unsqueeze(1)
        return absorb * u - scatter_weight * scatter

    def _reference_collision(self, u):
        scatter = torch.einsum("mn,bnij->bmij", self.KW.to(u.dtype), u)
        absorb = self.ref_absorb.to(u.dtype).view(-1, 1, 1, 1)
        scatter_weight = self.ref_scatter.to(u.dtype).view(-1, 1, 1, 1)
        return absorb * u - scatter_weight * scatter

    def _return_dtype(self, out, template):
        if torch.is_complex(template):
            return out
        return out.real.to(template.dtype)

    def A_op(self, u):
        if not self._is_setup:
            raise RuntimeError("PseudoSpectralRTE2d.setup must be called before A_op.")
        u_complex = u.to(self.device).to(self.complex_dtype)
        out = self._spectral_streaming(u_complex) + self._variable_collision(u_complex)
        return self._return_dtype(out, u)

    def G_inv(self, u):
        if not self._is_setup:
            raise RuntimeError("PseudoSpectralRTE2d.setup must be called before G_inv.")
        u_complex = u.to(self.device).to(self.complex_dtype)
        out = self._spectral_streaming(u_complex) + self._reference_collision(u_complex)
        return self._return_dtype(out, u)

    def G(self, rhs):
        if not self._is_setup:
            raise RuntimeError("PseudoSpectralRTE2d.setup must be called before G.")
        rhs_complex = rhs.to(self.device).to(self.complex_dtype)
        rhs_hat = torch.fft.fftn(rhs_complex, dim=(-2, -1)).permute(0, 2, 3, 1)
        sol_hat = torch.linalg.solve(self.reference_symbol, rhs_hat.unsqueeze(-1)).squeeze(-1)
        sol = torch.fft.ifftn(sol_hat.permute(0, 3, 1, 2), dim=(-2, -1))
        return sol

    def born_residual(self, u, rhs):
        return self.G(rhs - self.A_op(u))

    def cbs_preconditioner(self, rhs, mode="green"):
        if mode != "green":
            raise ValueError("Only the matrix-valued Green preconditioner is implemented for RTE.")
        return self.G(rhs)
