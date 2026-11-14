import math

import torch
import torch.nn as nn


def _as_4d(value, name):
    if value.ndim == 2:
        return value.unsqueeze(0).unsqueeze(0)
    if value.ndim == 3:
        return value.unsqueeze(1)
    if value.ndim == 4:
        return value
    raise ValueError(f"{name} must have shape (H,W), (B,H,W), or (B,1,H,W)")


class PeriodicRTE2D(nn.Module):
    """
    Periodic 2-D discrete-ordinates radiative transfer operator with an
    FFT-applied constant-coefficient transport reference inverse.

    The steady-state isotropic-scattering system is

        (Omega_l . grad + sigma_t) psi_l - sigma_s * phi = q_l,
        phi = sum_m w_m psi_m.

    The reference operator is

        L_eta,l = Omega_l . grad + sigma_ref + eta,

    so G_eta = L_eta^{-1} is diagonal in spatial Fourier variables for periodic
    boundaries. This gives the same NPBS/Born identity as Helmholtz:

        I - G_eta V_eta = G_eta A,  V_eta = L_eta - A.

    Tensor convention:
        psi, source: (batch, n_angles, height, width), complex.
        sigma_t, sigma_s: (batch, 1, height, width), real.
    """

    def __init__(
        self,
        shape=(64, 64),
        dx=1.0,
        n_angles=16,
        anisotropy_g=0.0,
        sigma_ref=None,
        eta_scale=0.2,
        eta_floor=1e-6,
        diffusion_floor=1e-8,
        fdtype=torch.float32,
        device="cpu",
    ):
        super().__init__()
        self.shape = tuple(int(v) for v in shape)
        self.dx = float(dx)
        self.n_angles = int(n_angles)
        self.anisotropy_g = float(anisotropy_g)
        self.sigma_ref_value = sigma_ref
        self.eta_scale = float(eta_scale)
        self.eta_floor = float(eta_floor)
        self.diffusion_floor = float(diffusion_floor)
        self.float_dtype = fdtype
        self.complex_dtype = torch.complex128 if fdtype == torch.float64 else torch.complex64
        self.device_name = device

        angles = torch.arange(self.n_angles, dtype=fdtype) * (2.0 * math.pi / self.n_angles)
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        weights = torch.full((self.n_angles,), 1.0 / self.n_angles, dtype=fdtype)
        self.register_buffer("directions", directions, persistent=False)
        self.register_buffer("weights", weights, persistent=False)
        phase_matrix = self._make_phase_matrix(self.n_angles, self.anisotropy_g, fdtype)
        self.register_buffer("phase_matrix", phase_matrix, persistent=False)

        kx, ky = self._make_wavenumbers(self.shape, self.dx, fdtype)
        self.register_buffer("kx", kx, persistent=False)
        self.register_buffer("ky", ky, persistent=False)

        self.to(device)

    @staticmethod
    def _make_wavenumbers(shape, dx, dtype):
        height, width = shape
        kx = 2.0 * math.pi * torch.fft.fftfreq(height, d=dx).to(dtype).view(1, 1, height, 1)
        ky = 2.0 * math.pi * torch.fft.fftfreq(width, d=dx).to(dtype).view(1, 1, 1, width)
        return kx, ky

    @staticmethod
    def _make_phase_matrix(n_angles, anisotropy_g, dtype):
        angles = torch.arange(n_angles, dtype=dtype) * (2.0 * math.pi / n_angles)
        delta = angles.view(n_angles, 1) - angles.view(1, n_angles)
        g = float(anisotropy_g)
        if abs(g) < 1e-14:
            phase = torch.ones(n_angles, n_angles, dtype=dtype)
        else:
            if abs(g) >= 1.0:
                raise ValueError("anisotropy_g must satisfy |g| < 1")
            phase = (1.0 - g * g) / (1.0 - 2.0 * g * torch.cos(delta) + g * g)
            weights = torch.full((n_angles,), 1.0 / n_angles, dtype=dtype)
            phase = phase / torch.clamp(phase @ weights, min=1e-30).view(n_angles, 1)
        return phase

    def setup(self, sigma_t, sigma_s=None, source=None):
        sigma_t = _as_4d(sigma_t, "sigma_t").to(device=self.directions.device, dtype=self.float_dtype)
        if sigma_t.shape[-2:] != self.shape:
            raise ValueError(f"sigma_t spatial shape {sigma_t.shape[-2:]} does not match {self.shape}")
        if sigma_s is None:
            sigma_s = torch.zeros_like(sigma_t)
        else:
            sigma_s = _as_4d(sigma_s, "sigma_s").to(device=sigma_t.device, dtype=self.float_dtype)
        if sigma_s.shape != sigma_t.shape:
            raise ValueError("sigma_s must have the same shape as sigma_t")
        if torch.any(sigma_t <= 0):
            raise ValueError("sigma_t must be strictly positive")
        if torch.any(sigma_s < 0):
            raise ValueError("sigma_s must be nonnegative")
        if torch.any(sigma_s >= sigma_t):
            raise ValueError("this verifier assumes subcritical sigma_s < sigma_t")

        if self.sigma_ref_value is None:
            sigma_ref = torch.mean(sigma_t, dim=(-2, -1), keepdim=True)
        else:
            sigma_ref = torch.as_tensor(
                self.sigma_ref_value,
                dtype=self.float_dtype,
                device=sigma_t.device,
            ).reshape(1, 1, 1, 1)
            sigma_ref = sigma_ref.expand(sigma_t.shape[0], 1, 1, 1)

        heterogeneity = torch.amax(
            torch.abs(sigma_t - sigma_ref) + torch.abs(sigma_s),
            dim=(-2, -1),
            keepdim=True,
        )
        eta = self.eta_floor + self.eta_scale * heterogeneity
        sigma_s_ref = torch.mean(sigma_s, dim=(-2, -1), keepdim=True)
        sigma_a_ref = torch.clamp(sigma_ref - sigma_s_ref, min=self.diffusion_floor)
        diffusion_coeff = 1.0 / (2.0 * torch.clamp(sigma_ref, min=self.diffusion_floor))

        self.sigma_t = sigma_t
        self.sigma_s = sigma_s
        self.sigma_ref = sigma_ref
        self.sigma_s_ref = sigma_s_ref
        self.sigma_a_ref = sigma_a_ref
        self.diffusion_coeff = diffusion_coeff
        self.eta = eta
        self.sigma0 = sigma_ref + eta
        # Pointer/version pairs can repeat when an allocator recycles a freed
        # coefficient tensor. Consumers must distinguish successive setups.
        self.setup_generation = getattr(self, "setup_generation", 0) + 1
        # DSA and Born only need scalar Fourier divisions. A full angular
        # inverse at every coefficient batch wastes O(B H W n_angles^3) work.
        self._reference_transport_inverse = None
        self._reference_moment_inverse_cache = {}

        if source is None:
            self.source = torch.zeros(
                sigma_t.shape[0],
                self.n_angles,
                *self.shape,
                dtype=self.complex_dtype,
                device=sigma_t.device,
            )
        else:
            if source.ndim == 3:
                source = source.unsqueeze(0)
            if source.ndim != 4 or source.shape[1] != self.n_angles:
                raise ValueError("source must have shape (B,n_angles,H,W) or (n_angles,H,W)")
            if source.shape[-2:] != self.shape:
                raise ValueError(f"source spatial shape {source.shape[-2:]} does not match {self.shape}")
            if source.shape[0] == 1 and sigma_t.shape[0] > 1:
                source = source.expand(sigma_t.shape[0], -1, -1, -1)
            elif source.shape[0] != sigma_t.shape[0]:
                raise ValueError("source batch size must be 1 or match sigma_t batch size")
            self.source = source.to(device=sigma_t.device, dtype=self.complex_dtype)
        return self

    def scalar_flux(self, psi):
        weights = self.weights.to(device=psi.device, dtype=psi.real.dtype).view(1, self.n_angles, 1, 1)
        return torch.sum(weights * psi, dim=1, keepdim=True)

    def scattering_op(self, psi):
        weights = self.weights.to(device=psi.device, dtype=psi.real.dtype)
        phase = self.phase_matrix.to(device=psi.device, dtype=psi.dtype)
        weighted_psi = weights.view(1, self.n_angles, 1, 1) * psi
        scattered = torch.einsum("lm,bmhw->blhw", phase, weighted_psi)
        return self.sigma_s.to(dtype=psi.dtype, device=psi.device) * scattered

    def _stream_symbol(self, batch_size, dtype, device):
        directions = self.directions.to(device=device, dtype=self.float_dtype)
        omega_x = directions[:, 0].view(1, self.n_angles, 1, 1)
        omega_y = directions[:, 1].view(1, self.n_angles, 1, 1)
        symbol = omega_x * self.kx.to(device=device) + omega_y * self.ky.to(device=device)
        symbol = 1j * symbol.to(dtype=dtype)
        return symbol.expand(batch_size, self.n_angles, *self.shape)

    def streaming_op(self, psi):
        psi_hat = torch.fft.fftn(psi, dim=(-2, -1), norm="backward")
        symbol = self._stream_symbol(psi.shape[0], psi.dtype, psi.device)
        return torch.fft.ifftn(symbol * psi_hat, dim=(-2, -1), norm="backward")

    def A_op(self, psi):
        sigma_t = self.sigma_t.to(dtype=psi.dtype, device=psi.device)
        return self.streaming_op(psi) + sigma_t * psi - self.scattering_op(psi)

    def L_eta_op(self, psi):
        sigma0 = self.sigma0.to(dtype=psi.dtype, device=psi.device)
        return self.streaming_op(psi) + sigma0 * psi

    def V_op(self, psi):
        sigma0 = self.sigma0.to(dtype=psi.dtype, device=psi.device)
        sigma_t = self.sigma_t.to(dtype=psi.dtype, device=psi.device)
        return (sigma0 - sigma_t) * psi + self.scattering_op(psi)

    def G(self, rhs):
        rhs_hat = torch.fft.fftn(rhs, dim=(-2, -1), norm="backward")
        stream = self._stream_symbol(rhs.shape[0], rhs.dtype, rhs.device)
        sigma0 = self.sigma0.to(dtype=rhs.dtype, device=rhs.device)
        denom = stream + sigma0
        out_hat = rhs_hat / denom
        return torch.fft.ifftn(out_hat, dim=(-2, -1), norm="backward")

    def diffusion_solve(self, scalar_rhs):
        """
        Solve a constant-coefficient periodic diffusion equation for the DSA
        scalar-flux correction:

            (-D_ref Delta + sigma_a_ref) delta_phi = scalar_rhs.

        This is a deliberately simple synthetic-acceleration model. It captures
        the low-order scattering error mode but is not a replacement for a
        boundary-compatible production DSA discretization.
        """
        if scalar_rhs.ndim != 4 or scalar_rhs.shape[1] != 1:
            raise ValueError("scalar_rhs must have shape (B,1,H,W)")
        scalar_rhs_hat = torch.fft.fftn(scalar_rhs, dim=(-2, -1), norm="backward")
        k2 = self.kx.to(device=scalar_rhs.device) ** 2 + self.ky.to(device=scalar_rhs.device) ** 2
        diffusion_coeff = self.diffusion_coeff.to(dtype=scalar_rhs.dtype, device=scalar_rhs.device)
        sigma_a_ref = self.sigma_a_ref.to(dtype=scalar_rhs.dtype, device=scalar_rhs.device)
        denom = diffusion_coeff * k2.to(dtype=scalar_rhs.dtype) + sigma_a_ref
        out_hat = scalar_rhs_hat / denom
        return torch.fft.ifftn(out_hat, dim=(-2, -1), norm="backward")

    def _build_reference_transport_inverse(self):
        matrix = self._reference_transport_matrix()
        flat = matrix.reshape(-1, self.n_angles, self.n_angles)
        inv = torch.linalg.inv(flat).reshape(self.sigma0.shape[0], *self.shape, self.n_angles, self.n_angles)
        self._reference_transport_inverse = inv

    @property
    def reference_transport_inverse(self):
        if not hasattr(self, "sigma0"):
            raise RuntimeError("call setup(...) before reference_transport_inverse")
        if self._reference_transport_inverse is None:
            self._build_reference_transport_inverse()
        return self._reference_transport_inverse

    def _reference_transport_matrix(self):
        batch_size = self.sigma0.shape[0]
        device = self.sigma0.device
        dtype = self.complex_dtype
        stream = self._stream_symbol(batch_size, dtype, device).permute(0, 2, 3, 1)
        sigma0 = self.sigma0.to(dtype=dtype, device=device).view(batch_size, 1, 1, 1)
        diag_values = stream + sigma0
        eye = torch.eye(self.n_angles, dtype=dtype, device=device).view(1, 1, 1, self.n_angles, self.n_angles)
        scattering_matrix = (
            self.phase_matrix.to(device=device, dtype=self.float_dtype)
            * self.weights.to(device=device, dtype=self.float_dtype).view(1, self.n_angles)
        ).to(dtype=dtype)
        sigma_s_ref = self.sigma_s_ref.to(dtype=dtype, device=device).view(batch_size, 1, 1, 1, 1)
        matrix = diag_values.unsqueeze(-1) * eye - sigma_s_ref * scattering_matrix.view(
            1, 1, 1, self.n_angles, self.n_angles
        )
        return matrix

    def angular_mode_indices(self, harmonics):
        if harmonics is None:
            return None
        harmonics = int(harmonics)
        if harmonics < 0 or harmonics >= self.n_angles // 2:
            return None
        modes = [0]
        for mode in range(1, harmonics + 1):
            modes.extend([mode, -mode])
        return modes

    def angular_fourier_basis(self, harmonics, dtype=None, device=None):
        modes = self.angular_mode_indices(harmonics)
        if modes is None:
            return None
        device = self.directions.device if device is None else device
        real_dtype = self.float_dtype if dtype is None or not torch.is_complex(torch.empty((), dtype=dtype)) else (
            torch.float64 if dtype == torch.complex128 else torch.float32
        )
        angles = torch.arange(self.n_angles, dtype=real_dtype, device=device) * (2.0 * math.pi / self.n_angles)
        mode_tensor = torch.tensor(modes, dtype=real_dtype, device=device)
        basis = torch.exp(1j * angles[:, None] * mode_tensor[None, :]) / math.sqrt(self.n_angles)
        return basis.to(dtype=dtype or self.complex_dtype)

    def angular_project(self, rhs, harmonics=None):
        if harmonics is None:
            return rhs
        harmonics = int(harmonics)
        if harmonics < 0 or harmonics >= self.n_angles // 2:
            return rhs
        coeff = torch.fft.fft(rhs, dim=1, norm="backward")
        mask = torch.zeros(self.n_angles, dtype=rhs.real.dtype, device=rhs.device)
        mask[0] = 1.0
        if harmonics > 0:
            mask[1 : harmonics + 1] = 1.0
            mask[-harmonics:] = 1.0
        projected = torch.fft.ifft(coeff * mask.view(1, self.n_angles, 1, 1), dim=1, norm="backward")
        return projected

    def reference_transport_solve(self, rhs, harmonics=None):
        if not hasattr(self, "sigma0"):
            raise RuntimeError("call setup(...) before reference_transport_solve")
        rhs = self.angular_project(rhs, harmonics=harmonics)
        rhs_hat = torch.fft.fftn(rhs, dim=(-2, -1), norm="backward")
        rhs_vec = rhs_hat.permute(0, 2, 3, 1).unsqueeze(-1)
        inv = self.reference_transport_inverse.to(dtype=rhs.dtype, device=rhs.device)
        if inv.shape[0] == 1 and rhs.shape[0] > 1:
            inv = inv.expand(rhs.shape[0], -1, -1, -1, -1)
        elif inv.shape[0] != rhs.shape[0]:
            raise ValueError(
                f"reference transport inverse batch ({inv.shape[0]}) does not match rhs batch ({rhs.shape[0]})"
            )
        sol_hat = torch.matmul(
            inv.reshape(-1, self.n_angles, self.n_angles),
            rhs_vec.reshape(-1, self.n_angles, 1),
        )
        sol_hat = sol_hat.reshape(rhs.shape[0], *self.shape, self.n_angles).permute(0, 3, 1, 2)
        return torch.fft.ifftn(sol_hat, dim=(-2, -1), norm="backward")

    def reference_moment_transport_solve(self, rhs, harmonics=None, mode_weights=None):
        """
        Closed low-order angular Fourier reference solve.

        Unlike reference_transport_solve(..., harmonics=K), this applies the
        Galerkin inverse Q_K (Q_K^* A_ref Q_K)^{-1} Q_K^* and therefore keeps
        the correction inside the selected angular moment space.
        """
        if harmonics is None:
            raise ValueError("reference_moment_transport_solve requires explicit harmonics")
        if self.angular_mode_indices(harmonics) is None:
            return self.reference_transport_solve(rhs, harmonics=None)
        if not hasattr(self, "_reference_moment_inverse_cache"):
            self._reference_moment_inverse_cache = {}

        harmonics = int(harmonics)
        basis = self.angular_fourier_basis(harmonics, dtype=rhs.dtype, device=rhs.device)
        n_modes = basis.shape[1]
        cache_key = (harmonics, str(rhs.device), str(rhs.dtype))
        inv = self._reference_moment_inverse_cache.get(cache_key)
        if inv is None:
            matrix = self._reference_transport_matrix().to(dtype=rhs.dtype, device=rhs.device)
            basis_ref = self.angular_fourier_basis(harmonics, dtype=rhs.dtype, device=rhs.device)
            aq = torch.matmul(matrix, basis_ref.view(1, 1, 1, self.n_angles, n_modes))
            low_matrix = torch.matmul(
                basis_ref.conj().transpose(0, 1).view(1, 1, 1, n_modes, self.n_angles),
                aq,
            )
            inv = torch.linalg.inv(low_matrix.reshape(-1, n_modes, n_modes)).reshape(
                self.sigma0.shape[0],
                *self.shape,
                n_modes,
                n_modes,
            )
            self._reference_moment_inverse_cache[cache_key] = inv

        rhs_hat = torch.fft.fftn(rhs, dim=(-2, -1), norm="backward")
        rhs_vec = rhs_hat.permute(0, 2, 3, 1).unsqueeze(-1)
        low_rhs = torch.matmul(
            basis.conj().transpose(0, 1).view(1, 1, 1, n_modes, self.n_angles),
            rhs_vec,
        )
        if inv.shape[0] == 1 and rhs.shape[0] > 1:
            inv = inv.expand(rhs.shape[0], -1, -1, -1, -1)
        elif inv.shape[0] != rhs.shape[0]:
            raise ValueError(f"moment inverse batch ({inv.shape[0]}) does not match rhs batch ({rhs.shape[0]})")
        coeff = torch.matmul(inv.reshape(-1, n_modes, n_modes), low_rhs.reshape(-1, n_modes, 1))
        if mode_weights is not None:
            weights = mode_weights.to(dtype=rhs.dtype, device=rhs.device).reshape(1, 1, 1, n_modes, 1)
            coeff = coeff.reshape(rhs.shape[0], *self.shape, n_modes, 1) * weights
        else:
            coeff = coeff.reshape(rhs.shape[0], *self.shape, n_modes, 1)
        sol_vec = torch.matmul(basis.view(1, 1, 1, self.n_angles, n_modes), coeff)
        sol_hat = sol_vec.squeeze(-1).permute(0, 3, 1, 2)
        return torch.fft.ifftn(sol_hat, dim=(-2, -1), norm="backward")

    def lift_scalar_flux(self, scalar_flux, mode="isotropic"):
        if mode != "isotropic":
            raise NotImplementedError("only isotropic scalar-flux lifting is implemented")
        if scalar_flux.ndim != 4 or scalar_flux.shape[1] != 1:
            raise ValueError("scalar_flux must have shape (B,1,H,W)")
        return scalar_flux.expand(-1, self.n_angles, -1, -1)

    def transport_preconditioner(self, rhs):
        return self.G(rhs)

    def transport_dsa_preconditioner(self, rhs):
        """
        One transport sweep followed by a low-order diffusion synthetic
        correction of the residual left by that sweep.
        """
        transport_correction = self.transport_preconditioner(rhs)
        remaining_residual = rhs - self.A_op(transport_correction)
        scalar_rhs = self.scalar_flux(remaining_residual)
        scalar_correction = self.diffusion_solve(scalar_rhs)
        return transport_correction + self.lift_scalar_flux(scalar_correction, mode="isotropic")

    def transport_tsa_preconditioner(self, rhs, harmonics=None):
        """
        One transport reference correction followed by a low-order reference
        transport-scattering correction of the residual left by that sweep.
        The optional angular Fourier truncation controls the TSA space.
        """
        transport_correction = self.transport_preconditioner(rhs)
        remaining_residual = rhs - self.A_op(transport_correction)
        synthetic_correction = self.reference_transport_solve(remaining_residual, harmonics=harmonics)
        return transport_correction + synthetic_correction

    def transport_moment_tsa_preconditioner(self, rhs, harmonics=None, mode_weights=None):
        """
        One transport reference correction followed by a closed low-order
        angular Fourier transport-scattering correction.
        """
        transport_correction = self.transport_preconditioner(rhs)
        remaining_residual = rhs - self.A_op(transport_correction)
        synthetic_correction = self.reference_moment_transport_solve(
            remaining_residual,
            harmonics=harmonics,
            mode_weights=mode_weights,
        )
        return transport_correction + synthetic_correction

    def apply_preconditioner(self, rhs, kind="transport", tsa_harmonics=None):
        kind = str(kind).lower()
        if kind in ("transport", "green", "g_eta"):
            return self.transport_preconditioner(rhs)
        if kind in ("transport_dsa", "dsa", "synthetic", "synthetic_acceleration"):
            return self.transport_dsa_preconditioner(rhs)
        if kind in ("transport_tsa", "tsa", "transport_synthetic_acceleration"):
            return self.transport_tsa_preconditioner(rhs, harmonics=tsa_harmonics)
        if kind in ("transport_tsa_moment", "tsa_moment", "moment_tsa", "transport_moment_tsa"):
            return self.transport_moment_tsa_preconditioner(rhs, harmonics=tsa_harmonics)
        raise ValueError("kind must be 'transport', 'transport_dsa', 'transport_tsa', or 'transport_tsa_moment'")

    def born_fixed_point(self, psi):
        return self.G(self.V_op(psi) + self.source)

    def born_residual(self, psi):
        return self.born_fixed_point(psi) - psi

    def preconditioned_residual(self, psi, kind="transport", tsa_harmonics=None):
        return self.apply_preconditioner(self.source - self.A_op(psi), kind=kind, tsa_harmonics=tsa_harmonics)

    def metric_norm(self, residual, kind="transport", tsa_harmonics=None):
        return torch.linalg.norm(self.apply_preconditioner(residual, kind=kind, tsa_harmonics=tsa_harmonics))

    def preconditioned_richardson(
        self,
        maxiter=200,
        tol=1e-6,
        relaxation=1.0,
        psi0=None,
        preconditioner="transport",
        tsa_harmonics=None,
    ):
        if not hasattr(self, "source"):
            raise RuntimeError("call setup(...) before preconditioned_richardson")
        psi = (
            torch.zeros_like(self.source)
            if psi0 is None
            else psi0.to(device=self.source.device, dtype=self.source.dtype)
        )
        source_norm = torch.clamp(torch.linalg.norm(self.source), min=1e-30)
        history = []
        for _ in range(int(maxiter)):
            residual = self.source - self.A_op(psi)
            relres = (torch.linalg.norm(residual) / source_norm).item()
            history.append(relres)
            if relres < tol:
                break
            correction = self.apply_preconditioner(
                residual,
                kind=preconditioner,
                tsa_harmonics=tsa_harmonics,
            )
            psi = psi + float(relaxation) * correction
        residual = self.source - self.A_op(psi)
        history.append((torch.linalg.norm(residual) / source_norm).item())
        return psi, history

    def source_iteration(self, *args, **kwargs):
        """
        Backward-compatible alias for the prototype's stationary residual
        iteration. This is not a production variable-coefficient transport sweep
        source iteration.
        """
        return self.preconditioned_richardson(*args, **kwargs)
