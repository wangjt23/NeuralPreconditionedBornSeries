import torch
import torch.nn.functional as F

from .cbs import ConvergentBornSeries2d


def _complex_to_channels(u):
    return torch.cat([u.real, u.imag], dim=1)


def _channels_to_complex(u):
    return torch.complex(u[:, 0:1], u[:, 1:2])


class ShiftedLaplacianVcycle2d(ConvergentBornSeries2d):
    """
    Geometric V-cycle shifted-Laplacian preconditioner for the same padded
    Helmholtz operator used by ConvergentBornSeries2d.

    The finest-grid operator A_op remains the CBS discretization

        A u = dx^2 * (-Delta u - k(x)^2 u),

    where k(x)^2 includes the repository's existing ABC/PBL boundary treatment.
    The preconditioner applies one or more V-cycles to the shifted operator

        M_beta u = dx^2 * (-Delta u - (1 + i * beta) k(x)^2 u).

    Levels use rediscretized FFT Laplacians, full-weighting restriction,
    bilinear prolongation, and damped Jacobi smoothing.
    """

    def __init__(
        self,
        freq=1,
        N=(96, 96),
        dx=0.125,
        boundary_width=(8, 8),
        boundary_type="pbl",
        shift=0.5,
        precond_discretization="fft",
        num_levels=None,
        min_coarse_size=10,
        pre_smooth=2,
        post_smooth=2,
        coarse_smooth=30,
        direct_coarse=True,
        smoother="jacobi",
        vanka_patch_size=2,
        vanka_stride=1,
        vanka_weight=0.7,
        jacobi_weight=0.6,
        num_cycles=1,
        fdtype=torch.float32,
        device="cpu",
    ):
        super().__init__(
            freq=freq,
            N=N,
            dx=dx,
            boundary_width=boundary_width,
            boundary_type=boundary_type,
            fdtype=fdtype,
            device=device,
        )
        self.shift = float(shift)
        self.precond_discretization = str(precond_discretization).lower()
        if self.precond_discretization not in ("fft", "fd"):
            raise ValueError("precond_discretization must be 'fft' or 'fd'")
        self.num_levels = None if num_levels is None else int(num_levels)
        self.min_coarse_size = int(min_coarse_size)
        self.pre_smooth = int(pre_smooth)
        self.post_smooth = int(post_smooth)
        self.coarse_smooth = int(coarse_smooth)
        self.direct_coarse = bool(direct_coarse)
        self.smoother = str(smoother).lower()
        if self.smoother not in ("jacobi", "vanka"):
            raise ValueError("smoother must be 'jacobi' or 'vanka'")
        self.vanka_patch_size = int(vanka_patch_size)
        self.vanka_stride = int(vanka_stride)
        self.vanka_weight = float(vanka_weight)
        self.jacobi_weight = float(jacobi_weight)
        self.num_cycles = int(num_cycles)
        self.preconditioner_kind = "shifted_laplacian_vcycle"
        self._vanka_cache = {}

        restriction_kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=self.float_dtype,
        ) / 16.0
        self.register_buffer(
            "restriction_kernel",
            restriction_kernel.view(1, 1, 3, 3).repeat(2, 1, 1, 1),
            persistent=False,
        )

    def setup(self, sos, src=None):
        super().setup(sos=sos, src=src)
        self.k2 = self.inv_sos2 * (self.omega0**2)
        self._build_hierarchy()

    def setup_slowness(self, kappa, src=None, squared=True):
        """
        Setup from a slowness field.

        Args:
            kappa: slowness tensor on the physical domain. By default this is
                squared slowness, kappa = 1 / c^2, matching the Helmholtz term
                omega^2 * kappa used throughout this repository.
            src: optional source tensor on the physical domain.
            squared: if False, interpret kappa = 1 / c and square it first.
        """
        kappa = kappa.to(dtype=self.float_dtype, device=self.device)
        if not squared:
            kappa = kappa**2
        sos = torch.rsqrt(torch.clamp(kappa, min=torch.finfo(self.float_dtype).tiny))
        self.setup(sos=sos, src=src)

    def forward(self, rhs):
        return self.apply_preconditioner(rhs)

    def apply_preconditioner(self, rhs):
        if not hasattr(self, "level_shapes"):
            raise RuntimeError("ShiftedLaplacianVcycle2d.setup(...) must be called first.")
        if rhs.shape[-2:] != tuple(self.level_shapes[0]):
            raise ValueError(
                f"rhs spatial shape {tuple(rhs.shape[-2:])} does not match "
                f"padded Helmholtz shape {tuple(self.level_shapes[0])}; call setup first."
            )

        input_dtype = rhs.dtype if torch.is_complex(rhs) else self.complex_dtype
        rhs = rhs.to(self.complex_dtype)
        u = torch.zeros_like(rhs)
        residual = rhs

        for _ in range(self.num_cycles):
            correction = self._v_cycle(level=0, rhs=residual)
            u = u + correction
            if self.num_cycles > 1:
                residual = rhs - self._apply_shifted_operator(level=0, u=u)

        return u.to(input_dtype)

    def shifted_A_op(self, u):
        return self._apply_shifted_operator(level=0, u=u)

    def _build_hierarchy(self):
        self._vanka_cache = {}
        k2 = self.k2.to(self.complex_dtype)
        shape = tuple(int(v) for v in k2.shape[-2:])
        dx = float(self.dx)

        level_shapes = []
        dx_list = []
        p_dist2_list = []
        shifted_k2_list = []
        diag_list = []
        lap_matrix_list = []

        max_levels = self.num_levels if self.num_levels is not None else 10**9
        level = 0
        while True:
            level_shapes.append(shape)
            dx_list.append(dx)
            p_dist2 = self._make_p_dist2(shape=shape, dx=dx, device=k2.device)
            shifted_k2 = (1.0 + 1.0j * self.shift) * k2
            if self.precond_discretization == "fft":
                diag = (p_dist2.mean().view(1, 1, 1, 1) - shifted_k2) * (dx**2)
            else:
                diag = 4.0 - shifted_k2 * (dx**2)

            p_dist2_list.append(p_dist2)
            shifted_k2_list.append(shifted_k2)
            diag_list.append(self._safe_diag(diag))
            lap_matrix_list.append(None)

            if level + 1 >= max_levels:
                break
            if min(shape) <= self.min_coarse_size:
                break

            next_shape = ((shape[0] + 1) // 2, (shape[1] + 1) // 2)
            if next_shape == shape:
                break

            k2 = self._restrict_complex(k2)
            shape = tuple(int(v) for v in k2.shape[-2:])
            dx *= 2.0
            level += 1

        self.level_shapes = level_shapes
        self.dx_list = dx_list
        self.p_dist2_list = p_dist2_list
        self.shifted_k2_list = shifted_k2_list
        self.diag_list = diag_list
        self.lap_matrix_list = lap_matrix_list
        if self.direct_coarse:
            coarse_level = len(self.level_shapes) - 1
            self.lap_matrix_list[coarse_level] = self._make_lap_matrix(coarse_level)

    def _make_p_dist2(self, shape, dx, device):
        px = 2 * torch.pi * torch.fft.fftfreq(shape[0], d=dx, device=device)
        py = 2 * torch.pi * torch.fft.fftfreq(shape[1], d=dx, device=device)
        px = px.to(self.float_dtype).view(-1, 1)
        py = py.to(self.float_dtype).view(1, -1)
        return (px**2 + py**2).to(device)

    def _safe_diag(self, diag):
        eps = torch.finfo(self.float_dtype).eps
        return torch.where(torch.abs(diag) > eps, diag, diag + eps)

    def _make_lap_matrix(self, level):
        shape = self.level_shapes[level]
        n = shape[0] * shape[1]
        eye = torch.eye(n, dtype=self.complex_dtype, device=self.shifted_k2_list[level].device)
        basis = eye.reshape(n, 1, shape[0], shape[1])
        if self.precond_discretization == "fft":
            p_dist2 = self.p_dist2_list[level]
            dx = self.dx_list[level]
            lap_basis = torch.fft.ifftn(
                p_dist2 * torch.fft.fftn(basis, dim=[-1, -2], norm="backward"),
                dim=[-1, -2],
                norm="backward",
            ) * (dx**2)
        else:
            lap_basis = self._apply_fd_negative_laplacian(basis)
        return lap_basis.reshape(n, n).transpose(0, 1).contiguous()

    def _apply_shifted_operator(self, level, u):
        shifted_k2 = self.shifted_k2_list[level]
        dx = self.dx_list[level]

        if self.precond_discretization == "fft":
            p_dist2 = self.p_dist2_list[level]
            lap_term = torch.fft.ifftn(
                p_dist2 * torch.fft.fftn(u, dim=[-1, -2], norm="backward"),
                dim=[-1, -2],
                norm="backward",
            )
            return (lap_term - shifted_k2 * u) * (dx**2)

        lap_term = self._apply_fd_negative_laplacian(u)
        return lap_term - shifted_k2 * u * (dx**2)

    def _apply_fd_negative_laplacian(self, u):
        padded = F.pad(u, (1, 1, 1, 1), mode="constant", value=0)
        center = padded[..., 1:-1, 1:-1]
        left = padded[..., 1:-1, :-2]
        right = padded[..., 1:-1, 2:]
        up = padded[..., :-2, 1:-1]
        down = padded[..., 2:, 1:-1]
        return 4.0 * center - left - right - up - down

    def _smooth(self, level, u, rhs, num_sweeps):
        if self.smoother == "vanka" and self.precond_discretization == "fd":
            return self._vanka_smooth(level=level, u=u, rhs=rhs, num_sweeps=num_sweeps)

        diag = self.diag_list[level]
        for _ in range(num_sweeps):
            residual = rhs - self._apply_shifted_operator(level=level, u=u)
            u = u + self.jacobi_weight * residual / diag
        return u

    def _patch_starts(self, size):
        patch_size = self.vanka_patch_size
        stride = max(self.vanka_stride, 1)
        if size <= patch_size:
            return torch.zeros(1, dtype=torch.long)
        starts = torch.arange(0, size - patch_size + 1, stride, dtype=torch.long)
        last = torch.tensor([size - patch_size], dtype=torch.long)
        return torch.unique(torch.cat([starts, last]))

    def _vanka_smooth(self, level, u, rhs, num_sweeps):
        patch_size = self.vanka_patch_size
        if patch_size != 2:
            raise NotImplementedError("Only 2x2 Vanka patches are implemented.")

        cache = self._get_vanka_cache(level=level, u=u)
        patch_indices = cache["patch_indices"]
        flat_indices = cache["flat_indices"]
        count = cache["count"]
        local_inverse = cache["local_inverse"]

        height, width = self.level_shapes[level]
        batch_channels = u.shape[0] * u.shape[1]
        for _ in range(num_sweeps):
            residual = rhs - self._apply_shifted_operator(level=level, u=u)
            residual_flat = residual.reshape(batch_channels, height * width)
            local_rhs = residual_flat[:, patch_indices]
            local_correction = torch.matmul(
                local_inverse,
                local_rhs.unsqueeze(-1),
            ).squeeze(-1)

            correction = torch.zeros_like(residual_flat)
            correction.scatter_add_(
                1,
                flat_indices,
                local_correction.reshape(batch_channels, -1),
            )
            correction = correction / count
            u = u + self.vanka_weight * correction.reshape_as(u)

        return u

    def _get_vanka_cache(self, level, u):
        patch_size = self.vanka_patch_size
        key = (
            level,
            str(u.device),
            u.dtype,
            u.shape[0],
            u.shape[1],
            self.vanka_stride,
            self.vanka_weight,
        )
        if key in self._vanka_cache:
            return self._vanka_cache[key]

        height, width = self.level_shapes[level]
        row_starts = self._patch_starts(height).to(device=u.device)
        col_starts = self._patch_starts(width).to(device=u.device)
        base = (row_starts[:, None] * width + col_starts[None, :]).reshape(-1)
        offsets = torch.tensor([0, 1, width, width + 1], device=u.device)
        patch_indices = base[:, None] + offsets[None, :]
        patch_count = patch_indices.shape[0]
        dofs = patch_size * patch_size

        offdiag = torch.zeros(dofs, dofs, dtype=u.dtype, device=u.device)
        for a, b in ((0, 1), (2, 3), (0, 2), (1, 3)):
            offdiag[a, b] = -1.0
            offdiag[b, a] = -1.0

        shifted_diag = (
            4.0 - self.shifted_k2_list[level].to(dtype=u.dtype) * (self.dx_list[level] ** 2)
        ).reshape(u.shape[0], 1, -1)
        if shifted_diag.shape[1] == 1 and u.shape[1] > 1:
            shifted_diag = shifted_diag.expand(-1, u.shape[1], -1)

        batch_channels = u.shape[0] * u.shape[1]
        flat_indices = patch_indices.reshape(1, -1).expand(batch_channels, -1)
        count = torch.zeros(batch_channels, height * width, dtype=self.float_dtype, device=u.device)
        count.scatter_add_(
            1,
            flat_indices,
            torch.ones_like(flat_indices, dtype=self.float_dtype),
        )
        count = torch.clamp(count, min=1.0)

        diag_flat = shifted_diag.reshape(batch_channels, height * width)
        local_diag = diag_flat[:, patch_indices]
        local_matrix = offdiag.view(1, 1, dofs, dofs).expand(
            batch_channels, patch_count, dofs, dofs
        ).clone()
        diag_view = local_matrix.diagonal(dim1=-2, dim2=-1)
        diag_view.copy_(local_diag)
        local_inverse = torch.linalg.inv(local_matrix.reshape(-1, dofs, dofs)).reshape(
            batch_channels, patch_count, dofs, dofs
        )

        cache = {
            "patch_indices": patch_indices,
            "flat_indices": flat_indices,
            "count": count,
            "local_inverse": local_inverse,
        }
        self._vanka_cache[key] = cache
        return cache

    def _v_cycle(self, level, rhs):
        u = torch.zeros_like(rhs)
        last_level = len(self.level_shapes) - 1

        if level == last_level:
            if self.direct_coarse:
                return self._coarse_solve(level=level, rhs=rhs)
            return self._smooth(level=level, u=u, rhs=rhs, num_sweeps=self.coarse_smooth)

        u = self._smooth(level=level, u=u, rhs=rhs, num_sweeps=self.pre_smooth)
        residual = rhs - self._apply_shifted_operator(level=level, u=u)
        coarse_rhs = self._restrict_complex(residual)
        coarse_correction = self._v_cycle(level=level + 1, rhs=coarse_rhs)
        u = u + self._prolong_complex(coarse_correction, target_shape=rhs.shape[-2:])
        u = self._smooth(level=level, u=u, rhs=rhs, num_sweeps=self.post_smooth)
        return u

    def _coarse_solve(self, level, rhs):
        base_matrix = self.lap_matrix_list[level]
        shifted_diag = (self.shifted_k2_list[level] * (self.dx_list[level] ** 2)).reshape(
            rhs.shape[0], 1, -1
        )
        if shifted_diag.shape[1] == 1 and rhs.shape[1] > 1:
            shifted_diag = shifted_diag.expand(-1, rhs.shape[1], -1)
        rhs_flat = rhs.reshape(rhs.shape[0], rhs.shape[1], -1)
        out = torch.empty_like(rhs_flat)
        for batch_idx in range(rhs.shape[0]):
            for channel_idx in range(rhs.shape[1]):
                matrix = base_matrix - torch.diag(shifted_diag[batch_idx, channel_idx])
                out[batch_idx, channel_idx] = torch.linalg.solve(
                    matrix, rhs_flat[batch_idx, channel_idx]
                )
        return out.reshape_as(rhs)

    def _restrict_complex(self, u):
        u_ri = _complex_to_channels(u)
        kernel = self.restriction_kernel.to(device=u_ri.device, dtype=u_ri.dtype)
        pad_mode = "circular" if self.precond_discretization == "fft" else "constant"
        u_ri = F.pad(u_ri, (1, 1, 1, 1), mode=pad_mode)
        coarse_ri = F.conv2d(u_ri, kernel, stride=2, groups=2)
        return _channels_to_complex(coarse_ri)

    def _prolong_complex(self, u, target_shape):
        u_ri = _complex_to_channels(u)
        fine_ri = F.interpolate(
            u_ri,
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        )
        return _channels_to_complex(fine_ri)


class FiniteDifferenceShiftedLaplacianVcycle2d(ShiftedLaplacianVcycle2d):
    """
    Five-point finite-difference Helmholtz operator with a matching
    finite-difference shifted-Laplacian V-cycle preconditioner.

    Setup and absorbing-layer construction are inherited from CBS so the
    padded medium and PBL/ABC coefficient modification stay consistent with
    the rest of the repository. The finest-grid operator is

        A_h u = (-Delta_h u) - dx^2 * k(x)^2 u,

    with zero exterior values at the outer edge of the padded absorbing layer.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("precond_discretization", "fd")
        super().__init__(*args, **kwargs)

    def A_op(self, u):
        k2 = self.k2.to(dtype=u.dtype, device=u.device)
        return self._apply_fd_negative_laplacian(u) - k2 * u * (self.dx**2)


class MatrixFreeShiftedLaplacian2d(FiniteDifferenceShiftedLaplacianVcycle2d):
    """
    Matrix-free PyTorch shifted-Laplacian preconditioner for the five-point
    finite-difference Helmholtz operator.

    The preconditioner approximately solves the shifted equation

        (-Delta_h - (1 + i beta) dx^2 k(x)^2) z = r

    with a fixed number of inner FGMRES iterations. No sparse matrix is formed;
    all applications use the same PyTorch stencil as A_op. The inner Krylov
    method can be unpreconditioned, Jacobi-preconditioned, or V-cycle
    preconditioned. With inner_preconditioner="vcycle" and direct_coarse=False,
    the complete shifted solve remains matrix-free.
    """

    def __init__(
        self,
        *args,
        inner_maxiter=8,
        inner_preconditioner="jacobi",
        inner_jacobi_sweeps=1,
        inner_vcycles=1,
        **kwargs,
    ):
        kwargs.setdefault("direct_coarse", False)
        super().__init__(*args, **kwargs)
        self.inner_maxiter = int(inner_maxiter)
        self.inner_preconditioner = str(inner_preconditioner).lower()
        if self.inner_preconditioner not in ("identity", "jacobi", "vcycle"):
            raise ValueError("inner_preconditioner must be 'identity', 'jacobi', or 'vcycle'")
        self.inner_jacobi_sweeps = int(inner_jacobi_sweeps)
        self.inner_vcycles = int(inner_vcycles)
        self.preconditioner_kind = "matrix_free_shifted_laplacian"

    def apply_preconditioner(self, rhs):
        if not hasattr(self, "level_shapes"):
            raise RuntimeError("MatrixFreeShiftedLaplacian2d.setup(...) must be called first.")
        if rhs.shape[-2:] != tuple(self.level_shapes[0]):
            raise ValueError(
                f"rhs spatial shape {tuple(rhs.shape[-2:])} does not match "
                f"padded Helmholtz shape {tuple(self.level_shapes[0])}; call setup first."
            )

        input_dtype = rhs.dtype if torch.is_complex(rhs) else self.complex_dtype
        rhs = rhs.to(self.complex_dtype)
        return self._fixed_fgmres(
            A=self.shifted_A_op,
            b=rhs,
            M=self._inner_preconditioner,
            maxiter=self.inner_maxiter,
        ).to(input_dtype)

    def _inner_preconditioner(self, rhs):
        if self.inner_preconditioner == "identity":
            return rhs
        if self.inner_preconditioner == "vcycle":
            u = torch.zeros_like(rhs)
            residual = rhs
            for _ in range(max(self.inner_vcycles, 1)):
                correction = self._v_cycle(level=0, rhs=residual)
                u = u + correction
                residual = rhs - self.shifted_A_op(u)
            return u

        diag = self.diag_list[0]
        u = torch.zeros_like(rhs)
        for _ in range(max(self.inner_jacobi_sweeps, 1)):
            residual = rhs - self.shifted_A_op(u)
            u = u + self.jacobi_weight * residual / diag
        return u

    def _fixed_fgmres(self, A, b, M, maxiter):
        maxiter = int(maxiter)
        if maxiter < 1:
            return torch.zeros_like(b)

        b_norm = torch.norm(b)
        if b_norm.item() <= 1e-30:
            return torch.zeros_like(b)

        arnoldi_tol = 1e-14
        v = b / b_norm
        V = [v]
        Z = []
        H = torch.zeros((maxiter + 1, maxiter), dtype=b.dtype, device=b.device)
        actual_iters = 0

        for j in range(maxiter):
            z = M(V[j])
            Z.append(z)
            w = A(z)

            for i in range(j + 1):
                H[i, j] = torch.sum(w * V[i].conj())
                w = w - H[i, j] * V[i]

            h_next = torch.norm(w)
            H[j + 1, j] = h_next
            actual_iters = j + 1
            if h_next.item() <= arnoldi_tol:
                break
            if j + 1 < maxiter:
                V.append(w / h_next)

        H_j = H[: actual_iters + 1, :actual_iters]
        e1 = torch.zeros(actual_iters + 1, dtype=b.dtype, device=b.device)
        e1[0] = b_norm
        y = torch.linalg.lstsq(H_j, e1).solution[:actual_iters]

        z_tensor = torch.stack(Z[:actual_iters], dim=0)
        y_shape = [actual_iters] + [1] * (z_tensor.ndim - 1)
        return torch.sum(z_tensor * y.reshape(y_shape), dim=0)


class SparseShiftedLaplacian2d(ConvergentBornSeries2d):
    """
    Finite-difference Helmholtz operator with a sparse shifted-Laplacian
    preconditioner.

    This is a CPU-oriented reference implementation for competitive 2-D
    comparisons: the physical operator is the same zero-exterior five-point
    stencil as FiniteDifferenceShiftedLaplacianVcycle2d, while the shifted
    preconditioner is factored once with SciPy sparse LU and reused inside
    Krylov iterations.
    """

    def __init__(
        self,
        freq=1,
        N=(96, 96),
        dx=0.125,
        boundary_width=(8, 8),
        boundary_type="pbl",
        shift=0.5,
        fdtype=torch.float32,
        device="cpu",
    ):
        super().__init__(
            freq=freq,
            N=N,
            dx=dx,
            boundary_width=boundary_width,
            boundary_type=boundary_type,
            fdtype=fdtype,
            device=device,
        )
        self.shift = float(shift)
        self.preconditioner_kind = "sparse_shifted_laplacian"

    def setup(self, sos, src=None):
        super().setup(sos=sos, src=src)
        self.k2 = self.inv_sos2 * (self.omega0**2)
        self._build_sparse_factorizations()

    def setup_slowness(self, kappa, src=None, squared=True):
        kappa = kappa.to(dtype=self.float_dtype, device=self.device)
        if not squared:
            kappa = kappa**2
        sos = torch.rsqrt(torch.clamp(kappa, min=torch.finfo(self.float_dtype).tiny))
        self.setup(sos=sos, src=src)

    def forward(self, rhs):
        return self.apply_preconditioner(rhs)

    def A_op(self, u):
        k2 = self.k2.to(dtype=u.dtype, device=u.device)
        return self._apply_fd_negative_laplacian(u) - k2 * u * (self.dx**2)

    def shifted_A_op(self, u):
        k2 = self.k2.to(dtype=u.dtype, device=u.device)
        return self._apply_fd_negative_laplacian(u) - (1.0 + 1.0j * self.shift) * k2 * u * (
            self.dx**2
        )

    def _apply_fd_negative_laplacian(self, u):
        padded = F.pad(u, (1, 1, 1, 1), mode="constant", value=0)
        center = padded[..., 1:-1, 1:-1]
        left = padded[..., 1:-1, :-2]
        right = padded[..., 1:-1, 2:]
        up = padded[..., :-2, 1:-1]
        down = padded[..., 2:, 1:-1]
        return 4.0 * center - left - right - up - down

    def _build_sparse_factorizations(self):
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla
        except ImportError as exc:
            raise ImportError("SparseShiftedLaplacian2d requires scipy.") from exc

        shape = tuple(int(v) for v in self.k2.shape[-2:])
        height, width = shape
        dtype = "complex128" if self.complex_dtype == torch.complex128 else "complex64"
        tx = sp.diags(
            [-1.0, 2.0, -1.0],
            offsets=[-1, 0, 1],
            shape=(width, width),
            dtype=dtype,
            format="csc",
        )
        ty = sp.diags(
            [-1.0, 2.0, -1.0],
            offsets=[-1, 0, 1],
            shape=(height, height),
            dtype=dtype,
            format="csc",
        )
        lap = sp.kron(sp.eye(height, dtype=dtype, format="csc"), tx, format="csc")
        lap = lap + sp.kron(ty, sp.eye(width, dtype=dtype, format="csc"), format="csc")

        k2_cpu = self.k2.detach().cpu().numpy()
        self._sparse_shape = shape
        self._sparse_lu = []
        for batch_idx in range(k2_cpu.shape[0]):
            shifted_diag = (1.0 + 1.0j * self.shift) * k2_cpu[batch_idx, 0].reshape(-1)
            matrix = lap - sp.diags(shifted_diag * (self.dx**2), offsets=0, format="csc")
            self._sparse_lu.append(spla.splu(matrix))

    def apply_preconditioner(self, rhs):
        if not hasattr(self, "_sparse_lu"):
            raise RuntimeError("SparseShiftedLaplacian2d.setup(...) must be called first.")
        if rhs.shape[-2:] != self._sparse_shape:
            raise ValueError(
                f"rhs spatial shape {tuple(rhs.shape[-2:])} does not match "
                f"padded Helmholtz shape {self._sparse_shape}; call setup first."
            )

        input_dtype = rhs.dtype if torch.is_complex(rhs) else self.complex_dtype
        rhs_work = rhs.to(self.complex_dtype)
        rhs_cpu = rhs_work.detach().cpu()
        out_cpu = torch.empty_like(rhs_cpu)
        for batch_idx in range(rhs_cpu.shape[0]):
            solver = self._sparse_lu[min(batch_idx, len(self._sparse_lu) - 1)]
            for channel_idx in range(rhs_cpu.shape[1]):
                flat_rhs = rhs_cpu[batch_idx, channel_idx].reshape(-1).numpy()
                flat_out = solver.solve(flat_rhs)
                out_cpu[batch_idx, channel_idx] = torch.from_numpy(flat_out.reshape(self._sparse_shape))
        return out_cpu.to(device=rhs.device, dtype=input_dtype)
