"""Layered preconditioner prototype; the original Fourier RTE remains unchanged.

Tangential Fourier diagonalization plus cyclic block PCR of either upwind or
fourth-order compact normal streaming. Optional rank-L exact Nyquist repair.
Factors are material-specific. No training, dense full-space inverse or
grazing-direction division is used.
"""
import math
import torch


class CyclicBlockPCR:
    def __init__(self, diagonal, lower, upper):
        # [tangential frequency, normal point, angle, angle]
        n = diagonal.shape[1]
        if n < 2 or n & (n-1):
            raise ValueError("PCR currently requires a power-of-two normal grid >=2")
        self.n, self.steps = n, []
        a, b, c = diagonal, lower.expand_as(diagonal).clone(), upper.expand_as(diagonal).clone()
        stride = 1
        while stride < n:
            inverse = torch.linalg.inv(a)
            alpha = -b @ torch.roll(inverse, stride, 1)
            beta = -c @ torch.roll(inverse, -stride, 1)
            new_a = a + alpha @ torch.roll(c, stride, 1) + beta @ torch.roll(b, -stride, 1)
            new_b = alpha @ torch.roll(b, stride, 1)
            new_c = beta @ torch.roll(c, -stride, 1)
            self.steps.append((stride, alpha, beta))
            a, b, c = new_a, new_b, new_c
            stride *= 2
        # At the last level both remaining couplings wrap to the same point.
        # Folding BOTH self-couplings is essential for the final 2-point stage.
        self.final_inverse = torch.linalg.inv(a+b+c)
        if not bool(torch.isfinite(self.final_inverse).all()):
            raise RuntimeError("nonfinite PCR factor")
        self.factor_bytes = self.final_inverse.numel()*self.final_inverse.element_size()
        self.factor_bytes += sum(x.numel()*x.element_size()+y.numel()*y.element_size()
                                 for _, x, y in self.steps)

    def __call__(self, rhs):
        # [frequency, point, angle, RHS columns]
        value = rhs
        for stride, alpha, beta in self.steps:
            value = value + alpha @ torch.roll(value, stride, 1) + beta @ torch.roll(value, -stride, 1)
        return self.final_inverse @ value


def background_from_material(t, s):
    if t.shape[0] != 1:
        raise ValueError("one material per factor")
    candidates = []
    for normal_axis, reduction_axis in ((0, -1), (1, -2)):
        profile, indices = t.min(reduction_axis, keepdim=True)
        scattering = torch.gather(s, reduction_axis, indices)
        score = float((t-profile).square().sum())
        candidates.append((score, normal_axis, profile, scattering))
    score, axis, tp, sp = min(candidates, key=lambda item: item[0])
    return axis, tp.expand_as(t).clone(), sp.expand_as(s).clone(), score


class LayeredInverse:
    def __init__(self, operator, *, scheme="compact4", nyquist=True,
                 dtype=torch.complex64, background=None):
        if scheme not in ("compact4", "upwind"):
            raise ValueError(scheme)
        if operator.shape[0] != operator.shape[1]:
            raise ValueError("square grids only in this prototype")
        self.op, self.scheme, self.nyquist, self.dtype = operator, scheme, bool(nyquist), dtype
        self.n, self.l = operator.shape[0], operator.n_angles
        self.real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        if background is None:
            self.axis, self.t_bg, self.s_bg, self.background_score = background_from_material(
                operator.sigma_t, operator.sigma_s)
        else:
            self.axis, self.t_bg, self.s_bg = background
            self.background_score = float((operator.sigma_t-self.t_bg).square().sum())
        t = self.t_bg[0,0,:,0] if self.axis == 0 else self.t_bg[0,0,0,:]
        s = self.s_bg[0,0,:,0] if self.axis == 0 else self.s_bg[0,0,0,:]
        device = t.device
        self.phase = (operator.phase_matrix.to(self.real_dtype)*operator.weights.to(self.real_dtype)[None]).to(dtype)
        # Promote stored geometry before multiplication in the chosen factor precision.
        # Physical V/acceptance independently use the original FP64 operator.
        self.normal = operator.directions[:,self.axis].to(self.real_dtype)
        self.tangent = operator.directions[:,1-self.axis].to(self.real_dtype)
        tangent_k = (operator.ky if self.axis == 0 else operator.kx).flatten().to(self.real_dtype)
        normal_k = (operator.kx if self.axis == 0 else operator.ky).flatten().to(self.real_dtype)
        identity = torch.eye(self.l,dtype=dtype,device=device)
        self.eye = identity
        collision = t.to(dtype)[None,:,None,None]*identity - s.to(dtype)[None,:,None,None]*self.phase
        collision = collision + (1j*tangent_k[:,None,None,None])*torch.diag(self.tangent.to(dtype))[None,None]
        h = operator.dx
        if scheme == "compact4":
            directional = torch.diag((3/(4*h)*self.normal).to(dtype))[None,None]
            diagonal = collision
            lower = .25*torch.roll(collision,1,1)-directional
            upper = .25*torch.roll(collision,-1,1)+directional
        else:
            directional = self.normal/h
            diagonal = collision+torch.diag(directional.abs().to(dtype))[None,None]
            lower = -torch.diag(directional.clamp_min(0).to(dtype))[None,None]
            upper = torch.diag(directional.clamp_max(0).to(dtype))[None,None]
        self.pcr = CyclicBlockPCR(diagonal,lower,upper)
        self.nyquist_z = self.nyquist_inverse = None
        if nyquist:
            # Compact derivative: Nyquist=0. Upwind Nyquist=2|Omega|/h.
            # Replace only that mode by the original stored Fourier normal symbol.
            old = torch.zeros_like(self.normal) if scheme == "compact4" else 2*self.normal.abs()/h
            delta = 1j*normal_k[self.n//2]*self.normal-old
            signs = (1-2*(torch.arange(self.n,device=device)%2)).to(self.real_dtype)
            self.nyquist_signs = signs
            u = signs[None,:,None,None]*torch.diag(delta.to(dtype))[None,None]
            u = u.expand(self.n,-1,-1,-1).contiguous()
            self.nyquist_z = self._base_modes(u)
            reduced = (self.nyquist_z*signs[None,:,None,None]).mean(1)
            self.nyquist_inverse = torch.linalg.inv(identity[None]+reduced)
        self.factor_bytes = self.pcr.factor_bytes
        if self.nyquist_z is not None:
            self.factor_bytes += self.nyquist_z.numel()*self.nyquist_z.element_size()
            self.factor_bytes += self.nyquist_inverse.numel()*self.nyquist_inverse.element_size()

    def _base_modes(self, rhs):
        if self.scheme == "compact4":
            rhs = rhs+.25*(torch.roll(rhs,1,1)+torch.roll(rhs,-1,1))
        return self.pcr(rhs)

    def solve_modes(self, rhs):
        value = self._base_modes(rhs)
        if self.nyquist_z is not None:
            reduced = (value*self.nyquist_signs[None,:,None,None]).mean(1)
            coeff = self.nyquist_inverse @ reduced
            value = value-self.nyquist_z @ coeff[:,None]
        return value

    def __call__(self, rhs):
        field = rhs.to(self.dtype)
        if self.axis:
            field = field.transpose(-1,-2)
        modes = torch.fft.fft(field[0],dim=-1).permute(2,1,0).unsqueeze(-1).contiguous()
        value = self.solve_modes(modes).squeeze(-1).permute(2,1,0)
        value = torch.fft.ifft(value,dim=-1).unsqueeze(0)
        return value.transpose(-1,-2) if self.axis else value

    def surrogate_action(self, value):
        field = value.to(self.dtype)
        if self.axis:
            field = field.transpose(-1,-2)
        n, h = self.n, self.op.dx
        theta = 2*math.pi*torch.fft.fftfreq(n,device=field.device,dtype=self.real_dtype)
        if self.scheme == "compact4":
            symbol = 1j*(3/(2*h))*theta.sin()/(1+.5*theta.cos())
            symbol[n//2] = 0
            normal = symbol[None,:,None]*self.normal[:,None,None]
        else:
            pos = self.normal.clamp_min(0)[:,None,None]
            neg = self.normal.clamp_max(0)[:,None,None]
            normal = (pos*(1-torch.exp(-1j*theta))[None,:,None] +
                      neg*(torch.exp(1j*theta)-1)[None,:,None])/h
        nk = (self.op.kx if self.axis == 0 else self.op.ky).flatten().to(self.real_dtype)
        tk = (self.op.ky if self.axis == 0 else self.op.kx).flatten().to(self.real_dtype)
        if self.nyquist:
            normal[:,n//2,:] = (1j*nk[n//2]*self.normal)[:,None]
        stream = normal+1j*self.tangent[:,None,None]*tk[None,None,:]
        streaming = torch.fft.ifft2(stream[None]*torch.fft.fft2(field))
        t = self.t_bg.transpose(-1,-2) if self.axis else self.t_bg
        s = self.s_bg.transpose(-1,-2) if self.axis else self.s_bg
        result = streaming+t*field-s*torch.einsum("lm,bmxy->blxy",self.phase,field)
        return result.transpose(-1,-2) if self.axis else result

    def perturbation(self, value):
        # Physical V uses products of promoted stored geometry, matching original A64.
        phase = (self.op.phase_matrix.double()*self.op.weights.double()[None]).to(value.dtype)
        dt = self.op.sigma_t.to(value.real.dtype)-self.t_bg.to(value.real.dtype)
        ds = self.op.sigma_s.to(value.real.dtype)-self.s_bg.to(value.real.dtype)
        return dt*value-ds*torch.einsum("lm,bmxy->blxy",phase,value)


class CapturedAction:
    """Private raw CUDA capture, matching the optimized classical setup policy."""
    def __init__(self, action, example):
        self.input = example.clone()
        caller = torch.cuda.current_stream()
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            for _ in range(2):
                self.output = action(self.input)
        caller.wait_stream(self.stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        # Unlike torch.cuda.graph context entry, do not trigger process-wide
        # garbage collection or allocator cache flushing.
        with torch.cuda.stream(self.stream), torch.no_grad():
            self.graph.capture_begin(capture_error_mode="global")
            try:
                self.output = action(self.input)
            except BaseException as body_error:
                try:
                    self.graph.capture_end()
                except BaseException as end_error:
                    raise body_error from end_error
                raise
            else:
                self.graph.capture_end()
        caller.wait_stream(self.stream)
        torch.cuda.synchronize()
        expected = action(example)
        self.input.copy_(example)
        self.graph.replay()
        torch.cuda.synchronize()
        self.relative_capture_difference = float((self.output-expected).norm()/expected.norm().clamp_min(1e-30))

    def __call__(self, rhs):
        self.input.copy_(rhs)
        self.graph.replay()
        return self.output.clone()
