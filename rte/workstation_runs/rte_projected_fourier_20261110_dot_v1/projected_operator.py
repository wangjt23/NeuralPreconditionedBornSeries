"""Exact finite P8 representation of stored material multiplication; CPU only."""
import time
import hashlib
import torch

MODES = list(range(-8, 9))

def tensor_hash(x):
    return hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()

def relative(error, reference):
    """Per batch, strict zero and scale-safe nonzero relative norm."""
    a, b = error.reshape(error.shape[0], -1), reference.reshape(reference.shape[0], -1)
    active = (b != 0).any(dim=1)
    scale = b.abs().amax(dim=1)
    safe = torch.where(active, scale, torch.ones_like(scale))
    den = (b / safe[:, None]).norm(dim=1)
    value = (a / safe[:, None]).norm(dim=1) / torch.where(active, den, torch.ones_like(den))
    return torch.where(active, value, torch.where((a != 0).any(dim=1), torch.full_like(value, float('inf')), torch.zeros_like(value)))

class ProjectedOperator:
    def __init__(self, high, factor):
        start = time.perf_counter()
        self.high, self.factor = high, factor
        self.slots = torch.tensor([k % 128 for k in MODES], dtype=torch.long)
        d = torch.tensor(MODES)[:, None] - torch.tensor(MODES)[None, :]
        self.difference_slots = d.remainder(128)
        self.ct = (torch.fft.fft(high.sigma_t[0, 0].to(torch.complex128), dim=-1) / 128)[:, self.difference_slots]
        self.cs = (torch.fft.fft(high.sigma_s[0, 0].to(torch.complex128), dim=-1) / 128)[:, self.difference_slots]
        self.symbol = high._stream_symbol(1, torch.complex128, high.source.device).index_select(-1, self.slots).clone()
        self.weights = high.weights.clone()
        self.phase = high.phase_matrix.to(torch.complex128)
        self.action_calls = 0
        self.inverse_profiles = []
        self.setup_seconds = time.perf_counter() - start
        self.bytes = sum(x.numel() * x.element_size() for x in [self.slots, self.difference_slots, self.ct, self.cs, self.symbol, self.weights, self.phase])
        self.fingerprints = {k: tensor_hash(getattr(self, k)) for k in ['ct', 'cs', 'symbol', 'weights', 'phase']}

    def restrict(self, x):
        return torch.fft.fft(x, dim=-1, norm='ortho').index_select(-1, self.slots)

    def lift(self, z):
        f = torch.zeros((*z.shape[:-1], 128), dtype=z.dtype, device=z.device)
        f[..., self.slots] = z
        return torch.fft.ifft(f, dim=-1, norm='ortho')

    def action(self, z):
        self.action_calls += 1
        streaming = torch.fft.ifft(self.symbol * torch.fft.fft(z, dim=-2), dim=-2)
        weighted = self.weights[None, :, None, None] * z
        scattered = torch.einsum('lm,bmnk->blnk', self.phase, weighted)
        return streaming + torch.einsum('nkl,banl->bank', self.ct, z) - torch.einsum('nkl,banl->bank', self.cs, scattered)

    def inverse(self, z):
        out = torch.zeros_like(z)
        records = []
        for j, k in enumerate(MODES):
            q = z[..., j].permute(1, 2, 0).reshape(1536, z.shape[0])
            x, st = self.factor.blocks[k].repaired(q, tolerance=1e-10, max_corrections=3)
            records.append(dict(k=k, **st))
            if not st['success']:
                self.inverse_profiles.append(records)
                raise RuntimeError('Original-G repair failure in reduced inverse: ' + str(k))
            out[..., j] = x.reshape(12, 128, z.shape[0]).permute(2, 0, 1)
        self.inverse_profiles.append(records)
        return out
