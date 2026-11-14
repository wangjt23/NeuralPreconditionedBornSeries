"""Oracle global-coherence repair around an independently compressed response.

For orthonormal V and exact B=H^-1, set K = K0 + (B V-K0 V)V*.
Then I-H K = (I-H K0)(I-VV*). The residual-defect spectral/Frobenius
norm cannot increase in exact arithmetic. This does NOT guarantee a faster
physical solve. V is estimated from residual-defect right singular vectors.
The dense inverse used to construct the repair remains an expensive oracle.
"""

import torch


class RepairedResponse:
    def __init__(self, base, correction, basis):
        self.base, self.correction, self.basis = base, correction, basis
        self.dimension, self.groups, self.records = base.dimension, base.groups, base.records

    def __call__(self, rhs):
        return self.base(rhs)+(rhs@self.basis.conj())@self.correction.T

    def to(self, *, dtype):
        return RepairedResponse(self.base.to(dtype=dtype), self.correction.to(dtype), self.basis.to(dtype))

    def diagnostics(self):
        result = self.base.diagnostics()
        extra = self.basis.numel()+self.correction.numel()
        byte_count = extra*self.basis.element_size()
        result['repair_rank'] = self.basis.shape[1]
        result['coefficient_scalars'] += extra
        result['coefficient_bytes'] += byte_count
        result['stored_bytes'] += byte_count
        result['stored_scalar_fraction'] += extra/self.dimension**2
        result['ideal_unpadded_fraction'] += extra/self.dimension**2
        result['total_byte_fraction'] += extra/self.dimension**2
        result['complex_multiply_adds'] += extra
        return result


@torch.no_grad()
def residual_repair_basis(matrix, inverse, response, rank=32, oversample=8, powers=2, seed=418):
    dimension = response.dimension
    if not 1 <= rank <= dimension or powers < 0 or oversample < 0:
        raise ValueError('invalid sketch rank, oversampling or power count')
    width = min(dimension, rank+oversample)
    generator = torch.Generator(device=matrix.device).manual_seed(seed)
    probes = torch.randn(width, dimension, dtype=matrix.dtype, device=matrix.device, generator=generator)
    defect = lambda rhs: rhs-response(rhs)@matrix.T
    adjoint = lambda rhs: rhs-response.adjoint(rhs@matrix.conj())
    orthogonal = lambda rows: torch.linalg.qr(rows.T, mode='reduced')[0].T
    left = orthogonal(defect(probes))
    for _ in range(powers):
        right = orthogonal(adjoint(left))
        left = orthogonal(defect(right))
    # E* Q_L has right singular vectors of E as LEFT singular vectors.
    vectors, singular, _ = torch.linalg.svd(adjoint(left).T, full_matrices=False)
    basis = vectors[:, :rank]
    correction = inverse@basis-response(basis.T).T
    return correction, basis, singular
