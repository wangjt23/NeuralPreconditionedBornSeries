"""Recursive, directed H-matrix response oracle with a genuinely factored action.

Teacher acquisition is dense and deliberately NOT part of a scalable solver.
The online representation retains only grouped dense leaves and low-rank far
blocks. Residual weighting controls ||H[:,I] (B_IJ - Btilde_IJ)||_F, not a
global operator-norm bound. No Hermitian/SPD claim is made for the result.
"""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Cluster:
    level: int
    x: int
    y: int

    def children(self):
        return [Cluster(self.level+1, 2*self.x+i, 2*self.y+j)
                for i in range(2) for j in range(2)]


def separated(a, b):
    """Periodic strong admissibility, including wraparound neighbours."""
    if a.level != b.level:
        raise ValueError('this block tree uses equal-level clusters')
    count = 2**a.level
    dx, dy = abs(a.x-b.x), abs(a.y-b.y)
    return max(min(dx, count-dx), min(dy, count-dy)) >= 2


def retained_rank(singular, tolerance):
    if tolerance < 0 or tolerance >= 1:
        raise ValueError('relative tolerance must lie in [0, 1)')
    if tolerance == 0:
        return len(singular)
    energy = singular.square()
    if not bool(energy.sum() > 0):
        return 0
    return min(len(singular), int(torch.searchsorted(
        energy.cumsum(0)/energy.sum(), 1-tolerance*tolerance))+1)


class HierarchicalResponse:
    """A coefficient-conditioned, residual-linear map; NOT a trained network.

    Blocks are grouped by (kind, dimension, padded rank), permitting a bounded
    number of batched matrix products per application. Indices and padding are
    explicitly included in the storage diagnostics. No teacher tensors survive
    construction. Reconstructing a dense matrix is reserved for small tests.
    """

    def __init__(self, dimension, groups, records):
        self.dimension, self.groups, self.records = dimension, groups, records

    @classmethod
    @torch.no_grad()
    def from_dense(cls, inverse, channels, size, *, matrix=None,
                   tolerance=.001, leaf_size=2, rank_multiple=4, metric='residual'):
        if size < leaf_size or size % leaf_size or size & (size-1) or leaf_size & (leaf_size-1):
            raise ValueError('size and leaf_size must be powers of two with leaf_size <= size')
        if rank_multiple < 1:
            raise ValueError('rank_multiple must be positive')
        retained_rank(torch.ones(1), tolerance)
        dimension = channels*size*size
        if inverse.shape != (dimension, dimension):
            raise ValueError('teacher has incorrect dimensions')
        if matrix is not None and matrix.shape != inverse.shape:
            raise ValueError('residual metric matrix has incorrect dimensions')
        if metric not in ('residual', 'energy', 'euclidean') or (metric == 'energy' and matrix is None):
            raise ValueError('energy weighting requires a matrix; invalid metric')
        grid = torch.arange(dimension, device=inverse.device).reshape(channels, size, size)
        indices, triangular, pending, records = {}, {}, {}, []
        last_level = int(math.log2(size//leaf_size))

        def ids(cluster):
            if cluster not in indices:
                width = size//2**cluster.level
                indices[cluster] = grid[:, cluster.x*width:(cluster.x+1)*width,
                                        cluster.y*width:(cluster.y+1)*width].flatten()
            return indices[cluster]

        def weight(cluster):
            if cluster not in triangular:
                row = ids(cluster)
                if metric == 'energy':
                    local = matrix[row[:, None], row[None, :]]
                    triangular[cluster] = torch.linalg.cholesky((local+local.mH)/2).mH
                else:
                    triangular[cluster] = torch.linalg.qr(matrix[:, row], mode='r')[1]
            return triangular[cluster]

        def visit(a, b):
            row, col = ids(a), ids(b)
            n = len(row)
            block, factors, rank, padded = None, None, n, n
            far = separated(a, b)
            if far and tolerance > 0:
                block = inverse[row[:, None], col[None, :]]
                if matrix is not None and metric != 'euclidean':
                    r = weight(a)
                    weighted = r @ block
                    right = weight(b) if metric == 'energy' else None
                    if right is not None:
                        weighted = weighted @ right.mH
                else:
                    r, right, weighted = None, None, block
                u, s, vh = torch.linalg.svd(weighted, full_matrices=False)
                rank = retained_rank(s, tolerance)
                padded = math.ceil(rank/rank_multiple)*rank_multiple
                # Refine an unprofitable far block; do not count unpadded
                # factors as the storage of an actually padded representation.
                if 2*n*padded < n*n:
                    left = u[:, :rank]*s[:rank]
                    if r is not None:
                        left = torch.linalg.solve_triangular(r, left, upper=True)
                    right_factor = vh[:rank]
                    if right is not None:
                        right_factor = torch.linalg.solve_triangular(right, right_factor.mH, upper=True).mH
                    up = inverse.new_zeros(n, padded)
                    vp = inverse.new_zeros(padded, n)
                    up[:, :rank], vp[:rank] = left, right_factor
                    factors = (up, vp)
            if factors is None and a.level < last_level:
                for ca in a.children():
                    for cb in b.children():
                        visit(ca, cb)
                return
            kind = 'low_rank' if factors is not None else 'dense'
            if factors is None:
                if block is None:
                    block = inverse[row[:, None], col[None, :]]
                factors = (block,)
                rank = padded = n
            key = (kind, n, padded)
            pending.setdefault(key, []).append((row, col, factors))
            records.append(dict(level=a.level, row_x=a.x, row_y=a.y,
                col_x=b.x, col_y=b.y, dimension=n, kind=kind,
                rank=rank, padded_rank=padded, far=far))

        visit(Cluster(0, 0, 0), Cluster(0, 0, 0))
        # Break the recursive closure's self-reference so temporary teacher
        # views/factors need not wait for cyclic garbage collection.
        del visit
        groups = []
        for (kind, n, rank), blocks in sorted(pending.items()):
            groups.append(dict(kind=kind, rows=torch.stack([x[0] for x in blocks]),
                cols=torch.stack([x[1] for x in blocks]),
                factors=tuple(torch.stack([x[2][k] for x in blocks]) for k in range(len(blocks[0][2])))))
        return cls(dimension, groups, records)

    def __call__(self, rhs):
        if rhs.ndim != 2 or rhs.shape[1] != self.dimension:
            raise ValueError('expected batch x dimension input')
        result = torch.zeros_like(rhs)
        for group in self.groups:
            # [groups, block_dimension, batch] works for both batch=1 and probes.
            gathered = rhs[:, group['cols']].permute(1, 2, 0)
            factors = group['factors']
            if group['kind'] == 'dense':
                values = torch.bmm(factors[0], gathered)
            else:
                values = torch.bmm(factors[0], torch.bmm(factors[1], gathered))
            result.index_add_(1, group['rows'].flatten(), values.permute(2, 0, 1).flatten(1))
        return result

    def adjoint(self, rhs):
        if rhs.ndim != 2 or rhs.shape[1] != self.dimension:
            raise ValueError('expected batch x dimension input')
        result = torch.zeros_like(rhs)
        for group in self.groups:
            gathered = rhs[:, group['rows']].permute(1, 2, 0)
            factors = group['factors']
            if group['kind'] == 'dense':
                values = torch.bmm(factors[0].mH, gathered)
            else:
                values = torch.bmm(factors[1].mH, torch.bmm(factors[0].mH, gathered))
            result.index_add_(1, group['cols'].flatten(), values.permute(2, 0, 1).flatten(1))
        return result

    def diagnostics(self):
        coefficients = sum(f.numel() for g in self.groups for f in g['factors'])
        coefficient_bytes = sum(f.numel()*f.element_size() for g in self.groups for f in g['factors'])
        index_bytes = sum(g[k].numel()*g[k].element_size() for g in self.groups for k in ('rows', 'cols'))
        element_bytes = self.groups[0]['factors'][0].element_size()
        low = [r for r in self.records if r['kind'] == 'low_rank']
        ideal = sum(r['dimension']**2 if r['kind'] == 'dense' else 2*r['dimension']*r['rank'] for r in self.records)
        return dict(blocks=len(self.records), groups=len(self.groups),
            low_rank_blocks=len(low), dense_blocks=len(self.records)-len(low),
            mean_far_rank=sum(r['rank'] for r in low)/max(1, len(low)),
            max_far_rank=max((r['rank'] for r in low), default=0),
            coefficient_scalars=coefficients, coefficient_bytes=coefficient_bytes,
            index_bytes=index_bytes, stored_bytes=coefficient_bytes+index_bytes,
            stored_scalar_fraction=coefficients/self.dimension**2,
            ideal_unpadded_fraction=ideal/self.dimension**2,
            total_byte_fraction=(coefficient_bytes+index_bytes)/(self.dimension**2*element_bytes),
            complex_multiply_adds=coefficients)

    def to(self, *, dtype):
        return HierarchicalResponse(self.dimension,
            [dict(kind=g['kind'], rows=g['rows'], cols=g['cols'],
                  factors=tuple(f.to(dtype=dtype) for f in g['factors'])) for g in self.groups],
            self.records)

    def dense(self, chunk=32):
        """Diagnostic only. The online solve must call the factored action."""
        first = self.groups[0]['factors'][0]
        columns = []
        for start in range(0, self.dimension, chunk):
            end = min(start+chunk, self.dimension)
            probes = first.new_zeros(end-start, self.dimension)
            probes[torch.arange(end-start, device=first.device), torch.arange(start, end, device=first.device)] = 1
            columns.append(self(probes).T)
        return torch.cat(columns, 1)


class HierarchicalParityPreconditioner:
    def __init__(self, parity, response):
        self.parity, self.response = parity, response

    def __call__(self, rhs):
        self.parity.check_current()
        reduced = self.parity.reduced_rhs(rhs)
        dtype = self.response.groups[0]['factors'][0].dtype
        even = self.response(reduced.flatten(1).to(dtype)).to(reduced.dtype).reshape_as(reduced)
        return self.parity.recover(even, rhs)
