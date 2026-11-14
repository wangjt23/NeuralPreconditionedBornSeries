"""CPU parity and stored-physics checks for the conservative TSA identity.

Run directly with Python or unittest discovery. The local read-only repository
snapshot is used when present; on FWI, the normal repository PYTHONPATH is used.
No solver, dataset generator, checkpoint, or training import is required.
"""

from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "remote_reference"
if SNAPSHOT.is_dir():
    sys.path.insert(0, str(SNAPSHOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from algebraic_tsa import AlgebraicTSA
from rte.models import PeriodicRTE2D
from rte.transport_baselines import ClassicalTransportStack


def make_operator(dtype, heterogeneous=True):
    """Small full-angle material with positive absorption and all Fourier modes."""
    n, angles = 8, 6
    op = PeriodicRTE2D(shape=(n, n), dx=1.0 / n, n_angles=angles,
                       anisotropy_g=.65, eta_scale=0., eta_floor=0.,
                       fdtype=dtype, device="cpu")
    x = torch.arange(n, dtype=dtype).reshape(n, 1) / n
    y = torch.arange(n, dtype=dtype).reshape(1, n) / n
    if heterogeneous:
        sigma_t = 2.2 + .35 * torch.cos(2 * torch.pi * x) + .2 * torch.sin(4 * torch.pi * y)
        sigma_s = sigma_t * (.45 + .08 * torch.cos(2 * torch.pi * (x + y)))
    else:
        sigma_t = torch.full((n, n), 2.2, dtype=dtype)
        sigma_s = torch.full_like(sigma_t, 1.1)
    op.setup(sigma_t, sigma_s)
    return op


def tolerance(dtype):
    return dict(atol=2e-12, rtol=2e-11) if dtype == torch.float64 else dict(atol=2e-5, rtol=2e-4)


class AlgebraicTSATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(6101)

    def test_field_parity_across_dtype_shift_depth_and_material(self):
        for dtype in (torch.float64, torch.float32):
            for eta in (0., .15):
                for depth in (1, 16):
                    for heterogeneous in (False, True):
                        for relaxation in (1., .8):
                            with self.subTest(dtype=dtype, eta=eta, depth=depth,
                                              heterogeneous=heterogeneous, relaxation=relaxation):
                                op = make_operator(dtype, heterogeneous)
                                original = ClassicalTransportStack(op, depth=depth, eta_scale=eta,
                                                                   relaxation=relaxation)
                                optimized = AlgebraicTSA(op, depth=depth, eta_scale=eta,
                                                         relaxation=relaxation)
                                rhs = torch.randn_like(op.source)
                                torch.testing.assert_close(optimized(rhs), original(rhs), **tolerance(dtype))

    def test_eliminated_defect_matches_physical_defect(self):
        for dtype in (torch.float64, torch.float32):
            for eta in (0., .15):
                for heterogeneous in (False, True):
                    with self.subTest(dtype=dtype, eta=eta, heterogeneous=heterogeneous):
                        op = make_operator(dtype, heterogeneous)
                        model = AlgebraicTSA(op, eta_scale=eta)
                        # Random complex fields cover the stored even-grid Nyquist modes.
                        rhs = torch.randn_like(op.source)
                        z = model.reference.G(rhs)
                        actual = rhs - op.A_op(z)
                        eliminated = (model.reference.sigma0 - op.sigma_t) * z + op.scattering_op(z)
                        error = torch.linalg.vector_norm(actual - eliminated) / torch.linalg.vector_norm(rhs)
                        self.assertLess(float(error), 2e-14 if dtype == torch.float64 else 8e-7)
                        expected = z + model.reference.reference_transport_solve(actual)
                        torch.testing.assert_close(model.single(rhs), expected, **tolerance(dtype))

    def test_phase_additivity_and_zero(self):
        for dtype in (torch.float64, torch.float32):
            for depth in (1, 16):
                with self.subTest(dtype=dtype, depth=depth):
                    op = make_operator(dtype)
                    model = AlgebraicTSA(op, depth=depth, eta_scale=.15)
                    r, s = torch.randn_like(op.source), torch.randn_like(op.source)
                    zr, zs = model(r), model(s)
                    for phase in (1j, -1., -1j, .2 + .3j):
                        torch.testing.assert_close(model(phase * r), phase * zr, **tolerance(dtype))
                    torch.testing.assert_close(model((.2 + .3j) * r + s), (.2 + .3j) * zr + zs,
                                               **tolerance(dtype))
                    torch.testing.assert_close(model(torch.zeros_like(r)), torch.zeros_like(r), atol=0., rtol=0.)

    def test_spatial_fft_pair_counts(self):
        op = make_operator(torch.float64)
        rhs = torch.randn_like(op.source)
        fftn, ifftn = torch.fft.fftn, torch.fft.ifftn
        for depth, original_count, optimized_count in ((1, 6, 4), (16, 126, 94)):
            for implementation, expected in ((ClassicalTransportStack, original_count),
                                              (AlgebraicTSA, optimized_count)):
                with self.subTest(depth=depth, implementation=implementation.__name__):
                    model = implementation(op, depth=depth)
                    # Preparation is deliberately outside these application counts.
                    model.prepare()
                    with mock.patch.object(torch.fft, "fftn", wraps=fftn) as forward_fft:
                        with mock.patch.object(torch.fft, "ifftn", wraps=ifftn) as inverse_fft:
                            model(rhs)
                    self.assertEqual(forward_fft.call_count, expected // 2)
                    self.assertEqual(inverse_fft.call_count, expected // 2)

    def test_guard_rejects_changed_stored_stream_phase_and_weights(self):
        for owner in ("physical", "reference"):
            for name in ("directions", "kx", "ky", "phase_matrix", "weights"):
                with self.subTest(owner=owner, tensor=name):
                    op = make_operator(torch.float64)
                    model = AlgebraicTSA(op)
                    model(op.source)  # Establish a valid guard-cache entry first.
                    target = op if owner == "physical" else model.reference
                    with torch.no_grad():
                        value = getattr(target, name)
                        value.reshape(-1)[1].add_(.01)
                    message = "streaming symbols" if name in ("directions", "kx", "ky") else "phase or quadrature"
                    with self.assertRaisesRegex(RuntimeError, message):
                        model(op.source)
        with self.assertRaisesRegex(ValueError, "only kind"):
            AlgebraicTSA(make_operator(torch.float64), kind="reference")

    def test_guard_cache_reuse_invalidation_and_material_refresh(self):
        op = make_operator(torch.float64)
        model = AlgebraicTSA(op, eta_scale=.15)
        rhs = torch.randn_like(op.source)
        equal = torch.equal
        with mock.patch.object(torch, "equal", wraps=equal) as comparisons:
            original = model(rhs)
            initial_calls = comparisons.call_count
            self.assertGreater(initial_calls, 0)
            first_signature = model._stream_guard_signature
            torch.testing.assert_close(model(rhs), original, atol=0., rtol=0.)
            self.assertEqual(comparisons.call_count, initial_calls)
            # A no-op mutation changes the tensor version while keeping physics equal.
            with torch.no_grad():
                op.kx.add_(0.)
            model(rhs)
            self.assertGreater(comparisons.call_count, initial_calls)
            self.assertNotEqual(model._stream_guard_signature, first_signature)
            after_geometry_calls = comparisons.call_count
            with torch.no_grad():
                op.sigma_t.add_(.1)
            refreshed = model(rhs)
            self.assertGreater(comparisons.call_count, after_geometry_calls)
            torch.testing.assert_close(model.reference.sigma_ref, op.sigma_t.mean((-2, -1), keepdim=True))
            self.assertFalse(torch.allclose(refreshed, original))
            control = ClassicalTransportStack(op, eta_scale=.15)
            torch.testing.assert_close(refreshed, control(rhs), **tolerance(torch.float64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
