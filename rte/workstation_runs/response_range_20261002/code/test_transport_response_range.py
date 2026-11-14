"""Small CPU checks against the unchanged full physical periodic operator.

Fixtures are deterministic N8/N16 unit examples, not development/test source
families. A local remote_reference archive is read-only; staged tests import
the actual repository's rte modules through PYTHONPATH/cwd instead.
"""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "remote_reference"
if not (SOURCE_ROOT / "rte/models.py").is_file():
    SOURCE_ROOT = Path.cwd() if (Path.cwd() / "rte/models.py").is_file() else ROOT
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
from transport_response_range import TransportResponseRange


class TransportResponseRangeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def fixture(self, size=8, batch=1, promoted=True, homogeneous=False, perturbed=False):
        dtype = torch.float32 if promoted else torch.float64
        low = PeriodicRTE2D(shape=(size, size), dx=1./size, n_angles=12,
                            anisotropy_g=.65, eta_scale=0., eta_floor=0.,
                            fdtype=dtype, device="cpu")
        axis = torch.arange(size, dtype=dtype) * (2 * torch.pi / size)
        x, y = axis[:, None], axis[None, :]
        t = (3. + .22*x.cos() + .13*y.sin()).expand(size, size)
        s = (1. + .11*x.sin() + .07*y.cos()).expand(size, size)
        if homogeneous:
            t, s = torch.full_like(t, 3.), torch.full_like(s, 1.)
        t = torch.stack([t + .3*i for i in range(batch)])[:, None]
        s = torch.stack([s + .05*i for i in range(batch)])[:, None]
        low.setup(t, s, torch.zeros(batch, 12, size, size,
                                   dtype=torch.complex64 if promoted else torch.complex128))
        high = high_precision_copy(low) if promoted else low
        if perturbed:
            # Deliberately destroy ideal circulant/symmetric-quadrature shortcuts.
            high.directions = high.directions.clone()
            high.directions[2, 0] += .037
            high.directions[7, 1] -= .019
            high.weights = torch.linspace(.055, .105, 12, dtype=torch.float64)
            high.weights = high.weights / high.weights.sum()
            high.phase_matrix = high.phase_matrix.clone()
            high.phase_matrix[1, 4] += .13
            high.phase_matrix[5, 2] -= .04
            high.kx = high.kx.clone()*1.013
            high.ky = high.ky.clone()*.987
        return high

    @staticmethod
    def field(high, channels=3, seed=351, batch=None):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(len(high.sigma_t) if batch is None else batch,
                           channels, *high.shape, dtype=torch.complex128, generator=generator)

    def assert_close(self, actual, expected, atol=3e-12, rtol=3e-12):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def assert_adjoint(self, forward, backward, left, right):
        lhs = (left.conj()*forward(right)).sum()
        rhs = (backward(left).conj()*right).sum()
        normalized = (lhs-rhs).abs()/(left.norm()*right.norm()).clamp_min(1e-30)
        self.assertLess(float(normalized), 3e-12)

    @staticmethod
    def explicit_h0(high, response):
        # Independent angular-entry assembly retains stored Nyquist values.
        stream = high._stream_symbol(len(high.sigma_t), torch.complex128, "cpu")
        matrix = torch.empty(len(high.sigma_t), *high.shape, 12, 12, dtype=torch.complex128)
        t = high.sigma_t.mean((-2, -1))[:, 0, None, None]
        s = high.sigma_s.mean((-2, -1))[:, 0, None, None]
        for row in range(12):
            for col in range(12):
                matrix[..., row, col] = -s*high.phase_matrix[row, col]*high.weights[col]
                if row == col:
                    matrix[..., row, col] += stream[:, row]+t
        return matrix

    def test_homogeneous_full_physical_image_is_q_on_both_stored_precisions(self):
        for size, promoted in ((8, True), (16, False)):
            with self.subTest(size=size, promoted=promoted):
                high = self.fixture(size=size, promoted=promoted, homogeneous=True)
                response = TransportResponseRange(high)
                c = self.field(high)
                q = response.angular_lift(c)
                self.assert_close(response.action(c), q, atol=2e-11)
                self.assert_close(response.homogeneous_action(response.lift(c)), q, atol=2e-11)
                self.assert_close(response.oblique_action(c),
                                  torch.einsum("ij,bjhw->bihw", response.gram, c), atol=2e-11)
                self.assert_close(response.gram, torch.eye(3, dtype=torch.complex128))
                self.assertIsNone(high._reference_transport_inverse)

    def test_actual_h0_uses_full_angular_phase_weights_geometry_and_nyquist(self):
        high = self.fixture(size=8, batch=2, perturbed=True)
        response = TransportResponseRange(high)
        matrix = self.explicit_h0(high, response)
        self.assert_close(response.homogeneous_matrix(), matrix)
        q = response.basis.reshape(1, 1, 1, 12, 3).expand(2, 8, 8, 12, 3)
        expected = torch.linalg.solve(matrix, q)
        self.assert_close(response.response_symbol, expected)
        self.assert_close(torch.matmul(matrix, response.response_symbol), q, atol=2e-12)
        stripe = (-1.)**torch.arange(8, dtype=torch.float64)
        c = torch.zeros(2, 3, 8, 8, dtype=torch.complex128)
        c[:, 1] = stripe[None, None, :]
        chat = torch.fft.fftn(c, dim=(-2, -1))
        full_hat = torch.einsum("bhwlm,bmhw->blhw", expected, chat)
        self.assert_close(response.lift(c), torch.fft.ifftn(full_hat, dim=(-2, -1)))
        # Full transport coupling must generate angles outside the input Q span.
        full = response.lift(self.field(high))
        complement = full-response.angular_lift(response.restrict(full))
        self.assertGreater(float(complement.norm()/full.norm()), .05)
        self.assertEqual(response.response_symbol.numel()*response.response_symbol.element_size(), 2*8*8*12*3*16)
        self.assertNotIn("reference_inverse", response.__dict__)
        self.assertIsNone(high._reference_transport_inverse)

    def test_heterogeneous_physical_images_and_all_adjoint_identities(self):
        high = self.fixture(size=16, promoted=True, perturbed=True)
        response = TransportResponseRange(high)
        c, d, full = self.field(high), self.field(high, seed=352), self.field(high, 12, 353)
        self.assert_close(response.action(c), high.A_op(response.lift(c)), atol=0, rtol=0)
        self.assert_close(response.oblique_action(c), response.restrict(high.A_op(response.lift(c))))
        self.assert_close(response.galerkin_action(c), response.adjoint(high.A_op(response.lift(c))))
        self.assert_adjoint(response.lift, response.adjoint, full, c)
        self.assert_adjoint(high.A_op, response.physical_adjoint, full, self.field(high, 12, 354))
        self.assert_adjoint(response.action, response.adjoint_action, full, c)
        self.assert_adjoint(response.oblique_action, response.oblique_adjoint, d, c)
        self.assert_adjoint(response.angular_lift, response.angular_restrict, full, c)

    def test_legacy_harmonic_input_reference_is_same_mean_response(self):
        high = self.fixture(size=8, promoted=True)
        response = TransportResponseRange(high)
        full = self.field(high, 12, 355)
        legacy = PeriodicRTE2D(shape=high.shape, dx=high.dx, n_angles=12,
                               anisotropy_g=high.anisotropy_g, eta_scale=0., eta_floor=0.,
                               fdtype=torch.float64, device="cpu")
        for name in ("directions", "phase_matrix", "weights", "kx", "ky"):
            setattr(legacy, name, getattr(high, name).clone())
        legacy.setup(high.sigma_t, high.sigma_s)
        original = legacy.reference_transport_solve(full, harmonics=1)
        self.assert_close(response.reference_response(full), original, atol=2e-12)
        c = response.restrict(full)
        self.assert_close(response.solve_qmean(c), response.lift(c), atol=0, rtol=0)
        # The range is not the closed three-moment Galerkin inverse.
        closed = legacy.reference_moment_transport_solve(full, harmonics=1)
        self.assertGreater(float((original-closed).norm()/original.norm()), .05)

    def test_zero_complex_linearity_and_source_independence(self):
        high = self.fixture(size=8, promoted=False)
        response = TransportResponseRange(high)
        c, d = self.field(high), self.field(high, seed=356)
        full, other = self.field(high, 12, 357), self.field(high, 12, 358)
        alpha, beta = .31-.57j, -.22+.43j
        for action, a, b in ((response.lift, c, d), (response.action, c, d),
                              (response.oblique_action, c, d),
                              (response.adjoint, full, other),
                              (response.adjoint_action, full, other)):
            with self.subTest(action=action.__name__):
                self.assert_close(action(alpha*a+beta*b), alpha*action(a)+beta*action(b))
                self.assert_close(action(torch.zeros_like(a)), torch.zeros_like(action(a)), atol=0, rtol=0)
        saved = response.lift(c)
        high.source = torch.ones_like(high.source)*(2.+3j)
        self.assert_close(response.lift(c), saved, atol=0, rtol=0)

    def test_material_batch_broadcast_and_individual_reference_agree(self):
        high = self.fixture(size=8, batch=1)
        response = TransportResponseRange(high)
        c, v = self.field(high, batch=3), self.field(high, 12, 359, batch=3)
        for action, field in ((response.lift, c), (response.action, c),
                              (response.adjoint, v), (response.adjoint_action, v)):
            with self.subTest(action=action.__name__):
                individual = torch.cat([action(field[i:i+1]) for i in range(3)])
                self.assert_close(action(field), individual)
        high2 = self.fixture(size=8, batch=2, promoted=False)
        response2 = TransportResponseRange(high2)
        c2 = self.field(high2)
        matrix = self.explicit_h0(high2, response2)
        self.assert_close(torch.matmul(matrix, response2.response_symbol),
                          response2.basis.reshape(1, 1, 1, 12, 3).expand(2, 8, 8, 12, 3))
        self.assertEqual(response2.lift(c2).shape, (2, 12, 8, 8))
        for wrong in (c2[:1], c2.repeat(2, 1, 1, 1)):
            with self.assertRaises(ValueError):
                response2.lift(wrong)

    def test_image_norm_bound_is_positive_conservative_and_not_oblique_claim(self):
        high = self.fixture(size=8, promoted=False)
        response = TransportResponseRange(high)
        c = self.field(high)
        lower = response.image_norm_lower_bound()
        self.assertEqual(lower.shape, (1,))
        self.assertGreater(float(lower[0]), 0.)
        self.assertGreaterEqual(float(response.action(c).norm()/c.norm()), float(lower[0])*(1-1e-12))
        largest = torch.linalg.matrix_norm(response.homogeneous_matrix(), ord=2).flatten(1).amax(1)
        self.assertTrue(bool((largest <= response.homogeneous_norm_upper_bound()*(1+1e-12)).all()))
        # The bound belongs to ||HU c||; no oblique positive-margin API is given.
        self.assertFalse(hasattr(response, "oblique_coercivity_margin"))

    def test_frozen_cache_identity_layout_versions_and_setup_are_guarded(self):
        high = self.fixture(size=8)
        response = TransportResponseRange(high)
        c = self.field(high)
        expected = response.lift(c)
        cases = [(high, "sigma_t", high.sigma_t.transpose(-1, -2)),
                 (high, "phase_matrix", high.phase_matrix.T),
                 (high, "directions", high.directions.clone()),
                 (response, "response_symbol", response.response_symbol.transpose(1, 2)),
                 (response, "basis", response.basis.view_as(response.basis)),
                 (response, "mean_t", response.mean_t.clone()),
                 (response, "physical", copy.copy(high)),
                 (response, "high", copy.copy(high)),
                 (high, "setup_generation", high.setup_generation+1)]
        for owner, name, replacement in cases:
            with self.subTest(field=name):
                saved = getattr(owner, name)
                setattr(owner, name, replacement)
                try:
                    with self.assertRaises(RuntimeError):response.prepare()
                    with self.assertRaises(RuntimeError):response.lift(c)
                finally:
                    setattr(owner, name, saved)
                self.assert_close(response.lift(c), expected, atol=0, rtol=0)
        with patch.object(high, "_build_reference_transport_inverse", side_effect=AssertionError("built unused full inverse")):
            response.prepare()
            self.assertIsNone(high._reference_transport_inverse)
        self.assertIs(response._captured_caches[-1], response.response_symbol)
        high.weights.add_(.001)
        with self.assertRaises(RuntimeError):response.lift(c)
        reconstructed = TransportResponseRange(high)
        self.assertTrue(bool(torch.isfinite(reconstructed.lift(c)).all()))

    def test_invalid_precision_fields_and_modes_are_rejected(self):
        high = self.fixture(size=8)
        response = TransportResponseRange(high)
        c = self.field(high)
        for wrong in (c.to(torch.complex64), c.real, c[..., :-1], c[:, :2],
                      torch.empty_like(c, device="meta"), c[:0]):
            with self.subTest(dtype=wrong.dtype, shape=tuple(wrong.shape), device=str(wrong.device)):
                with self.assertRaises(ValueError):response.lift(wrong)
        for modes in ((), (0, 12), (0, 1.5), (False, 1)):
            with self.assertRaises(ValueError):TransportResponseRange(high, modes=modes)
        low = PeriodicRTE2D(shape=(8, 8), n_angles=12, fdtype=torch.float32, device="cpu")
        low.setup(torch.ones(1, 1, 8, 8)*3., torch.ones(1, 1, 8, 8))
        with self.assertRaises(ValueError):TransportResponseRange(low)


if __name__ == "__main__":
    unittest.main(verbosity=2)
