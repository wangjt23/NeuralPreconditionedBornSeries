"""Small CPU checks against the unchanged physical periodic RTE operator.

The local archive is only read. When copied into the full project, imports fall
back to that repository's rte package. These deterministic N8/N16 fixtures are
unit examples, not development/test source families or a timing experiment.
"""

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "remote_reference"
if not (SOURCE_ROOT / "rte" / "models.py").is_file():
    SOURCE_ROOT = Path.cwd() if (Path.cwd() / "rte" / "models.py").is_file() else ROOT
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
from angular_galerkin import AngularGalerkin


class AngularGalerkinTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def fixture(self, size=8, batch=1, promoted=True, homogeneous=False):
        dtype = torch.float32 if promoted else torch.float64
        operator = PeriodicRTE2D(
            shape=(size, size), dx=1. / size, n_angles=12, anisotropy_g=.65,
            eta_scale=0., eta_floor=0., fdtype=dtype, device="cpu",
        )
        axis = torch.arange(size, dtype=dtype) * (2 * torch.pi / size)
        x, y = axis.reshape(size, 1), axis.reshape(1, size)
        t = (3. + .22 * x.cos() + .13 * y.sin()).expand(size, size)
        s = (1. + .11 * x.sin() + .07 * y.cos()).expand(size, size)
        if homogeneous:
            t, s = torch.full_like(t, 3.), torch.full_like(s, 1.)
        t = torch.stack([t + .3 * index for index in range(batch)])[:, None]
        s = torch.stack([s + .05 * index for index in range(batch)])[:, None]
        cdtype = torch.complex64 if promoted else torch.complex128
        source = torch.zeros(batch, 12, size, size, dtype=cdtype)
        operator.setup(t, s, source)
        physical = high_precision_copy(operator) if promoted else operator
        self.assertEqual(physical.float_dtype, torch.float64)
        self.assertEqual(physical.complex_dtype, torch.complex128)
        if promoted:
            for name in ("directions", "weights", "phase_matrix", "kx", "ky"):
                self.assertTrue(torch.equal(getattr(physical, name), getattr(operator, name).double()))
        return physical

    def cases(self):
        for size in (8, 16):
            for batch in (1, 2):
                for promoted in (True, False):
                    for homogeneous in (True, False):
                        label = dict(size=size, batch=batch, promoted=promoted, homogeneous=homogeneous)
                        yield label, self.fixture(**label)

    def vector(self, physical, channels=3, seed=313):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(
            len(physical.sigma_t), channels, *physical.shape,
            generator=generator, dtype=torch.complex128,
        )

    def assert_close(self, actual, expected, atol=2e-12, rtol=2e-12):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    @staticmethod
    def physical_adjoint(physical, field):
        """Independent stored-matrix adjoint, including its complex Nyquist mode."""
        symbol = physical._stream_symbol(len(field), field.dtype, field.device)
        streaming = torch.fft.ifftn(
            symbol.conj() * torch.fft.fftn(field, dim=(-2, -1)), dim=(-2, -1)
        )
        # K = phase @ diag(weights). Its adjoint weights the OUTPUT angle;
        # this deliberately does not assume a symmetric normalized phase.
        phase = physical.phase_matrix.to(torch.complex128)
        scattered = torch.einsum("lm,blxy->bmxy", phase.conj(), physical.sigma_s * field)
        scattered = physical.weights.reshape(1, -1, 1, 1) * scattered
        return streaming + physical.sigma_t * field - scattered

    def assert_adjoint_pair(self, forward, backward, left, right):
        first = (left.conj() * forward(right)).sum()
        second = (backward(left).conj() * right).sum()
        scale = left.norm() * right.norm()
        self.assertLess(float((first - second).abs() / scale.clamp_min(1e-30)), 2e-12)

    def reference_matrix(self, physical, galerkin):
        q = galerkin.basis
        stream = physical._stream_symbol(len(physical.sigma_t), torch.complex128, "cpu")
        projected_stream = torch.einsum("la,blxy,lc->bxyac", q.conj(), stream, q)
        mean_t = physical.sigma_t.mean((-2, -1))[:, 0, None, None, None, None]
        mean_s = physical.sigma_s.mean((-2, -1))[:, 0, None, None, None, None]
        return projected_stream + mean_t * galerkin.gram - mean_s * galerkin.kq

    def test_lift_restrict_and_small_matrices_use_actual_stored_geometry(self):
        for label, physical in self.cases():
            with self.subTest(**label):
                coarse = AngularGalerkin(physical, modes=(0, 1, -1))
                q, field = self.vector(physical), self.vector(physical, 12, 314)
                basis = coarse.basis
                self.assertEqual(basis.shape, (12, 3))
                self.assertEqual(basis.dtype, torch.complex128)
                lifted = torch.einsum("la,baxy->blxy", basis, q)
                restricted = torch.einsum("la,blxy->baxy", basis.conj(), field)
                self.assert_close(coarse.lift(q), lifted)
                self.assert_close(coarse.restrict(field), restricted)
                self.assert_close(coarse.gram, basis.conj().T @ basis)
                for name, component in (("mx", 0), ("my", 1)):
                    matrix = basis.conj().T @ (physical.directions[:, component, None] * basis)
                    self.assert_close(getattr(coarse, name), matrix)
                weighted_phase = physical.phase_matrix.to(torch.complex128) * physical.weights[None]
                self.assert_close(coarse.kq, basis.conj().T @ weighted_phase @ basis)
                self.assert_close(coarse.restrict(coarse.lift(q)), torch.einsum("ac,bcxy->baxy", coarse.gram, q))

    def test_actions_equal_exact_restriction_of_original_physical_action(self):
        for label, physical in self.cases():
            with self.subTest(**label):
                coarse = AngularGalerkin(physical)
                q = self.vector(physical)
                image = physical.A_op(coarse.lift(q))
                self.assert_close(coarse.full_action(q), image)
                self.assert_close(coarse.action(q), coarse.restrict(image))
                self.assert_close(coarse(q), coarse.action(q))
                self.assertEqual(coarse.action(q).dtype, torch.complex128)
                self.assertEqual(coarse.action(q).shape, q.shape)

    def test_adjoint_matches_original_physical_adjoint_and_inner_products(self):
        for label, physical in self.cases():
            with self.subTest(**label):
                coarse = AngularGalerkin(physical)
                q, p = self.vector(physical), self.vector(physical, seed=315)
                field = self.vector(physical, 12, 316)
                expected_full = coarse.restrict(self.physical_adjoint(physical, field))
                self.assert_close(coarse.full_adjoint(field), expected_full)
                expected_small = coarse.restrict(self.physical_adjoint(physical, coarse.lift(p)))
                self.assert_close(coarse.adjoint(p), expected_small)
                self.assert_adjoint_pair(coarse.action, coarse.adjoint, p, q)
                self.assert_adjoint_pair(coarse.full_action, coarse.full_adjoint, field, q)
                self.assert_adjoint_pair(coarse.lift, coarse.restrict, field, q)
                self.assertEqual(coarse.adjoint(p).dtype, torch.complex128)

    def test_zero_additivity_and_complex_phase_are_preserved(self):
        for label, physical in self.cases():
            with self.subTest(**label):
                coarse = AngularGalerkin(physical)
                q, p = self.vector(physical), self.vector(physical, seed=315)
                for name in ("lift", "action", "adjoint", "full_action", "reference_solve", "reference_adjoint_solve"):
                    operation = getattr(coarse, name)
                    with self.subTest(operation=name):
                        self.assert_close(operation(torch.zeros_like(q)), torch.zeros_like(operation(q)), atol=0, rtol=0)
                        self.assert_close(operation(q + p), operation(q) + operation(p))
                        self.assert_close(operation(1j * q), 1j * operation(q))
                        self.assertEqual(operation(q).dtype, torch.complex128)

    def test_reference_inverse_and_adjoint_use_exact_projected_homogeneous_matrix(self):
        for label, physical in self.cases():
            with self.subTest(**label):
                coarse = AngularGalerkin(physical)
                q, p = self.vector(physical), self.vector(physical, seed=315)
                inverse = coarse.reference_inverse
                self.assertEqual(inverse.shape, (label["batch"], label["size"], label["size"], 3, 3))
                self.assertEqual(inverse.dtype, torch.complex128)
                matrix = self.reference_matrix(physical, coarse)
                eye = torch.eye(3, dtype=torch.complex128).expand_as(inverse)
                self.assert_close(matrix @ inverse, eye)
                solution = coarse.reference_solve(q)
                spectrum = torch.fft.fftn(solution, dim=(-2, -1)).permute(0, 2, 3, 1)
                image_spectrum = torch.einsum("bxyac,bxyc->bxya", matrix, spectrum)
                image = torch.fft.ifftn(image_spectrum.permute(0, 3, 1, 2), dim=(-2, -1))
                self.assert_close(image, q)
                if label["homogeneous"]:
                    self.assert_close(coarse.action(solution), q)
                self.assert_adjoint_pair(coarse.reference_solve, coarse.reference_adjoint_solve, p, q)

    def test_coercivity_bound_is_conservative_per_batch_without_phase_symmetry_assumption(self):
        for label, physical in self.cases():
            with self.subTest(**label):
                coarse = AngularGalerkin(physical)
                gram = (coarse.gram + coarse.gram.conj().T) / 2
                scattering = (coarse.kq + coarse.kq.conj().T) / 2
                smallest = torch.linalg.eigvalsh(gram).min()
                largest = torch.linalg.eigvalsh(scattering).max()
                expected = (physical.sigma_t * smallest - physical.sigma_s * largest).flatten(1).min(1).values
                bound = coarse.coercivity_bound()
                self.assertEqual(bound.shape, (label["batch"],))
                self.assertEqual(bound.dtype, torch.float64)
                self.assert_close(bound, expected)
                self.assertTrue(bool((bound > 0).all()))
                q = self.vector(physical)
                energy = (q.conj() * coarse.action(q)).real.flatten(1).sum(1)
                norm_squared = q.abs().square().flatten(1).sum(1)
                self.assertTrue(bool((energy >= bound * norm_squared - 2e-12 * norm_squared).all()))

    def test_complex64_and_real_inputs_are_rejected_instead_of_silently_promoted(self):
        physical = self.fixture()
        coarse = AngularGalerkin(physical)
        for name in ("lift", "action", "adjoint", "full_action", "reference_solve", "reference_adjoint_solve"):
            for dtype in (torch.complex64, torch.float64):
                with self.subTest(operation=name, dtype=dtype):
                    with self.assertRaises(ValueError):
                        getattr(coarse, name)(torch.zeros(1, 3, 8, 8, dtype=dtype))
        for name in ("restrict", "full_adjoint"):
            for dtype in (torch.complex64, torch.float64):
                with self.subTest(operation=name, dtype=dtype):
                    with self.assertRaises(ValueError):
                        getattr(coarse, name)(torch.zeros(1, 12, 8, 8, dtype=dtype))
        low = PeriodicRTE2D(shape=(8, 8), n_angles=12, fdtype=torch.float32)
        low.setup(torch.full((1, 1, 8, 8), 3.), torch.ones(1, 1, 8, 8))
        with self.assertRaises(ValueError):
            AngularGalerkin(low)

    def test_cache_refreshes_after_pointer_version_and_setup_generation_changes(self):
        physical = self.fixture(batch=2)
        coarse = AngularGalerkin(physical)
        q, field = self.vector(physical), self.vector(physical, 12, 316)
        signature, inverse_pointer = coarse.cached_signature, coarse.reference_inverse.data_ptr()
        coarse.prepare()
        self.assertEqual(coarse.cached_signature, signature)
        self.assertEqual(coarse.reference_inverse.data_ptr(), inverse_pointer)

        def mutate_t_in_place():
            physical.sigma_t.add_(.125)

        def replace_s_pointer():
            physical.sigma_s = physical.sigma_s.clone() * .97

        def mutate_direction_version():
            physical.directions[0, 0].mul_(.93)

        def replace_kx_pointer():
            physical.kx = physical.kx.clone() * 1.07

        def mutate_phase_version():
            physical.phase_matrix[0, 1].add_(.11)

        def replace_weights_pointer():
            physical.weights = physical.weights.clone()
            physical.weights[2].mul_(1.03)

        def advance_setup_generation():
            physical.setup_generation += 1

        for mutation in (
            mutate_t_in_place, replace_s_pointer, mutate_direction_version,
            replace_kx_pointer, mutate_phase_version, replace_weights_pointer,
            advance_setup_generation,
        ):
            with self.subTest(mutation=mutation.__name__):
                prior = coarse.cached_signature
                mutation()
                coarse.prepare()
                self.assertNotEqual(coarse.cached_signature, prior)
                expected = coarse.restrict(physical.A_op(coarse.lift(q)))
                self.assert_close(coarse.action(q), expected)
                self.assert_close(coarse.full_adjoint(field), coarse.restrict(self.physical_adjoint(physical, field)))
                self.assert_close(self.reference_matrix(physical, coarse) @ coarse.reference_inverse,
                                  torch.eye(3, dtype=torch.complex128).expand_as(coarse.reference_inverse))


if __name__ == "__main__":
    unittest.main(verbosity=2)
