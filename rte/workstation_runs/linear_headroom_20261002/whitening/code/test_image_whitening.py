"""CPU checks of the full homogeneous image whitening of angular fields.

These deterministic N8/N16 fixtures reuse the physical-operator fixtures from
test_angular_galerkin. They do not use development media, held-out seeds,
training, GPUs, or timing experiments. In particular, whitening uses H0 Q's
full-angle image, not the projected square matrix Q^H H0 Q.
"""

import unittest

import torch

import test_angular_galerkin as galerkin_tests
from angular_galerkin import AngularGalerkin
from image_whitening import ImageWhitening


class ImageWhiteningTests(unittest.TestCase):
    # Reuse fixtures without inheriting and rerunning the Galerkin test suite.
    fixture = galerkin_tests.AngularGalerkinTests.fixture
    cases = galerkin_tests.AngularGalerkinTests.cases
    vector = galerkin_tests.AngularGalerkinTests.vector
    assert_close = galerkin_tests.AngularGalerkinTests.assert_close
    assert_adjoint_pair = galerkin_tests.AngularGalerkinTests.assert_adjoint_pair
    physical_adjoint = staticmethod(galerkin_tests.AngularGalerkinTests.physical_adjoint)

    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def independent_image_symbol(physical, coarse):
        """Assemble H0 Q from the actual stored physical H buffers."""
        q = coarse.basis
        stream = physical._stream_symbol(
            len(physical.sigma_t), torch.complex128, physical.sigma_t.device
        ).permute(0, 2, 3, 1)[..., None]
        mean_t = physical.sigma_t.mean((-2, -1))[:, 0, None, None, None, None]
        mean_s = physical.sigma_s.mean((-2, -1))[:, 0, None, None, None, None]
        phase = physical.phase_matrix.to(torch.complex128)
        kq = phase @ (physical.weights[:, None] * q)
        return (stream + mean_t) * q - mean_s * kq

    @staticmethod
    def spectral_action(symbol, value):
        spectrum = torch.fft.fftn(value, dim=(-2, -1)).permute(0, 2, 3, 1)
        image = torch.einsum("bxyla,bxya->bxyl", symbol, spectrum)
        return torch.fft.ifftn(image.permute(0, 3, 1, 2), dim=(-2, -1))

    def assert_whitening_identity(self, physical, coarse, whitening):
        image = self.independent_image_symbol(physical, coarse)
        self.assert_close(whitening.full_reference_symbol, image)
        symbol = whitening.whitening_symbol
        normal = image.conj().transpose(-2, -1) @ image
        self.assert_close(symbol, symbol.conj().transpose(-2, -1))
        self.assert_close(whitening.matrix, symbol)
        self.assertTrue(bool((whitening.eigenvalues > 0).all()))
        self.assertAlmostEqual(float(whitening.minimum_reference_eigenvalue),
                               float(whitening.eigenvalues.min()), places=12)
        self.assert_close(whitening.eigenvalues, torch.linalg.eigvalsh(normal),
                          atol=3e-11, rtol=3e-12)
        eye = torch.eye(coarse.n_modes, dtype=torch.complex128).expand_as(symbol)
        self.assert_close(symbol.conj().transpose(-2, -1) @ normal @ symbol,
                          eye, atol=5e-12, rtol=5e-12)
        return image

    def test_symbol_is_the_full_original_homogeneous_image_and_whitens_it(self):
        for label, physical in self.cases():
            with self.subTest(**label):
                coarse = AngularGalerkin(physical)
                whitening = ImageWhitening(coarse)
                whitening.prepare()
                self.assertIs(whitening.coarse, coarse)
                self.assertEqual(whitening.full_reference_symbol.shape,
                                 (label["batch"], label["size"], label["size"], 12, 3))
                self.assertEqual(whitening.whitening_symbol.shape,
                                 (label["batch"], label["size"], label["size"], 3, 3))
                self.assertEqual(whitening.eigenvalues.shape,
                                 (label["batch"], label["size"], label["size"], 3))
                self.assertEqual(whitening.full_reference_symbol.dtype, torch.complex128)
                self.assertEqual(whitening.whitening_symbol.dtype, torch.complex128)
                self.assertEqual(whitening.eigenvalues.dtype, torch.float64)
                self.assert_whitening_identity(physical, coarse, whitening)

    def test_homogeneous_image_equals_direct_physical_action_including_nyquist(self):
        for label, physical in self.cases():
            if not label["homogeneous"]:
                continue
            with self.subTest(**label):
                coarse = AngularGalerkin(physical)
                whitening = ImageWhitening(coarse)
                q = self.vector(physical)
                image = self.independent_image_symbol(physical, coarse)
                # Random fields contain the original even-grid Nyquist modes.
                self.assert_close(self.spectral_action(image, q),
                                  physical.A_op(coarse.lift(q)))
                wq = whitening(q)
                expected = self.spectral_action(image, wq)
                actual = coarse.full_action(wq)
                self.assert_close(actual, expected)
                self.assert_close(actual.abs().square().flatten(1).sum(1),
                                  q.abs().square().flatten(1).sum(1),
                                  atol=2e-10, rtol=5e-12)

    def test_action_adjoint_and_actual_hqw_adjoint_use_physical_stored_buffers(self):
        for label, physical in self.cases():
            with self.subTest(**label):
                coarse = AngularGalerkin(physical)
                whitening = ImageWhitening(coarse)
                q, p = self.vector(physical), self.vector(physical, seed=315)
                field = self.vector(physical, 12, 316)
                self.assert_close(whitening(q), whitening.action(q))
                self.assert_close(whitening.solve(q), whitening.action(q))
                self.assert_close(whitening.adjoint_solve(q), whitening.adjoint(q))
                self.assert_close(whitening.adjoint(q), whitening.action(q))
                self.assert_close(whitening.action(q),
                                  self.spectral_action(whitening.whitening_symbol, q))
                self.assert_adjoint_pair(whitening.action, whitening.adjoint, p, q)
                expected = whitening.adjoint(
                    coarse.restrict(self.physical_adjoint(physical, field)))
                actual = whitening.adjoint(coarse.full_adjoint(field))
                self.assert_close(actual, expected)
                self.assert_adjoint_pair(
                    lambda value: coarse.full_action(whitening(value)),
                    lambda value: whitening.adjoint(coarse.full_adjoint(value)),
                    field, q,
                )
                self.assertEqual(whitening(q).dtype, torch.complex128)
                self.assertEqual(whitening(q).shape, q.shape)

    def test_zero_additivity_and_complex_linearity(self):
        for label, physical in self.cases():
            with self.subTest(**label):
                whitening = ImageWhitening(AngularGalerkin(physical))
                q, p = self.vector(physical), self.vector(physical, seed=315)
                for name in ("action", "adjoint", "solve", "adjoint_solve"):
                    operation = getattr(whitening, name)
                    self.assert_close(operation(torch.zeros_like(q)), torch.zeros_like(q),
                                      atol=0, rtol=0)
                    self.assert_close(operation(q + p), operation(q) + operation(p))
                    self.assert_close(operation(1j * q), 1j * operation(q))

    def test_one_material_broadcasts_to_multiple_rhs_and_other_mode_counts(self):
        for promoted in (True, False):
            for homogeneous in (True, False):
                for modes in ((0,), (0, 1, -1), (0, 1, -1, 2, -2)):
                    with self.subTest(promoted=promoted, homogeneous=homogeneous, modes=modes):
                        physical = self.fixture(promoted=promoted, homogeneous=homogeneous)
                        coarse = AngularGalerkin(physical, modes=modes)
                        whitening = ImageWhitening(coarse)
                        self.assert_whitening_identity(physical, coarse, whitening)
                        generator = torch.Generator(device="cpu").manual_seed(317)
                        q = torch.randn(3, len(modes), 8, 8, generator=generator,
                                        dtype=torch.complex128)
                        actual = whitening(q)
                        expected = torch.cat([whitening(row[None]) for row in q], dim=0)
                        self.assert_close(actual, expected)
                        self.assert_close(actual, self.spectral_action(whitening.whitening_symbol, q))
                        field = torch.randn(3, 12, 8, 8, generator=generator,
                                            dtype=torch.complex128)
                        self.assert_adjoint_pair(
                            lambda value: coarse.full_action(whitening(value)),
                            lambda value: whitening.adjoint(coarse.full_adjoint(value)),
                            field, q,
                        )

    def test_invalid_precision_shape_and_material_batch_are_rejected(self):
        physical = self.fixture(batch=2)
        whitening = ImageWhitening(AngularGalerkin(physical))
        invalid = (
            torch.zeros(2, 3, 8, 8, dtype=torch.complex64),
            torch.zeros(2, 3, 8, 8, dtype=torch.float64),
            torch.zeros(2, 2, 8, 8, dtype=torch.complex128),
            torch.zeros(2, 3, 8, 7, dtype=torch.complex128),
            torch.zeros(3, 3, 8, 8, dtype=torch.complex128),
            torch.zeros(0, 3, 8, 8, dtype=torch.complex128),
            torch.zeros(3, 8, 8, dtype=torch.complex128),
        )
        for operation in (whitening.action, whitening.adjoint):
            for value in invalid:
                with self.subTest(operation=operation.__name__, shape=value.shape, dtype=value.dtype):
                    with self.assertRaises(ValueError):
                        operation(value)

    def test_cache_refreshes_for_actual_physical_and_angular_buffer_changes(self):
        physical = self.fixture(batch=2)
        coarse = AngularGalerkin(physical)
        whitening = ImageWhitening(coarse)
        whitening.prepare()
        pointer = whitening.whitening_symbol.data_ptr()
        whitening.prepare()
        self.assertEqual(whitening.whitening_symbol.data_ptr(), pointer)
        q, field = self.vector(physical), self.vector(physical, 12, 316)

        def mutate_t_in_place():
            physical.sigma_t.add_(.125)

        def replace_s_pointer():
            physical.sigma_s = physical.sigma_s.clone() * .97

        def mutate_direction_version():
            physical.directions[0, 0].mul_(.93)

        def replace_kx_pointer():
            physical.kx = physical.kx.clone() * 1.07

        def mutate_ky_version():
            physical.ky.mul_(.98)

        def mutate_phase_version():
            physical.phase_matrix[0, 1].add_(.11)

        def replace_weights_pointer():
            physical.weights = physical.weights.clone()
            physical.weights[2].mul_(1.03)

        def mutate_basis_version():
            coarse.basis[0, 0].mul_(.98 + .01j)

        def advance_setup_generation():
            physical.setup_generation += 1

        for mutation in (
            mutate_t_in_place, replace_s_pointer, mutate_direction_version,
            replace_kx_pointer, mutate_ky_version, mutate_phase_version,
            replace_weights_pointer, mutate_basis_version, advance_setup_generation,
        ):
            with self.subTest(mutation=mutation.__name__):
                prior_signature = coarse.cached_signature
                # Retain the old symbol so allocator reuse cannot mimic no refresh.
                prior_symbol = whitening.whitening_symbol
                mutation()
                whitening.prepare()
                self.assertNotEqual(coarse.cached_signature, prior_signature)
                self.assertNotEqual(whitening.whitening_symbol.data_ptr(), prior_symbol.data_ptr())
                self.assert_whitening_identity(physical, coarse, whitening)
                self.assert_adjoint_pair(
                    lambda value: coarse.full_action(whitening(value)),
                    lambda value: whitening.adjoint(coarse.full_adjoint(value)),
                    field, q,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
