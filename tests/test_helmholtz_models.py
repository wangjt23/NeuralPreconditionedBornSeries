"""Regressions for cached implicit kernels and odd-grid multigrid transfers."""

import unittest

import torch

from helmholtz.models.model2d import ES_ImplicitStep, MGNO_precond
from helmholtz.models.model3d import MGNO_precond_3d
from helmholtz.models.cbs import ConvergentBornSeries2d, ConvergentBornSeries3d


class HelmholtzModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_implicit_cache_updates_after_weights_change(self):
        layer = ES_ImplicitStep(1e-5, 4).eval()
        rhs = torch.randn(2, 4, 8, 8)
        with torch.no_grad():
            layer(rhs)
            layer.weights.add_(0.01)
            updated = layer(rhs)
            layer._green = None
            reference = layer(rhs)
        torch.testing.assert_close(updated, reference)

    def test_cbs_defaults_apply_absorption_and_keep_updates_finite(self):
        for dimensions, constructor in [(2, ConvergentBornSeries2d), (3, ConvergentBornSeries3d)]:
            with self.subTest(dimensions=dimensions):
                model = constructor(N=[8] * dimensions, boundary_width=[2] * dimensions)
                shape = (1, 1) + (8,) * dimensions
                model.setup(torch.full(shape, 2500.0), torch.ones(shape))
                self.assertTrue(torch.is_complex(model.inv_sos2))
                self.assertGreater(model.inv_sos2.imag.abs().max().item(), 0.0)
                update = model(torch.zeros_like(model.b, dtype=model.complex_dtype))
                self.assertTrue(torch.isfinite(update).all())

    def test_implicit_cache_tracks_both_spatial_dimensions(self):
        layer = ES_ImplicitStep(1e-5, 4).eval()
        with torch.no_grad():
            layer(torch.randn(2, 4, 8, 8))
            rhs = torch.randn(2, 4, 10, 8)
            updated = layer(rhs)
            layer._green = None
            reference = layer(rhs)
        torch.testing.assert_close(updated, reference)

    def test_implicit_eval_allows_independent_backward_passes(self):
        layer = ES_ImplicitStep(1e-5, 4).eval()
        for _ in range(2):
            layer.zero_grad(set_to_none=True)
            layer(torch.randn(2, 4, 8, 8)).square().mean().backward()
            self.assertTrue(torch.isfinite(layer.weights.grad).all())

    def test_mgno_odd_grids_preserve_shape_and_backpropagate(self):
        for dimensions, constructor in [(2, MGNO_precond), (3, MGNO_precond_3d)]:
            with self.subTest(dimensions=dimensions):
                model = constructor(num_channel_u=2)
                shape = (1, 2) + (17,) * dimensions
                rhs = torch.randn(shape, requires_grad=True)
                a_list, s_list = model.setup(torch.randn(shape))
                out = model(rhs, a_list, s_list)
                self.assertEqual(out.shape, rhs.shape)
                out.square().mean().backward()
                self.assertTrue(torch.isfinite(rhs.grad).all())
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))


if __name__ == "__main__":
    unittest.main()
