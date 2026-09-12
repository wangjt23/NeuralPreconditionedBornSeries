"""CPU regressions for angular channel layout and bounded mode mixing."""

import unittest
from unittest.mock import patch

import torch

from rte.npbs_unet_ablation import (
    AngularLowRankMixer,
    AngularScaleModeMixer,
    AngularSpectralDefect2Preconditioner,
    ModeGroupedPeriodicUNet,
    NPBSUNetPreconditioner,
)
from rte.run_hard_case_experiments import make_case


class AngularStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(7)

    def test_unet_isolates_each_mode_with_mixers_off(self):
        for n_modes, levels, height, width in ((1, 1, 6, 8), (3, 2, 8, 10), (4, 3, 9, 11)):
            with self.subTest(n_modes=n_modes, levels=levels, shape=(height, width)):
                model = ModeGroupedPeriodicUNet(
                    n_modes, mode_width=2, levels=levels, gate_coeff_channels=3,
                    mode_mix_scale=0.0, output_init="small",
                )
                with torch.no_grad():
                    for module in model.modules():
                        if isinstance(module, AngularScaleModeMixer):
                            module.static_raw.normal_()
                            module.coeff_net[-1].weight.normal_()
                            module.coeff_net[-1].bias.normal_()
                coeff = []
                h, w = height, width
                for _ in range(levels):
                    coeff.append(torch.randn(2, 3, h, w))
                    h, w = (h + 1) // 2, (w + 1) // 2
                for active_mode in range(n_modes):
                    with self.subTest(active_mode=active_mode), torch.no_grad():
                        x = torch.zeros(2, n_modes, 4, height, width)
                        x[:, active_mode] = torch.randn(2, 4, height, width)
                        out = model(x.flatten(1, 2), coeff).reshape(2, n_modes, 2, height, width)
                        self.assertGreater(float(out[:, active_mode].norm()), 1e-12)
                        inactive = [mode for mode in range(n_modes) if mode != active_mode]
                        torch.testing.assert_close(
                            out[:, inactive], torch.zeros_like(out[:, inactive]), rtol=0, atol=0,
                        )

    def test_low_rank_identity_initialization(self):
        mixer = AngularLowRankMixer(5, coeff_channels=3, rank=3)
        modes = torch.randn(2, 5, 4, 3, dtype=torch.complex64)
        out = mixer(modes, torch.randn(2, 3, 4, 3))
        torch.testing.assert_close(out, modes, rtol=0, atol=0)

    def test_low_rank_factors_have_independent_storage(self):
        mixer = AngularLowRankMixer(5, coeff_channels=3, rank=3)
        right = mixer.right.detach().clone()
        with torch.no_grad():
            mixer.left.add_(1.0)
        torch.testing.assert_close(mixer.right, right, rtol=0, atol=0)

    def test_low_rank_operator_bound_for_large_collinear_factors(self):
        n_modes = 5
        for rank in (1, 3, 5):
            for scale in (0.0, 0.25, -0.4):
                for dtype in (torch.complex64, torch.complex128):
                    with self.subTest(rank=rank, scale=scale, dtype=dtype):
                        mixer = AngularLowRankMixer(n_modes, coeff_channels=3, rank=rank, scale=scale)
                        with torch.no_grad():
                            mixer.left.fill_(1e4)
                            mixer.right.fill_(2e5)
                            mixer.coeff_net[-1].bias.fill_(20.0)
                        eye = torch.eye(n_modes, dtype=dtype)
                        modes = eye.view(1, n_modes, n_modes, 1).expand(2, -1, -1, -1)
                        out = mixer(modes, torch.randn(2, 3, 3, 4))
                        perturbation = out[..., 0] - eye
                        norms = torch.linalg.matrix_norm(perturbation, ord=2)
                        self.assertTrue(bool(torch.isfinite(norms).all()))
                        self.assertLessEqual(float(norms.detach().max()), abs(scale) + 1e-6)
                        torch.testing.assert_close(
                            norms, torch.full_like(norms, abs(scale)), rtol=1e-5, atol=1e-6,
                        )

    def test_low_rank_random_operator_bound_and_factor_rescaling(self):
        mixer = AngularLowRankMixer(6, coeff_channels=3, rank=3, scale=0.35)
        with torch.no_grad():
            mixer.left.normal_()
            mixer.right.normal_()
            mixer.coeff_net[-1].weight.normal_()
            mixer.coeff_net[-1].bias.normal_()
        eye = torch.eye(6, dtype=torch.complex64)
        modes = eye.view(1, 6, 6, 1).expand(2, -1, -1, -1)
        coeff = torch.randn(2, 3, 3, 4)
        with torch.no_grad():
            before = mixer(modes, coeff)
            mixer.left.mul_(1e6)
            mixer.right.mul_(1e4)
            after = mixer(modes, coeff)
        torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
        norms = torch.linalg.matrix_norm(after[..., 0] - eye, ord=2)
        self.assertLessEqual(float(norms.max()), mixer.scale + 1e-6)

    def test_low_rank_gradients_and_degenerate_factors_are_finite(self):
        for degenerate in (False, True):
            with self.subTest(degenerate=degenerate):
                mixer = AngularLowRankMixer(5, coeff_channels=3, rank=3)
                with torch.no_grad():
                    mixer.coeff_net[-1].bias.fill_(0.5)
                    if degenerate:
                        mixer.left[:, 0].zero_()
                        mixer.right[1].zero_()
                        mixer.left[:, 2].fill_(1e-15)
                modes = torch.randn(2, 5, 3, 4, dtype=torch.complex64, requires_grad=True)
                out = mixer(modes, torch.randn(2, 3, 3, 4))
                self.assertTrue(bool(torch.isfinite(out).all()))
                out.abs().square().mean().backward()
                self.assertTrue(bool(torch.isfinite(modes.grad).all()))
                for name, parameter in mixer.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                if not degenerate:
                    self.assertGreater(float(mixer.left.grad.norm()), 0.0)
                    self.assertGreater(float(mixer.right.grad.norm()), 0.0)
                    self.assertGreater(float(mixer.coeff_net[-1].bias.grad.norm()), 0.0)

    def test_angular_zero_init_recovers_defect2_and_trains(self):
        rte, _ = make_case("mixed", 8, 6, torch.float32, 0.2)
        model = AngularSpectralDefect2Preconditioner(
            rte, base_kind="transport", width=8, levels=2, mode_width=2,
            defect_relaxation=0.7, output_init="zero",
        )
        residual = torch.randn_like(rte.source)
        z0 = model.base(residual)
        z1 = model.base(residual - rte.A_op(z0))
        out = model(residual)
        torch.testing.assert_close(out, z0 + 0.7 * z1, rtol=0, atol=0)
        (residual - rte.A_op(out)).abs().square().mean().backward()
        gradient = model.unet.proj.conv.weight.grad
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.norm()), 0.0)

    def test_correction_parts_match_forward_without_duplicate_base_calls(self):
        rte, _ = make_case("mixed", 8, 6, torch.float32, 0.2)
        for form in ("residual", "defect2", "angular_spectral_defect2"):
            with self.subTest(form=form):
                kwargs = dict(
                    rte=rte, base_kind="transport", width=8, levels=2,
                    delta_scale=0.3, defect_relaxation=0.7, output_init="small",
                )
                if form == "angular_spectral_defect2":
                    model = AngularSpectralDefect2Preconditioner(mode_width=2, **kwargs)
                else:
                    model = NPBSUNetPreconditioner(form=form, **kwargs)
                residual = torch.randn_like(rte.source).requires_grad_()
                z0 = model.base(residual)
                expected_classical = z0
                base_calls = 1
                if form != "residual":
                    expected_classical = z0 + 0.7 * model.base(residual - rte.A_op(z0))
                    base_calls = 2
                self.assertEqual(model.base_calls_per_apply, base_calls)
                with patch.object(model, "base", wraps=model.base) as base:
                    with patch.object(model.coeff_encoder, "forward", wraps=model.coeff_encoder.forward) as encoder:
                        classical, delta = model.correction_parts(residual)
                    self.assertEqual(base.call_count, base_calls)
                    self.assertEqual(encoder.call_count, 1)
                torch.testing.assert_close(classical, expected_classical, rtol=0, atol=0)
                self.assertGreater(float(delta.detach().norm()), 0.0)
                with patch.object(model, "correction_parts", wraps=model.correction_parts) as parts:
                    with patch.object(model, "base", wraps=model.base) as base:
                        out = model(residual)
                    self.assertEqual(base.call_count, base_calls)
                    self.assertEqual(parts.call_count, 1)
                torch.testing.assert_close(out, classical + delta, rtol=0, atol=0)
                model.delta_scale *= 2.0
                classical2, delta2 = model.correction_parts(residual)
                torch.testing.assert_close(classical2, classical, rtol=0, atol=0)
                torch.testing.assert_close(delta2, 2.0 * delta, rtol=0, atol=0)
                (classical + delta).abs().square().mean().backward()
                self.assertTrue(bool(torch.isfinite(residual.grad).all()))
                gradient = model.unet.proj.conv.weight.grad
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(float(gradient.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
