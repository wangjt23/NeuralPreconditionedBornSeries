import unittest

import torch

from rte.collision_transport import CollisionTransportPreconditioner
from rte.run_hard_case_experiments import make_case


class CollisionTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make(self, size=4, angles=4, dtype=torch.float64):
        return make_case("mixed", size, angles, dtype, .1)[0]

    def test_exact_collision_fourier_diagonalization(self):
        rte = self.make(5, 7)
        model = CollisionTransportPreconditioner(rte)
        r = torch.randn_like(rte.source)
        actual = torch.fft.ifft(model.collision_spectrum() * torch.fft.fft(r, dim=1), dim=1)
        expected = rte.sigma_t * r - rte.scattering_op(r)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_layer_against_independent_dense_hss(self):
        rte = self.make()
        model = CollisionTransportPreconditioner(rte, depth=1)
        dimension = rte.source.numel()
        basis = torch.eye(dimension, dtype=torch.complex128).reshape(dimension, *rte.source.shape[1:])
        streaming = rte.streaming_op(basis).flatten(1).T
        collision = (rte.sigma_t * basis - rte.scattering_op(basis)).flatten(1).T
        r = torch.randn_like(rte.source)
        (_, shifts, _, _), _ = model.fields()
        alpha = shifts.item()
        identity = torch.eye(dimension, dtype=torch.complex128)
        matrix = 2 * alpha * torch.linalg.solve(alpha * identity + streaming,
                                                torch.linalg.inv(alpha * identity + collision))
        torch.testing.assert_close(model(r).flatten(), matrix @ r.flatten(), atol=1e-12, rtol=1e-12)
        error = identity - matrix @ (streaming + collision)
        bound = ((alpha - torch.linalg.eigvalsh(collision)) / (alpha + torch.linalg.eigvalsh(collision))).abs().max()
        self.assertLessEqual(float(torch.linalg.eigvals(error).abs().max()), float(bound) + 1e-10)

    def test_learned_initialization_and_complex_linearity(self):
        rte = self.make(dtype=torch.float32)
        classical = CollisionTransportPreconditioner(rte, depth=2)
        learned = CollisionTransportPreconditioner(rte, depth=2, learned=True)
        r, s = torch.randn_like(rte.source), torch.randn_like(rte.source)
        torch.testing.assert_close(learned(r), classical(r))
        with torch.no_grad():
            learned.shift_net[-1].bias.fill_(.15)
            learned.gain_net[-1].bias.normal_()
        a = .3 + .7j
        torch.testing.assert_close(learned(a * r + s), a * learned(r) + learned(s), atol=1e-6, rtol=1e-5)
        loss = (r - rte.A_op(learned(r))).abs().square().mean()
        loss.backward()
        self.assertGreater(float(learned.shift_net[-1].weight.grad.norm()), 0)
        self.assertGreater(float(learned.gain_net[-1].weight.grad.norm()), 0)

    def test_cache_invalidation_and_no_repeated_network(self):
        rte = self.make(dtype=torch.float32)
        model = CollisionTransportPreconditioner(rte, learned=True).eval()
        calls = []
        hook = model.gain_net.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            a = model(rte.source)
            b = model(rte.source)
            self.assertEqual(len(calls), 1)
            torch.testing.assert_close(a, b)
            model.shift_net[-1].bias.add_(.1)
            model(rte.source)
            self.assertEqual(len(calls), 2)
            rte.setup(rte.sigma_t + .2, rte.sigma_s, rte.source)
            model(rte.source)
            self.assertEqual(len(calls), 3)
        hook.remove()

    def test_scalar_protection_and_positive_gains(self):
        model = CollisionTransportPreconditioner(self.make(), learned=True).double()
        with torch.no_grad():
            model.gain_net[-1].bias.fill_(100)
        (_, alpha, gains, _), _ = model.fields()
        self.assertTrue((alpha > 0).all())
        torch.testing.assert_close(gains[:, :, 0], torch.ones_like(gains[:, :, 0]))
        self.assertTrue((gains > 0).all())

    def test_batch_broadcast_and_signed_mode_symmetry(self):
        rte = self.make(5, 7, torch.float32)
        model = CollisionTransportPreconditioner(rte, learned=True).eval()
        with torch.no_grad():
            model.gain_net[-1].bias.normal_()
            batch = torch.randn(3, *rte.source.shape[1:], dtype=rte.source.dtype)
            torch.testing.assert_close(model(batch), torch.cat([model(r) for r in batch.split(1)]))
            (_, _, gains, _), _ = model.fields()
            torch.testing.assert_close(gains[:, :, 1], gains[:, :, -1])
            torch.testing.assert_close(gains[:, :, 2], gains[:, :, -2])

    def test_macro_against_independent_projected_dense_reference(self):
        rte = self.make()
        model = CollisionTransportPreconditioner(rte, depth=1, macro_harmonics=1)
        plain = CollisionTransportPreconditioner(rte, depth=1)
        n, size = rte.n_angles, rte.source.shape[-1]
        dim = rte.source.numel()
        basis = torch.eye(dim, dtype=torch.complex128).reshape(dim, n, size, size)
        reference_action = (rte.streaming_op(basis) + rte.sigma_t.mean() * basis
                            - rte.sigma_s.mean() * rte.scattering_op(basis) / rte.sigma_s)
        reference = reference_action.flatten(1).T
        angular_basis = torch.fft.ifft(torch.eye(n, dtype=torch.complex128), dim=0, norm="ortho")[:, [0, 1, n-1]]
        prolongation = torch.kron(angular_basis.contiguous(), torch.eye(size * size, dtype=torch.complex128))
        coarse = prolongation.mH @ reference @ prolongation
        residual = torch.randn_like(rte.source)
        baseline = plain(residual)
        defect = residual - rte.A_op(baseline)
        correction = prolongation @ torch.linalg.solve(coarse, prolongation.mH @ defect.flatten())
        torch.testing.assert_close(model(residual), baseline + correction.reshape_as(baseline), atol=1e-11, rtol=1e-11)

    def test_dx_in_streaming_and_gradient_after_preparation(self):
        rte = self.make(dtype=torch.float32)
        model = CollisionTransportPreconditioner(rte, learned=True)
        model.prepare()
        model.train()
        output = model(rte.source)
        output.abs().square().mean().backward()
        self.assertGreater(float(model.shift_net[-1].bias.grad.norm()), 0)
        # Grid spacing enters the actual k-grid, not an after-the-fact loss factor.
        r = torch.randn_like(rte.source)
        stream = rte.streaming_op(r)
        rte.kx = rte.kx / 2
        rte.ky = rte.ky / 2
        torch.testing.assert_close(rte.streaming_op(r), stream / 2)

    def test_stage_sequence_matches_separate_layers(self):
        rte = self.make()
        a = CollisionTransportPreconditioner(rte, depth=1, shift_factor=.5)
        b = CollisionTransportPreconditioner(rte, depth=1, shift_factor=2.)
        combined = CollisionTransportPreconditioner(rte, depth=2, stage_factors=(.5, 2.))
        r = torch.randn_like(rte.source)
        z = a(r)
        torch.testing.assert_close(combined(r), z + b(r - rte.A_op(z)), atol=1e-12, rtol=1e-12)

    def test_prepared_unet_exact_control(self):
        from rte.npbs_unet_ablation import NPBSUNetPreconditioner
        from rte.prepared_defect_unet import PreparedDefectUNet
        rte = self.make(8, 7, torch.float32)
        model = NPBSUNetPreconditioner(rte, form="defect2", base_kind="transport_tsa",
                                      width=8, levels=2, output_init="small").eval()
        prepared = PreparedDefectUNet(model)
        with torch.no_grad():
            r = torch.randn(3, *rte.source.shape[1:], dtype=rte.source.dtype)
            torch.testing.assert_close(prepared(r), model(r), atol=1e-6, rtol=1e-5)
            rte.sigma_t.add_(.1)
            with self.assertRaises(RuntimeError):
                prepared(r)

    def test_global_gain_ablation_and_bank_composition(self):
        from rte.run_collision_transport_experiment import make_bank
        from rte.run_residual_subspace_experiment import sources
        rte = self.make(8, 7, torch.float32)
        model = CollisionTransportPreconditioner(rte, learned=True, spatial_gains=False)
        with torch.no_grad():
            model.gain_net[-1].weight.normal_()
            (_, _, gains, _), _ = model.fields()
            torch.testing.assert_close(gains, gains[..., :1, :1].expand_as(gains))
        source_list = sources(8, 7, 3, 1, torch.device("cpu"))
        bank, labels = make_bank(rte, source_list, 2)
        self.assertEqual(len(bank), 18)
        self.assertEqual(labels.count("stage0"), 3)
        self.assertEqual(labels.count("smooth_action"), 6)
        torch.testing.assert_close(bank.abs().square().flatten(1).sum(1), torch.ones(18))

    def test_multiple_coefficient_fields_match_individual_calls(self):
        first, second, batched = [self.make(8, 7, torch.float32) for _ in range(3)]
        second.setup(second.sigma_t + .4, second.sigma_s * .9, second.source)
        batched.setup(torch.cat([first.sigma_t, second.sigma_t]),
                      torch.cat([first.sigma_s, second.sigma_s]),
                      torch.cat([first.source, second.source]))
        models = [CollisionTransportPreconditioner(rte, learned=True).eval() for rte in (first, second, batched)]
        with torch.no_grad():
            models[0].gain_net[-1].weight.normal_(std=.1)
            models[0].shift_net[-1].weight.normal_(std=.1)
            for model in models[1:]:
                model.load_state_dict(models[0].state_dict())
            expected = torch.cat([models[0](first.source), models[1](second.source)])
            torch.testing.assert_close(models[2](batched.source), expected, atol=1e-6, rtol=1e-5)

    def test_precomputed_filters_match_uncached_forward_and_refresh(self):
        rte = self.make(8, 7, torch.float32)
        model = CollisionTransportPreconditioner(rte, depth=4, learned=True)
        with torch.no_grad():
            model.gain_net[-1].bias.normal_(std=.2)
            model.shift_net[-1].bias.normal_(std=.1)
            residual = torch.randn(3, *rte.source.shape[1:], dtype=rte.source.dtype)
            uncached = model(residual)
            model.prepare()
            torch.testing.assert_close(model(residual), uncached, atol=1e-6, rtol=1e-5)
            before = model._prepared_filters[0].clone()
            model.gain_net[-1].bias.add_(.2)
            model(residual)
            self.assertFalse(torch.equal(before, model._prepared_filters[0]))


if __name__ == "__main__":
    unittest.main()
