import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from rte.compiled_hss import CompiledHSS
from rte.collision_transport import CollisionTransportPreconditioner
from rte.mixed_precision_transport import MixedPrecisionPreconditioner, high_precision_copy
from rte.neural_transport_operator import NeuralTransportOperator
from rte.run_hard_case_experiments import make_case
from rte.train_krylov_transport import short_fgmres, training_loss, run


class KrylovTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_short_solve_matches_dense_and_has_gradients(self):
        torch.manual_seed(52)
        matrix = torch.tensor([[3., 1., 0.], [-1., 2., 1.], [0., -.5, 4.]], dtype=torch.float64)
        rhs = torch.randn(2, 1, 1, 3, dtype=torch.float64).to(torch.complex128)
        gains = torch.nn.Parameter(torch.tensor([.8, 1., 1.2], dtype=torch.float32))
        operator = lambda x: torch.einsum('ij,blhj->blhi', matrix.to(x.dtype), x)
        method = lambda x: gains*x
        squared, solution = short_fgmres(operator, method, rhs, 3)
        self.assertLess(float(squared.max()), 1e-12)
        torch.testing.assert_close(operator(solution), rhs, atol=1e-6, rtol=1e-6)
        short_fgmres(operator, method, rhs, 2)[0].mean().backward()
        self.assertTrue(torch.isfinite(gains.grad).all())
        self.assertGreater(float(gains.grad.norm()), 1e-8)

    def test_mixed_wrapper_refresh_and_discrete_geometry(self):
        low = make_case('mixed', 8, 6, torch.float32, .1)[0]
        high = high_precision_copy(low)
        for name in ('directions', 'phase_matrix', 'weights', 'kx', 'ky'):
            torch.testing.assert_close(getattr(high, name), getattr(low, name).double(), atol=0, rtol=0)
        model = CompiledHSS(low, 2)
        mixed = MixedPrecisionPreconditioner(high, low, model)
        with torch.no_grad():
            first = mixed(high.source)
            self.assertEqual(first.dtype, torch.complex128)
            high.setup(high.sigma_t+.2, high.sigma_s, high.source)
            second = mixed(high.source)
            torch.testing.assert_close(low.sigma_t, high.sigma_t.float())
            self.assertFalse(torch.allclose(first, second))

    def test_neural_short_solve_backprop_and_field_restoration(self):
        low = make_case('mixed', 8, 6, torch.float32, .1)[0]
        high = high_precision_copy(low)
        model = NeuralTransportOperator(low, depth=2, width=8, base='hss')
        original = model.fields
        loss, squared = training_loss(high, model, low.source, 2)
        loss.backward()
        self.assertEqual(model.fields, original)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_ct_field_tuple_reuse(self):
        low = make_case('mixed', 8, 6, torch.float32, .1)[0]
        high = high_precision_copy(low)
        model = CollisionTransportPreconditioner(low, depth=2, learned=True)
        loss, _ = training_loss(high, model, low.source, 2)
        loss.backward()
        self.assertTrue(torch.isfinite(model.shift_net[-1].weight.grad).all())

    def test_partial_batch_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            op = make_case('mixed', 8, 4, torch.float32, .1)[0]
            model = NeuralTransportOperator(op, depth=2, width=4, base='hss')
            initial_args = dict(model='nto', depth=2, width=4, base='hss', no_mixing=False,
                                no_spectral=False, n_angles=4, g=.65)
            torch.save(dict(args=initial_args, model=model.state_dict()), root/'initial.pt')
            def arguments(output, **changes):
                values = dict(warm_start=str(root/'initial.pt'), spectral_upgrade=False, size=8,
                    seed=404, train_size=3, val_size=1, batch_size=2, steps=2, krylov_steps=2,
                    source_fraction=.75, lr=.0003, eval_every=1, selection_count=1, solver_steps=2,
                    tol=1e-3, device='cpu', resume='', stop_after=0, output=str(output))
                return SimpleNamespace(**{**values, **changes})
            run(arguments(root/'full'))
            run(arguments(root/'split', stop_after=1))
            run(arguments(root/'split', resume=str(root/'split'/'last.pt')))
            full = torch.load(root/'full'/'last.pt', weights_only=True)
            split = torch.load(root/'split'/'last.pt', weights_only=True)
            self.assertEqual(full['offset'], 3)
            self.assertEqual(full['scheduler'], split['scheduler'])
            for key in full['model']:
                torch.testing.assert_close(full['model'][key], split['model'][key], atol=0, rtol=0)
            torch.testing.assert_close(full['rng']['sampler'], split['rng']['sampler'], atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                run(arguments(root/'split', steps=3, resume=str(root/'split'/'last.pt')))


if __name__ == '__main__':
    unittest.main()
