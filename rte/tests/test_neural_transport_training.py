import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rte.train_neural_transport import run


class NeuralTransportTrainingTests(unittest.TestCase):
    def arguments(self, output, **kwargs):
        values = dict(model='nto', depth=2, width=4, base='hss', no_mixing=False,
                      no_spectral=False, size=8, n_angles=4, g=.65, train_size=3,
                      val_size=1, test_size=0, batch_size=2, selection_count=1,
                      steps=2, eval_every=1, lr=.001, broad_fraction=.2,
                      rollout_weight=.25, seed=404, solver_steps=2, tol=1e-3,
                      device='cpu', resume='', stop_after=0, output=str(output))
        return SimpleNamespace(**{**values, **kwargs})

    def test_partial_batch_update_resume_and_no_test_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run(self.arguments(root/'full'))
            run(self.arguments(root/'split', stop_after=1))
            run(self.arguments(root/'split', resume=str(root/'split'/'last.pt')))
            full = torch.load(root/'full'/'last.pt', weights_only=True)
            split = torch.load(root/'split'/'last.pt', weights_only=True)
            self.assertEqual(full['step'], 2)
            self.assertEqual(full['offset'], 3)
            self.assertEqual(full['scheduler'], split['scheduler'])
            torch.testing.assert_close(full['permutation'], split['permutation'], atol=0, rtol=0)
            torch.testing.assert_close(full['rng']['sampler'], split['rng']['sampler'], atol=0, rtol=0)
            for key in full['model']:
                torch.testing.assert_close(full['model'][key], split['model'][key], atol=0, rtol=0)
            self.assertEqual(set(full['fingerprints']), {'train', 'validation'})
            with self.assertRaisesRegex(ValueError, 'configuration'):
                run(self.arguments(root/'split', steps=3, resume=str(root/'split'/'last.pt')))


if __name__ == '__main__':
    unittest.main()
