import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from rte.run_large_angular_spectral_experiments import ExperimentConfig, dataset_fingerprint
from rte.train_collision_transport_ensemble import material_data, run


class CollisionEnsembleTests(unittest.TestCase):
    def args(self, output, stop_after):
        return argparse.Namespace(train_size=4, val_size=2, test_size=2, selection_count=1,
                                  size=8, n_angles=8, regimes="mixed", g=.65, epochs=3,
                                  batch_size=3, eval_every=1, probe_fraction=1/3, lr=.003,
                                  solver_steps=3, tol=1e-5, repeats=1, seed=21, threads=1,
                                  device="cpu", output=str(output), resume="", stop_after=stop_after)

    def test_material_reproduction_geometry_and_disjoint_splits(self):
        cfg = ExperimentConfig(dataset_size=8, size=8, n_angles=8, regimes="mixed")
        a, b, c = (material_data(cfg, 8, seed) for seed in (10, 10, 11))
        self.assertEqual(dataset_fingerprint(a), dataset_fingerprint(b))
        self.assertNotEqual(dataset_fingerprint(a), dataset_fingerprint(c))
        self.assertTrue((a["sigma_s"] < a["sigma_t"]).all())
        self.assertEqual(a["geometry"].shape, (8, 3))

    def test_exact_resume_before_any_test_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = self.args(root / "full", 2)
            self.assertEqual(run(full)["status"], "paused")
            partial = self.args(root / "split", 1)
            run(partial)
            resumed = self.args(root / "split", 2)
            resumed.resume = str(root / "split" / "last.pt")
            run(resumed)
            a = torch.load(root / "full" / "last.pt", weights_only=True)
            b = torch.load(root / "split" / "last.pt", weights_only=True)
            self.assertEqual(a["updates"], 4)  # Includes the partial final minibatch.
            self.assertEqual(a["history"], b["history"])
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
            self.assertFalse((root / "split" / "test_fingerprint.txt").exists())
            self.assertFalse((root / "split" / "results.csv").exists())


if __name__ == "__main__":
    unittest.main()
