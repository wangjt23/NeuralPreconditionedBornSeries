"""Serialization/restart checks only: no training steps or Krylov solves."""

import argparse
from pathlib import Path
import tempfile
import unittest

import torch

from rte.run_large_angular_spectral_experiments import ExperimentConfig, dataset_fingerprint
from rte.train_collision_transport_ensemble import material_data, material_split, run, warm_start


class WorkstationHandoffTests(unittest.TestCase):
    def test_weight_only_transfer_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            source, target = torch.nn.Linear(2, 3), torch.nn.Linear(2, 3)
            torch.save({"model": source.state_dict(), "selected_epoch": 60}, path)
            provenance = warm_start(target, path)
            self.assertEqual(provenance["source_epoch"], 60)
            self.assertEqual(provenance["optimizer_scheduler_rng"], "reset")
            self.assertEqual(len(provenance["sha256"]), 64)
            for key, value in source.state_dict().items():
                torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
            with self.assertRaises(RuntimeError):
                warm_start(torch.nn.Linear(3, 2), path)

    def test_snapshot_load_and_corruption_or_configuration_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(train_size=2, val_size=2, test_size=2,
                                      seed=1, g=.65, dataset_dir=directory)
            cfg = ExperimentConfig(size=8, n_angles=8, regimes="mixed")
            data = material_data(cfg, 2, 1001)
            payload = {"schema": 1, "g": .65, "data": data, "fingerprint": dataset_fingerprint(data)}
            path = Path(directory) / "train.pt"
            torch.save(payload, path)
            self.assertEqual(dataset_fingerprint(material_split(args, cfg, "train")), payload["fingerprint"])
            args.g = .8
            with self.assertRaisesRegex(ValueError, "configuration"):
                material_split(args, cfg, "train")
            args.g = .65
            payload["data"]["sigma_t"][0, 0, 0, 0] += 1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                material_split(args, cfg, "train")

    def test_resume_and_warm_start_rejected_before_any_training(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            run(argparse.Namespace(resume="last.pt", warm_start="best.pt"))


if __name__ == "__main__":
    unittest.main()
