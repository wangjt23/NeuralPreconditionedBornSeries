"""Regression coverage for nonlinear inference artifact serialization."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from nonlinear.inference import main
from nonlinear.model import ModelConfig, NeuralNewtonSolver, save_checkpoint


class InferenceArtifactsTest(unittest.TestCase):
    def test_multifile_results_and_json_sidecar_keep_separate_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            model = NeuralNewtonSolver(ModelConfig(grid_size=4, width=1, modes=(1,), padding=0))
            save_checkpoint(checkpoint, model)
            data_paths = [root / "first.pt", root / "second.pt"]
            for index, path in enumerate(data_paths):
                torch.save({"u": torch.full((3, 1, 4, 4), float(index))}, path)

            for suffix in (".pt", ".json"):
                with self.subTest(suffix=suffix):
                    output = root / ("result" + suffix)
                    arguments = ["inference.py", "--checkpoint", str(checkpoint), "--data",
                                 *map(str, data_paths), "--start", "1", "--count", "2",
                                 "--batch-size", "2", "--max-iterations", "0",
                                 "--output", str(output)]
                    with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                        main()
                    result = torch.load(output, map_location="cpu", weights_only=True)
                    self.assertEqual(tuple(result["solution"].shape), (4, 1, 4, 4))
                    self.assertEqual(
                        [(record["source_file"], record["sample"]) for record in result["records"]],
                        [(str(path), sample) for path in data_paths for sample in (1, 2)],
                    )
                    sidecar = output.with_suffix(".records.json" if suffix == ".json" else ".json")
                    self.assertEqual(json.loads(sidecar.read_text()), result["records"])


if __name__ == "__main__":
    unittest.main()
