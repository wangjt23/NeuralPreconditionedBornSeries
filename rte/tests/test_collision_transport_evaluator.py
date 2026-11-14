import math
import unittest
from types import SimpleNamespace

import torch

from rte.collision_transport import CollisionTransportPreconditioner
from rte.evaluate_collision_transport import (
    aggregate_summary, candidate_score, checkpoint_state, evaluate_method, percentile,
)
from rte.run_hard_case_experiments import make_case


class CollisionTransportEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_percentile_and_checkpoint_selection(self):
        self.assertEqual(percentile([1, 2, 3], .5), 2)
        self.assertAlmostEqual(percentile([1, 2], .95), 1.95)
        state, epoch = checkpoint_state({"model": {"x": 1}, "selected_epoch": 7})
        self.assertEqual((state, epoch), ({"x": 1}, 7))
        state, epoch = checkpoint_state({"model": {"x": 1}, "best_state": {"x": 2}, "best_epoch": 3})
        self.assertEqual((state, epoch), ({"x": 2}, 3))

    def test_measurement_retains_cold_warm_calls_and_memory(self):
        rte, _ = make_case("mixed", 4, 4, torch.float32, .1)
        learned = CollisionTransportPreconditioner(rte, depth=4, learned=True).eval()
        args = SimpleNamespace(solver_steps=5, tol=1e-3)
        spec = {"family": "hss", "depth": 1, "scale": 1.}
        case, samples = evaluate_method(spec, rte, rte.source, args, torch.device("cpu"),
                                        learned, repeats=2, phase="test", index=0, order=0)
        self.assertEqual([row["repeat_kind"] for row in samples], ["cold", "warm", "warm"])
        self.assertEqual(case["warm_repeats"], 2)
        self.assertIn("source_loss", case)
        self.assertGreaterEqual(case["source_action_operator_calls"], 1)
        for row in samples:
            self.assertGreaterEqual(row.get("operator_calls", 0), 1)
            self.assertGreaterEqual(row.get("preconditioner_calls", 0), 1)
            self.assertEqual(row.get("factor_stage_calls"), row.get("preconditioner_calls"))
        summary = aggregate_summary([case], samples, args.tol)
        self.assertEqual(summary[0]["warm_solves"], 2)
        self.assertEqual(summary[0]["failures"], sum(not row["converged"] for row in samples[1:]))

    def test_candidate_score_prioritizes_failures_then_time(self):
        converged = [{"warm_failures": 0, "warm_median_final": 1e-6, "warm_median_seconds": .2}]
        failed = [{"warm_failures": 1, "warm_median_final": 1e-2, "warm_median_seconds": .1}]
        self.assertEqual(candidate_score(converged, 1e-5)[0], 0)
        self.assertEqual(candidate_score(failed, 1e-5)[0], 1)
        self.assertTrue(math.isfinite(candidate_score(failed, 1e-5)[1]))

    def test_failed_solves_are_censored_in_unconditional_percentiles(self):
        cases = [{"method": "CT", "cold_converged": False, "warm_failures": 1}]
        samples = [{"method": "CT", "repeat_kind": "warm", "seconds": .1, "converged": True},
                   {"method": "CT", "repeat_kind": "warm", "seconds": .2, "converged": False}]
        row = aggregate_summary(cases, samples, 1e-5)[0]
        self.assertEqual(row["failed_media"], 1)
        self.assertEqual(row["cold_failures"], 1)
        self.assertEqual(row["p95_time_to_tolerance_seconds"], .1)
        self.assertTrue(row["p95_all_attempts_unreached"])
        self.assertIsNone(row["p95_all_attempts_time_to_tolerance_seconds"])


if __name__ == "__main__":
    unittest.main()
