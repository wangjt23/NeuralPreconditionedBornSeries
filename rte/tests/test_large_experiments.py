"""CPU regression tests; run with python -B -m unittest discover -s rte/tests.

CUDA tests are deliberately skipped unless hardware is present. A CPU pass is
not evidence of a CUDA run or a large-scale statistical result.
"""

import argparse
import csv
import io
import json
import math
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import torch

from rte import run_large_angular_spectral_experiments as runner
from rte.models import PeriodicRTE2D
from rte.npbs_unet_ablation import rte_coeff_feature_map


def tiny_config(**overrides):
    values = dict(dataset_size=3, eval_size=4, test_size=2, size=8, n_angles=4,
                  width=8, levels=2, angular_mode_width=2, angular_mix_rank=2,
                  batch_size=2, epochs=3, eval_every=2, eval_count=2,
                  fgmres_maxiter=2, teacher_depth=1, residual_teacher_depth=1,
                  rollout_steps=1, teacher_ramp_epochs=2, device="cpu",
                  regimes="diffusive", seed=17)
    values.update(overrides)
    return runner.ExperimentConfig(**values)


def make_fixture(cfg):
    data = runner.make_medium_dataset(cfg, cfg.dataset_size, cfg.seed)
    source = runner.source_from_data(cfg, data, torch.arange(cfg.dataset_size), cfg.device)
    rte = runner.make_operator(cfg, data["sigma_t"], data["sigma_s"], source, cfg.device)
    model = runner.make_model(cfg, rte).to(cfg.device)
    return data, source, rte, model


class LargeExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def assert_tree_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right), "tensor state differs")
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                if not isinstance(key, str) or ("seconds" not in key and key != "elapsed"):
                    self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for lhs, rhs in zip(left, right):
                self.assert_tree_equal(lhs, rhs)
        else:
            self.assertEqual(left, right)

    def test_global_regime_indices_and_reproducible_seeds(self):
        for seed in (3, 17, 103):
            for regimes in ("mixed", "anisotropic", "heterogeneous", "mixed,diffusive", "anisotropic,heterogeneous", "all"):
                with self.subTest(seed=seed, regimes=regimes):
                    cfg = tiny_config(regimes=regimes)
                    data = runner.make_medium_dataset(cfg, 12, seed)
                    duplicate = runner.make_medium_dataset(cfg, 12, seed)
                    self.assert_tree_equal(data, duplicate)
                    allowed = {runner.REGIMES.index(name) for name in runner.parse_regimes(regimes)}
                    self.assertTrue(set(data["regime_ids"].tolist()) <= allowed)
                    runner.validate_dataset(data, runner.dataset_metadata(cfg, 12, seed))
            mixed = runner.make_medium_dataset(tiny_config(regimes="mixed"), 12, seed)
            diffusive = runner.make_medium_dataset(tiny_config(regimes="diffusive"), 12, seed)
            self.assertFalse(torch.equal(mixed["sigma_t"], diffusive["sigma_t"]))
            self.assertLess(float(mixed["sigma_t"].min()), 1.5)
            self.assertGreater(float(mixed["sigma_t"].max()), 8.0)
        self.assertFalse(torch.equal(
            runner.make_medium_dataset(tiny_config(), 12, 3)["sigma_t"],
            runner.make_medium_dataset(tiny_config(), 12, 17)["sigma_t"]))
        with self.assertRaises(ValueError):
            runner.parse_regimes("mixed,mixed")

    def test_common_pool_prefix_is_identical(self):
        cfg = tiny_config(dataset_pool_size=12, regimes="all")
        for seed in (3, 17, 103):
            small = runner.make_medium_dataset(cfg, 3, seed)
            large = runner.make_medium_dataset(cfg, 12, seed)
            for key, value in small.items():
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(value, large[key][:3]), key)
            self.assertEqual(small["metadata"]["pool_size"], 12)

    def test_cache_round_trip_and_metadata_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = tiny_config(save_dataset=str(Path(directory) / "data.pt"))
            data = runner.save_or_load_dataset(cfg, 3, 17)
            self.assert_tree_equal(data, runner.save_or_load_dataset(cfg, 3, 17))
            for changes, count, seed in (({"regimes": "mixed"}, 3, 17), ({"size": 16}, 3, 17),
                                         ({"n_angles": 6}, 3, 17), ({}, 2, 17), ({}, 3, 18)):
                with self.subTest(changes=changes, count=count, seed=seed), self.assertRaisesRegex(ValueError, "metadata"):
                    runner.save_or_load_dataset(replace(cfg, **changes), count, seed)
            legacy = dict(data)
            legacy.pop("metadata")
            torch.save(legacy, cfg.save_dataset)
            with self.assertRaisesRegex(ValueError, "metadata"):
                runner.save_or_load_dataset(cfg, 3, 17)
            for key, value in (("sigma_s", float("nan")), ("sigma_t", -1.0), ("regime_ids", 3)):
                invalid = runner.cpu_tree(data)
                invalid[key].flatten()[0] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.validate_dataset(invalid, data["metadata"])

    def test_full_shuffled_epoch_and_explicit_cap(self):
        cfg = tiny_config(dataset_size=5)
        generator = torch.Generator().manual_seed(4)
        batches = runner.epoch_batches(cfg, generator)
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertEqual(sorted(torch.cat(batches).tolist()), list(range(5)))
        duplicate = runner.epoch_batches(cfg, torch.Generator().manual_seed(4))
        self.assert_tree_equal(batches, duplicate)
        capped = runner.epoch_batches(replace(cfg, steps_per_epoch=2), torch.Generator().manual_seed(4))
        self.assertEqual(sum(map(len, capped)), 4)
        self.assertEqual(len(torch.cat(capped).unique()), 4)
        with self.assertRaisesRegex(ValueError, "cap"):
            runner.validate_config(replace(cfg, steps_per_epoch=4))
        self.assertEqual(sum(map(len, runner.epoch_batches(replace(cfg, batch_size=8), generator))), 5)

    def test_csv_union_in_both_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            for rows in ([{"epoch": 1}, {"epoch": 2, "val_objective": 0.3}],
                         [{"epoch": 0, "val_objective": 0.4}, {"epoch": 1, "batch_objective": 0.2}]):
                runner.write_history(path, rows)
                with path.open() as handle:
                    read = list(csv.DictReader(handle))
                self.assertEqual(len(read), 2)
                self.assertEqual(read[1]["epoch"], str(rows[1]["epoch"]))
                self.assertIn("val_objective", read[0])

    def test_requested_cuda_fails_before_any_output(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                runner.run_experiment(tiny_config(device="cuda:0", output_root=directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_batched_coefficients_match_individual_operators(self):
        torch.manual_seed(71)
        cfg = tiny_config(regimes="all")
        data = runner.make_medium_dataset(cfg, cfg.dataset_size, cfg.seed)
        st, ss = data["sigma_t"].double(), data["sigma_s"].double()
        rhs = torch.randn(3, 4, 8, 8, dtype=torch.complex128)

        def operator(indices):
            return PeriodicRTE2D(shape=(8, 8), dx=1 / 8, n_angles=4, anisotropy_g=.65,
                                 fdtype=torch.float64).setup(st[indices], ss[indices], rhs[indices])

        batched = operator(slice(None))
        singles = [operator(slice(i, i + 1)) for i in range(3)]
        for method in ("A_op", "G", "transport_dsa_preconditioner", "reference_transport_solve"):
            expected = torch.cat([getattr(single, method)(rhs[i:i + 1]) for i, single in enumerate(singles)])
            torch.testing.assert_close(getattr(batched, method)(rhs), expected, rtol=2e-11, atol=2e-11)
        features = torch.cat([rte_coeff_feature_map(single, 1, extended=True) for single in singles])
        torch.testing.assert_close(rte_coeff_feature_map(batched, 3, extended=True), features)
        torch.testing.assert_close(batched.L_eta_op(rhs) - batched.V_op(rhs), batched.A_op(rhs))

    def test_batched_models_zero_init_and_nonzero_correction(self):
        for kind in ("angular_spectral_defect2", "unet_defect2"):
            with self.subTest(model=kind):
                torch.manual_seed(73)
                cfg = tiny_config(model_kind=kind, regimes="all")
                data, source, rte, model = make_fixture(cfg)
                z0 = model.base(source)
                expected = z0 + cfg.defect_relaxation * model.base(source - rte.A_op(z0))
                torch.testing.assert_close(model(source), expected, rtol=0, atol=0)
                with torch.no_grad():
                    model.unet.proj.conv.weight.normal_(std=0.02)
                batch_result = model(source)
                pieces = []
                for idx in range(cfg.dataset_size):
                    single = runner.setup_batch(cfg, rte, data, torch.tensor([idx]), "cpu")
                    pieces.append(model(single))
                torch.testing.assert_close(batch_result, torch.cat(pieces), rtol=3e-5, atol=2e-6)

    def test_batched_loss_and_gradients_are_sample_means(self):
        cfg = tiny_config(model_kind="unet_defect2", dataset_size=2, regimes="all")
        torch.manual_seed(71)
        data, residual, rte, model = make_fixture(cfg)
        loss, scalars = runner.training_objective(model, residual, cfg, cfg.teacher_weight)
        loss.backward()
        batch_gradients = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        single_losses, single_ratios = [], []
        for idx in range(2):
            single = runner.setup_batch(cfg, rte, data, torch.tensor([idx]), "cpu")
            single_loss, single_scalars = runner.training_objective(model, single, cfg, cfg.teacher_weight)
            single_losses.append(float(single_loss.detach()))
            single_ratios.append(single_scalars["raw_ratio"])
            (single_loss / 2).backward()
        self.assertAlmostEqual(float(loss.detach()), sum(single_losses) / 2, places=5)
        self.assertAlmostEqual(scalars["raw_ratio"], sum(single_ratios) / 2, places=6)
        for batched, parameter in zip(batch_gradients, model.parameters()):
            if batched is not None:
                torch.testing.assert_close(batched, parameter.grad, rtol=5e-5, atol=5e-6)

    def test_fixed_probe_evaluation_means_objective_and_rng_isolation(self):
        cfg = tiny_config(dataset_size=4, eval_count=4, random_probe_fraction=1.0)
        data, _, rte, model = make_fixture(cfg)
        state = torch.get_rng_state().clone()
        original_st = rte.sigma_t.clone()
        metrics, cases = runner.evaluate(model, rte, data, cfg, "cpu")
        again, again_cases = runner.evaluate(model, rte, data, cfg, "cpu")
        for key in metrics:
            if "seconds" not in key:
                self.assertAlmostEqual(metrics[key], again[key], places=6)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(original_st, rte.sigma_t))
        self.assertTrue(model.training)
        ratios = [case["physical_ratio"] for case in cases]
        self.assertAlmostEqual(metrics["physical_mean_square"], sum(r * r for r in ratios) / 4)
        self.assertAlmostEqual(metrics["physical_mean_ratio"], sum(ratios) / 4)
        self.assertGreater(metrics["physical_mean_square"] - metrics["physical_mean_ratio"] ** 2, 1e-9)
        self.assertAlmostEqual(metrics["objective"], sum(case["objective"] for case in cases) / 4)
        self.assertNotAlmostEqual(metrics["objective"], metrics["physical_mean_ratio"], places=4)
        self.assertEqual([case["index"] for case in cases], [case["index"] for case in again_cases])

    def test_residual_teacher_relaxation_changes_trajectory(self):
        cfg = tiny_config(random_probe_fraction=0, residual_teacher_depth=1)
        _, source, rte, _ = make_fixture(cfg)
        base = runner.make_training_residuals(rte, source, cfg, torch.Generator().manual_seed(5))
        damped = runner.make_training_residuals(rte, source, replace(cfg, residual_teacher_relaxation=.1),
                                                torch.Generator().manual_seed(5))
        self.assertFalse(torch.equal(base, damped))

    def test_priority_full_passes_nested_pool_and_matched_updates(self):
        cfg = tiny_config(dataset_size=4096, batch_size=64, epochs=1000, steps_per_epoch=0)
        plan = runner.priority_plan(argparse.Namespace(**asdict(cfg)))
        self.assertEqual(len(plan), 4)
        budgets = [item.epochs * math.ceil(item.dataset_size / item.batch_size) for item in plan]
        self.assertEqual(len(set(budgets)), 1)
        self.assertTrue(all(item.epochs >= 1000 and item.steps_per_epoch == 0 for item in plan))
        self.assertTrue(all(item.dataset_pool_size == 4096 for item in plan))
        ramps = [item.teacher_ramp_epochs * (item.dataset_size // item.batch_size) for item in plan]
        evaluation_steps = [item.eval_every * (item.dataset_size // item.batch_size) for item in plan]
        self.assertEqual(len(set(ramps)), 1)
        self.assertEqual(len(set(evaluation_steps)), 1)
        self.assertEqual(plan[2].model_kind, "unet_defect2")
        self.assertEqual(plan[3].loss_mode, "raw_rollout")
        self.assertEqual(plan[3].teacher_weight, 0)
        self.assertEqual(plan[3].rollout_weight, 0)
        for item in plan:
            runner.validate_config(item)

    def test_cli_priority_names_and_single_defaults(self):
        for arguments in (["runner", "--plan", "priority", "--device", "cpu"],
                          ["runner", "--device", "cpu", "--loss-mode", "raw_rollout"]):
            stream = io.StringIO()
            with mock.patch("sys.argv", arguments), mock.patch("sys.stdout", stream):
                runner.main()
            values = json.loads(stream.getvalue())
            if isinstance(values, list):
                self.assertEqual([item["name"] for item in values], [
                    "ensemble_small_angular_teacher", "ensemble_large_angular_teacher",
                    "ensemble_large_old_unet_teacher", "ensemble_large_angular_raw"])
                self.assertTrue(all(item["epochs"] >= 1000 and item["steps_per_epoch"] == 0 for item in values))
            else:
                self.assertEqual(values["steps_per_epoch"], 0)
                self.assertEqual(values["loss_mode"], "raw_rollout")
        with self.assertRaisesRegex(ValueError, "60 iterations"):
            runner.validate_config(tiny_config(fgmres_maxiter=61))

    def test_real_run_eval_every_two_and_partial_batch_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = tiny_config(output_root=directory)
            result = runner.run_experiment(cfg)
            path = Path(directory) / cfg.name
            with (path / "history.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["epoch"]) for row in rows], [0, 1, 2, 3])
            self.assertEqual(rows[1]["val_objective"], "")
            self.assertNotEqual(rows[2]["val_objective"], "")
            self.assertNotEqual(rows[2]["train_objective"], "")
            self.assertNotEqual(rows[1]["batch_objective"], "")
            self.assertEqual(result["updates"], 6)
            self.assertEqual(result["sample_presentations"], 9)
            self.assertEqual(result["effective_passes"], 3.0)
            self.assertEqual(result["test"]["count"], cfg.test_size)
            self.assertEqual(result["classical_dsa4_test"]["count"], cfg.test_size)
            self.assertGreater(result["classical_dsa4_test"]["solve_seconds_mean"], 0)
            for case in result["classical_dsa4_test_cases"]:
                self.assertEqual(case["base_calls"], 4 * case["preconditioner_calls"])
            self.assertEqual(result["runtime"]["device"], "cpu")
            best = torch.load(path / "best.pt", weights_only=True)
            self.assertEqual(best["epoch"], result["best_epoch"])
            self.assertTrue(best["selection_only"])
            self.assertTrue((path / "last.pt").exists())

    def test_capped_run_reports_effective_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = runner.run_experiment(tiny_config(output_root=directory, dataset_size=5,
                                                       steps_per_epoch=1, epochs=1))
            self.assertEqual(result["updates"], 1)
            self.assertEqual(result["sample_presentations"], 2)
            self.assertEqual(result["effective_passes"], 0.4)

    def test_epoch_zero_remains_best_and_test_is_final_only(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = tiny_config(output_root=directory, epochs=2, eval_every=1)
            calls = []

            def fake_evaluate(model, rte, data, config, device):
                split_seed = data["metadata"]["seed"]
                calls.append(split_seed)
                val_calls = calls.count(cfg.seed + 100003)
                score = float(max(1, val_calls))
                return {"objective": score + 2, "physical_mean_square": score ** 2,
                        "fgmres_mean_ratio": score}, [{"seed": split_seed}]

            with mock.patch.object(runner, "evaluate", side_effect=fake_evaluate):
                result = runner.run_experiment(cfg)
            self.assertEqual(result["best_epoch"], 0)
            self.assertEqual(calls[-3:], [cfg.seed + 200003] * 3)
            self.assertNotIn(cfg.seed + 200003, calls[:-3])
            path = Path(directory) / cfg.name
            best = torch.load(path / "best.pt", weights_only=True)
            last = torch.load(path / "last.pt", weights_only=True)
            self.assertEqual(best["epoch"], 0)
            self.assertEqual(last["epoch"], 2)
            self.assert_tree_equal(best["model"], last["best"]["model"])

    def test_final_test_loads_nonfinal_selected_best_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = tiny_config(output_root=directory, epochs=2, eval_every=1)
            validation_states, test_states = [], []

            def fake_evaluate(model, rte, data, config, device):
                seed = data["metadata"]["seed"]
                score = 1.0
                if seed == cfg.seed + 100003:
                    validation_states.append(runner.cpu_tree(model.state_dict()))
                    score = (5.0, 1.0, 3.0)[len(validation_states) - 1]
                if seed == cfg.seed + 200003:
                    test_states.append(runner.cpu_tree(model.state_dict()))
                return {"objective": score + 2, "physical_mean_square": score ** 2,
                        "fgmres_mean_ratio": score}, [{"seed": seed}]

            with mock.patch.object(runner, "evaluate", side_effect=fake_evaluate):
                result = runner.run_experiment(cfg)
            self.assertEqual(result["best_epoch"], 1)
            self.assert_tree_equal(test_states[0], validation_states[1])
            self.assertEqual(test_states[1], {})
            self.assertEqual(test_states[2], {})
            best = torch.load(Path(directory) / cfg.name / "best.pt", weights_only=True)
            self.assertEqual(best["epoch"], 1)
            self.assert_tree_equal(best["model"], validation_states[1])

    def test_classical_controls_have_no_network_and_match_dsa_stacks(self):
        cfg = tiny_config()
        _, source, rte, _ = make_fixture(cfg)
        for depth in (2, 4):
            model = runner.ClassicalStackPreconditioner(rte, depth)
            self.assertEqual(list(model.parameters()), [])
            expected = torch.zeros_like(source)
            for _ in range(depth):
                expected = expected + rte.apply_preconditioner(source - rte.A_op(expected), kind="transport_dsa")
            torch.testing.assert_close(model(source), expected, rtol=2e-5, atol=1e-6)

    def test_exact_epoch_boundary_resume_same_horizon_multiple_seeds(self):
        class Interrupted(Exception):
            pass

        for seed in (17, 103):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as directory:
                cfg = tiny_config(output_root=directory, seed=seed)
                runner.run_experiment(replace(cfg, name="uninterrupted"))
                original_save = runner.save_checkpoint

                def interrupt_after_saved_epoch(path, payload):
                    original_save(path, payload)
                    if Path(path).name == "last.pt" and payload["epoch"] == 1:
                        raise Interrupted()

                with mock.patch.object(runner, "save_checkpoint", side_effect=interrupt_after_saved_epoch):
                    with self.assertRaises(Interrupted):
                        runner.run_experiment(replace(cfg, name="interrupted"))
                interrupted = Path(directory) / "interrupted" / "last.pt"
                runner.run_experiment(replace(cfg, name="resumed", resume=str(interrupted)))
                full = torch.load(Path(directory) / "uninterrupted" / "last.pt", weights_only=True)
                resumed = torch.load(Path(directory) / "resumed" / "last.pt", weights_only=True)
                for key in ("model", "optimizer", "scheduler", "rng", "updates", "sample_presentations", "best"):
                    self.assert_tree_equal(full[key], resumed[key])
                for history in (full["history"], resumed["history"]):
                    for row in history:
                        row.pop("elapsed", None)
                self.assert_tree_equal(full["history"], resumed["history"])
                with self.assertRaisesRegex(ValueError, "horizon"):
                    runner.run_experiment(replace(cfg, name="bad_horizon", epochs=4, resume=str(interrupted)))
                self.assertFalse((Path(directory) / "bad_horizon").exists())
                changed = runner.cpu_tree(full)
                changed["dataset_fingerprint"] = "different"
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    runner.validate_resume(changed, replace(cfg, name="uninterrupted"), full["dataset_fingerprint"], full["runtime"])

    def test_same_seed_complete_runs_reproduce(self):
        for seed in (17, 103):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as directory:
                cfg = tiny_config(output_root=directory, epochs=1, seed=seed)
                for name in ("first", "second"):
                    runner.run_experiment(replace(cfg, name=name))
                first = torch.load(Path(directory) / "first" / "last.pt", weights_only=True)
                second = torch.load(Path(directory) / "second" / "last.pt", weights_only=True)
                for key in ("model", "optimizer", "scheduler", "rng", "best"):
                    self.assert_tree_equal(first[key], second[key])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable; no CUDA validation claimed")
    def test_cuda_forward_backward_and_artifact_device(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = tiny_config(output_root=directory, epochs=1, device="cuda:0")
            result = runner.run_experiment(cfg)
            self.assertEqual(result["runtime"]["device"], "cuda:0")
            self.assertTrue(math.isfinite(result["test"]["physical_mean_square"]))
            with (Path(directory) / cfg.name / "config.json").open() as handle:
                self.assertEqual(json.load(handle)["runtime"]["device"], "cuda:0")


if __name__ == "__main__":
    unittest.main()
