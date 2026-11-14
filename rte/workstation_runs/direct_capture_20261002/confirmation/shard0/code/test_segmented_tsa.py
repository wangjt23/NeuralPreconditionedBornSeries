"""CPU recipe/guard tests for segmented replay of the unchanged TSA stages.

Fake replay objects execute the actual stage functions into static buffers.
They verify recipe arithmetic and ownership; they make no CUDA timing claim.
The fixtures are deterministic N8/L12 unit examples, not held-out source data.
"""

import gc
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import weakref

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "remote_reference"
if not (SOURCE_ROOT / "rte" / "models.py").is_file():
    SOURCE_ROOT = Path.cwd() if (Path.cwd() / "rte" / "models.py").is_file() else ROOT
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rte.models import PeriodicRTE2D
from rte.transport_baselines import ClassicalTransportStack
from segmented_tsa import SegmentedTSA


class FakeStepReplay:
    def __init__(self, wrapper):
        self.wrapper, self.calls = wrapper, 0

    def replay(self):
        self.calls += 1
        value = SegmentedTSA._subsequent_stage(
            self.wrapper.base, self.wrapper.static_r, self.wrapper.static_z
        )
        self.wrapper.step_output.copy_(value)


class FakeFirstReplay:
    def __init__(self, wrapper):
        self.wrapper, self.calls = wrapper, 0

    def replay(self):
        self.calls += 1
        self.wrapper.first_output.copy_(
            SegmentedTSA._first_stage(self.wrapper.base, self.wrapper.static_r)
        )


class SegmentedTSATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def fixture(self, depth=4, capture_first=False):
        n = 8
        physical = PeriodicRTE2D(
            shape=(n, n), dx=1. / n, n_angles=12, anisotropy_g=.65,
            eta_scale=0., eta_floor=0., fdtype=torch.float32, device="cpu",
        )
        axis = torch.arange(n, dtype=torch.float32) * (2 * torch.pi / n)
        t = (3. + .21 * axis.cos()[:, None] + .13 * axis.sin()[None, :])[None, None]
        s = (1. + .09 * axis.sin()[:, None] + .07 * axis.cos()[None, :])[None, None]
        generator = torch.Generator(device="cpu").manual_seed(531)
        rhs = torch.randn(1, 12, n, n, dtype=torch.complex64, generator=generator)
        physical.setup(t, s, rhs)
        base = ClassicalTransportStack(physical, "tsa", depth=depth, eta_scale=.13, relaxation=.73)
        wrapper = object.__new__(SegmentedTSA)
        first_output = torch.zeros_like(rhs) if capture_first else None
        step_output = torch.zeros_like(rhs) if depth > 1 else None
        wrapper._bind_captured_state(
            base, rhs.clone(), torch.zeros_like(rhs), step_output,
            first_output=first_output, chunk=1, capture_first=capture_first,
        )
        wrapper.step_graph = FakeStepReplay(wrapper)
        wrapper.first_graph = FakeFirstReplay(wrapper) if capture_first else None
        return base, rhs, wrapper

    def assert_original(self, base, rhs, wrapper):
        depth = base.depth
        actual = wrapper(rhs)
        expected = base(rhs).to(torch.complex128)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.dtype, torch.complex128)
        self.assertEqual(actual.shape, rhs.shape)
        self.assertEqual(base.depth, depth)
        return actual

    def reject_and_restore(self, base, rhs, wrapper, owner, name, replacement):
        original = getattr(owner, name)
        count = wrapper.step_graph.calls
        setattr(owner, name, replacement)
        try:
            with self.assertRaises(RuntimeError):
                wrapper(rhs)
            self.assertEqual(wrapper.step_graph.calls, count)
            with self.assertRaises(RuntimeError):
                wrapper.prepare()
        finally:
            setattr(owner, name, original)
        self.assert_original(base, rhs, wrapper)

    def test_cpu_constructor_and_unsupported_chunks_are_rejected(self):
        base, rhs, _ = self.fixture()
        with self.assertRaises(ValueError):
            SegmentedTSA(base, rhs, chunk=1, capture_first=False)
        for chunk in (0, 2, 4):
            with self.subTest(chunk=chunk):
                wrapper = object.__new__(SegmentedTSA)
                with self.assertRaises(ValueError):
                    wrapper._bind_captured_state(
                        base, rhs.clone(), torch.zeros_like(rhs), torch.zeros_like(rhs),
                        chunk=chunk, capture_first=False,
                    )

    def test_stage_functions_keep_original_zero_add_and_subsequent_arithmetic(self):
        base, rhs, _ = self.fixture()
        expected_first = torch.zeros_like(rhs) + base.relaxation * base.single(rhs)
        actual_first = SegmentedTSA._first_stage(base, rhs)
        torch.testing.assert_close(actual_first, expected_first, atol=0, rtol=0)
        self.assertNotEqual(actual_first.data_ptr(), rhs.data_ptr())
        saved = actual_first.clone()
        expected_step = actual_first + base.relaxation * base.single(rhs - base.rte.A_op(actual_first))
        actual_step = SegmentedTSA._subsequent_stage(base, rhs, actual_first)
        torch.testing.assert_close(actual_step, expected_step, atol=0, rtol=0)
        torch.testing.assert_close(actual_first, saved, atol=0, rtol=0)
        self.assertNotEqual(actual_step.data_ptr(), actual_first.data_ptr())

    def test_all_requested_depths_and_first_modes_match_original_exactly(self):
        for depth in (1, 4, 8, 16):
            for capture_first in (False, True):
                with self.subTest(depth=depth, capture_first=capture_first):
                    base, rhs, wrapper = self.fixture(depth, capture_first)
                    self.assert_original(base, rhs, wrapper)
                    self.assert_original(base, torch.zeros_like(rhs), wrapper)
                    self.assertEqual(wrapper.step_graph.calls, 2 * (depth - 1))
                    self.assertEqual(wrapper.first_graph.calls if capture_first else 0, 2 if capture_first else 0)

    def test_recipe_has_original_four_depth_minus_one_forward_and_inverse_ffts(self):
        for depth in (1, 4, 8, 16):
            for capture_first in (False, True):
                with self.subTest(depth=depth, capture_first=capture_first):
                    base, rhs, wrapper = self.fixture(depth, capture_first)
                    with patch.object(torch.fft, "fftn", wraps=torch.fft.fftn) as forward, patch.object(
                        torch.fft, "ifftn", wraps=torch.fft.ifftn
                    ) as inverse:
                        wrapper(rhs)
                        self.assertEqual(forward.call_count, 4 * depth - 1)
                        self.assertEqual(inverse.call_count, 4 * depth - 1)

    def test_each_rhs_resets_state_and_returned_outputs_keep_owned_storage(self):
        for capture_first in (False, True):
            with self.subTest(capture_first=capture_first):
                base, rhs, wrapper = self.fixture(8, capture_first)
                first = self.assert_original(base, rhs, wrapper)
                snapshot = first.clone()
                second_rhs = rhs * complex(.37, -.42)
                second = self.assert_original(base, second_rhs, wrapper)
                third = self.assert_original(base, rhs, wrapper)
                torch.testing.assert_close(first, snapshot, atol=0, rtol=0)
                torch.testing.assert_close(third, snapshot, atol=0, rtol=0)
                self.assertEqual(len({first.data_ptr(), second.data_ptr(), third.data_ptr()}), 3)
                for value in (first, second, third):
                    for name in ("static_r", "static_z", "step_output", "first_output"):
                        static = getattr(wrapper, name)
                        if static is not None:
                            self.assertNotEqual(value.data_ptr(), static.data_ptr())
                self.assertEqual(wrapper.step_graph.calls, 3 * 7)
                self.assertEqual(wrapper.first_graph.calls if capture_first else 0, 3 if capture_first else 0)

    def test_native_complex128_uses_original_without_any_replay_or_depth_change(self):
        for capture_first in (False, True):
            with self.subTest(capture_first=capture_first):
                base, rhs, wrapper = self.fixture(16, capture_first)
                native = rhs.to(torch.complex128) + (1e-10 + 2e-10j)
                torch.testing.assert_close(wrapper(native), base(native), atol=0, rtol=0)
                self.assertEqual(wrapper.step_graph.calls, 0)
                self.assertEqual(wrapper.first_graph.calls if capture_first else 0, 0)
                self.assertEqual(base.depth, 16)

    def test_public_aliases_and_prepare_preserve_baseline_benchmark_access(self):
        base, rhs, wrapper = self.fixture()
        self.assertIs(wrapper.reference, base.reference)
        self.assertIs(wrapper.rte, base.rte)
        self.assertEqual(wrapper.depth, base.depth)
        self.assertIs(wrapper.prepare(), wrapper)
        torch.testing.assert_close(wrapper.reference.reference_transport_solve(rhs),
                                   base.reference.reference_transport_solve(rhs), atol=0, rtol=0)
        self.assert_original(base, rhs, wrapper)

    def test_operator_and_cache_identity_layout_and_settings_changes_reject_then_restore(self):
        base, rhs, wrapper = self.fixture()
        for owner, name in (
            (base.rte, "sigma_t"), (base.rte, "sigma_s"),
            (base.reference, "phase_matrix"), (base.reference, "_reference_transport_inverse"),
        ):
            original = getattr(owner, name)
            for replacement in (original.view_as(original), original.clone(), original.transpose(-1, -2)):
                with self.subTest(tensor=name, stride=tuple(replacement.stride())):
                    self.reject_and_restore(base, rhs, wrapper, owner, name, replacement)
        for owner, name, replacement in (
            (base, "depth", 5), (base, "relaxation", .7), (base, "eta_scale", .2),
            (base.rte, "dx", base.rte.dx * 1.1),
            (base.reference, "eta_floor", .002),
            (base.rte, "setup_generation", base.rte.setup_generation + 1),
            (wrapper, "depth", 5),
        ):
            with self.subTest(setting=name):
                self.reject_and_restore(base, rhs, wrapper, owner, name, replacement)

    def test_static_state_and_outputs_reject_view_replacement_but_copy_versions_advance(self):
        base, rhs, wrapper = self.fixture(capture_first=True)
        versions = {name: getattr(wrapper, name)._version for name in
                    ("static_r", "static_z", "step_output", "first_output")}
        self.assert_original(base, rhs, wrapper)
        self.assert_original(base, rhs * .3, wrapper)
        for name, version in versions.items():
            self.assertGreater(getattr(wrapper, name)._version, version)
            with self.subTest(buffer=name):
                tensor = getattr(wrapper, name)
                self.reject_and_restore(base, rhs, wrapper, wrapper, name, tensor.transpose(-1, -2))

    def test_in_place_operator_mutation_blocks_both_precision_paths_and_prepare(self):
        base, rhs, wrapper = self.fixture()
        base.rte.sigma_s.add_(.001)
        for value in (rhs, rhs.to(torch.complex128)):
            with self.subTest(dtype=value.dtype):
                with self.assertRaises(RuntimeError):
                    wrapper(value)
        with self.assertRaises(RuntimeError):
            wrapper.prepare()
        self.assertEqual(wrapper.step_graph.calls, 0)

    def test_missing_inverse_is_rejected_without_rebuilding_and_strong_cache_is_retained(self):
        base, rhs, wrapper = self.fixture()
        original = base.reference._reference_transport_inverse
        retained = weakref.ref(original)
        base.reference._reference_transport_inverse = None
        with patch.object(base.reference, "_build_reference_transport_inverse", side_effect=AssertionError("guard rebuilt cache")) as rebuild:
            with self.assertRaises(RuntimeError):
                wrapper(rhs)
            rebuild.assert_not_called()
            self.assertIsNone(base.reference._reference_transport_inverse)
        base.reference._reference_transport_inverse = original.clone()
        del original
        gc.collect()
        self.assertIsNotNone(retained())
        with self.assertRaises(RuntimeError):
            wrapper(rhs)
        base.reference._reference_transport_inverse = retained()
        self.assert_original(base, rhs, wrapper)


if __name__ == "__main__":
    unittest.main(verbosity=2)
