"""CPU recipe/guard tests for the single frozen gain-two classical action.

N8/L12 fixtures are deterministic unit examples. Fake graph replays and mocked
streams exercise storage and capture contracts without CUDA or timing claims.
For a flat stage, include the unchanged v2 helper as matched_reference_v2.py.
"""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_ROOT = ROOT / "remote_reference"
if not (SOURCE_ROOT / "rte" / "models.py").is_file():
    SOURCE_ROOT = Path.cwd() if (Path.cwd() / "rte" / "models.py").is_file() else ROOT
sys.path.insert(0, str(SOURCE_ROOT))
for directory in ("headroom", "segmented", "direct_capture"):
    sys.path.insert(0, str(HERE.parent / directory))
sys.path.insert(0, str(HERE))


def load_matched_v2():
    path = HERE / "matched_reference_v2.py"
    if not path.is_file():
        path = HERE.parent / "full_angle_v2" / "matched_reference.py"
    if not path.is_file():
        raise FileNotFoundError("stage frozen v2 helper as matched_reference_v2.py")
    spec = importlib.util.spec_from_file_location("matched_reference_v2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MATCHED_V2 = load_matched_v2()
import test_segmented_tsa as frozen
from direct_segmented_tsa import DirectSegmentedTSA
from direct_full_graph import _capture_full_action
from test_direct_full_graph import RawFullCaptureLifecycleTests
from gain_two_reference import GainTwoReferenceAction, RawGainTwoReferenceGraph


class FakeGainReplay:
    def __init__(self, wrapper):
        self.wrapper, self.calls = wrapper, 0

    def replay(self):
        self.calls += 1
        self.wrapper.static_output.copy_(self.wrapper.base(self.wrapper.static_input))


class GainTwoReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def fixture(self, segmented=False):
        base, rhs, _ = frozen.SegmentedTSATests().fixture(4, False)
        prefix = base
        if segmented:
            prefix = object.__new__(DirectSegmentedTSA)
            prefix._bind_captured_state(
                base, rhs.clone(), torch.zeros_like(rhs), torch.zeros_like(rhs),
                first_output=None, chunk=1, capture_first=False,
            )
            prefix.step_graph = frozen.FakeStepReplay(prefix)
            prefix.first_graph = None
        return base, rhs, prefix, GainTwoReferenceAction(base, prefix)

    @staticmethod
    def literal(base, prefix, rhs):
        g = prefix(rhs).to(rhs.dtype)
        defect = rhs - base.rte.A_op(g)
        # Use the original projector and perform the combination BEFORE R0.
        projected = base.reference.angular_project(defect, harmonics=1)
        weighted = 2 * defect - projected
        return g + base.reference.reference_transport_solve(weighted, harmonics=None)

    def assert_exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def reject_restore(self, base, rhs, action, owner, name, replacement):
        original = getattr(owner, name)
        setattr(owner, name, replacement)
        try:
            with self.assertRaises(RuntimeError):
                action.prepare()
            for value in (rhs, rhs.to(torch.complex128)):
                with self.assertRaises(RuntimeError):
                    action(value)
        finally:
            setattr(owner, name, original)
        self.assert_exact(action(rhs), self.literal(base, base, rhs))

    def graph_fixture(self):
        base, rhs, _, action = self.fixture()
        wrapper = object.__new__(RawGainTwoReferenceGraph)
        wrapper._bind_captured_state(action, rhs.clone(), action(rhs))
        wrapper.graph = FakeGainReplay(wrapper)
        return base, rhs, action, wrapper

    def test_literal_original_defect_weighting_parity_both_precisions_and_backends(self):
        for segmented in (False, True):
            base, rhs, prefix, action = self.fixture(segmented)
            for dtype in (torch.complex64, torch.complex128):
                with self.subTest(segmented=segmented, dtype=dtype):
                    value = rhs.to(dtype)
                    if dtype == torch.complex128:
                        value = value + complex(1e-10, -2e-10)
                    actual = action(value)
                    self.assertEqual(actual.dtype, dtype)
                    self.assert_exact(actual, self.literal(base, prefix, value))
                    self.assertEqual(base.depth, 4)

    def test_original_reference_receives_one_weighted_defect_with_no_output_reassociation(self):
        base, rhs, _, action = self.fixture()
        g = base(rhs)
        defect = rhs - base.rte.A_op(g)
        projected = base.reference.angular_project(defect, harmonics=1)
        weighted = 2 * defect - projected
        with patch.object(base.reference, "reference_transport_solve", wraps=base.reference.reference_transport_solve) as solve:
            actual = action(rhs)
        self.assertEqual(solve.call_count, 5)  # Four prefix stages plus one correction.
        self.assert_exact(solve.call_args.args[0], weighted)
        self.assertEqual(solve.call_args.kwargs, {"harmonics": None})
        full_image = base.reference.reference_transport_solve(defect)
        projected_image = base.reference.reference_transport_solve(projected)
        reassociated = g + 2 * full_image - projected_image
        self.assertFalse(torch.equal(actual, reassociated))
        self.assert_exact(actual, self.literal(base, base, rhs))

    def test_cached_projector_matches_original_and_fixed_gain_differs_from_v2_q3(self):
        base, rhs, prefix, action = self.fixture()
        self.assertEqual(action.beta, 2)
        self.assertEqual(action.mode, "gain_two")
        for dtype in (torch.complex64, torch.complex128):
            value = rhs.to(dtype)
            self.assert_exact(action.project_q3(value), base.reference.angular_project(value, 1))
        old_q3 = MATCHED_V2.ReferenceDefectAction(base, prefix, harmonics=1)
        self.assertGreater(float((action(rhs) - old_q3(rhs)).norm()), 1e-5)

    def test_original_recipe_has_seventeen_spatial_and_one_angular_fft_pairs(self):
        for segmented in (False, True):
            _, rhs, _, action = self.fixture(segmented)
            for dtype in (torch.complex64, torch.complex128):
                with self.subTest(segmented=segmented, dtype=dtype):
                    with patch.object(torch.fft, "fftn", wraps=torch.fft.fftn) as fftn, patch.object(
                        torch.fft, "ifftn", wraps=torch.fft.ifftn
                    ) as ifftn, patch.object(torch.fft, "fft", wraps=torch.fft.fft) as fft, patch.object(
                        torch.fft, "ifft", wraps=torch.fft.ifft
                    ) as ifft:
                        action(rhs.to(dtype))
                    self.assertEqual((fftn.call_count, ifftn.call_count, fft.call_count, ifft.call_count), (17, 17, 1, 1))
                    self.assertEqual(action.application_fft_counts, {"fftn": 17, "ifftn": 17, "fft": 1, "ifft": 1})

    def test_segmented_backcast_and_native128_fallback_retain_original_rounding(self):
        base, rhs, prefix, action = self.fixture(True)
        self.assertEqual(prefix(rhs).dtype, torch.complex128)
        self.assert_exact(action(rhs), self.literal(base, base, rhs))
        self.assertEqual(prefix.step_graph.calls, 6)
        count = prefix.step_graph.calls
        value = rhs.to(torch.complex128) + complex(1e-10, -2e-10)
        self.assert_exact(action(value), self.literal(base, base, value))
        self.assertEqual(prefix.step_graph.calls, count)

    def test_zero_linearity_phase_and_material_batch_broadcast(self):
        for segmented in (False, True):
            _, rhs, _, action = self.fixture(segmented)
            for dtype, tolerance in ((torch.complex64, 6e-6), (torch.complex128, 4e-13)):
                with self.subTest(segmented=segmented, dtype=dtype):
                    value = rhs.to(dtype)
                    other = value.flip(-1) * complex(.23, -.17)
                    first, second = action(value), action(other)
                    self.assertLess(float((action(value + other) - first - second).norm() / (first.norm() + second.norm())), tolerance)
                    phase = complex(.37, -.42)
                    self.assertLess(float((action(phase * value) - phase * first).norm() / first.norm()), tolerance)
                    zero = action(torch.zeros_like(value))
                    self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))
                    if not segmented:
                        batch = action(torch.cat((value, other), dim=0))
                        torch.testing.assert_close(batch, torch.cat((first, second), dim=0), atol=tolerance, rtol=tolerance)

    def test_mask_and_original_inverse_are_retained_without_builds_writes_or_refresh(self):
        for segmented in (False, True):
            base, rhs, _, action = self.fixture(segmented)
            mask, cache = action.q3_mask, base.reference._reference_transport_inverse
            mask_state, cache_state = (mask.data_ptr(), mask._version), (cache.data_ptr(), cache._version)
            mask_values = mask.clone()
            originals = {dtype: self.literal(base, base, rhs.to(dtype)) for dtype in (torch.complex64, torch.complex128)}
            with patch.object(torch, "zeros", side_effect=AssertionError("mask rebuilt")) as zeros, patch.object(
                base.reference, "_build_reference_transport_inverse", side_effect=AssertionError("inverse rebuilt")
            ) as rebuild:
                for dtype in (torch.complex64, torch.complex128):
                    self.assert_exact(action(rhs.to(dtype)), originals[dtype])
                zeros.assert_not_called()
                rebuild.assert_not_called()
            self.assertIs(action.q3_mask, mask)
            self.assertIs(base.reference._reference_transport_inverse, cache)
            self.assertEqual((mask.data_ptr(), mask._version), mask_state)
            self.assertEqual((cache.data_ptr(), cache._version), cache_state)
            self.assertEqual(cache.dtype, torch.complex64)
            self.assert_exact(mask, mask_values)
            self.assertTrue(any(value is mask for value in action._captured_tensors))

    def test_owned_outputs_remain_independent_across_rhs_and_native_wide_calls(self):
        for segmented in (False, True):
            _, rhs, prefix, action = self.fixture(segmented)
            first = action(rhs)
            wide = action.owned128(rhs)
            saved, saved_wide = first.clone(), wide.clone()
            second = action(rhs * complex(.2, .3))
            native = action.owned128(rhs.to(torch.complex128))
            self.assert_exact(first, saved)
            self.assert_exact(wide, saved_wide)
            self.assert_exact(wide, first.to(torch.complex128))
            self.assertEqual(native.dtype, torch.complex128)
            values = (first, wide, second, native)
            self.assertEqual(len({value.data_ptr() for value in values}), len(values))
            for value in values:
                self.assertNotEqual(value.untyped_storage().data_ptr(), rhs.untyped_storage().data_ptr())
                if segmented:
                    for static in (prefix.static_r, prefix.static_z, prefix.step_output):
                        self.assertNotEqual(value.untyped_storage().data_ptr(), static.untyped_storage().data_ptr())

    def test_constructor_accepts_only_fixed_gain_and_preserves_prepared_original_cache(self):
        for kwargs in ({"beta": 2}, {"gain": 2}, {"harmonics": 1}, {"depth": 4}, {"gain": 1.5}):
            base, _, _, _ = self.fixture()
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                GainTwoReferenceAction(base, base, **kwargs)
        base, _, _, _ = self.fixture()
        with self.assertRaises(TypeError):
            GainTwoReferenceAction(base, base, 2)
        for scenario in ("valid", "stale_sigma_s", "missing_inverse", "mismatched_grid"):
            base, _, _ = frozen.SegmentedTSATests().fixture(4, False)
            cache = base.reference._reference_transport_inverse
            state, signature = (cache.data_ptr(), cache._version), base._signature
            if scenario == "stale_sigma_s":
                base.rte.sigma_s.add_(0)
            elif scenario == "missing_inverse":
                base.reference._reference_transport_inverse = None
            elif scenario == "mismatched_grid":
                base.rte.dx *= 1.1
            expected_cache = base.reference._reference_transport_inverse
            with self.subTest(scenario=scenario):
                with patch.object(base, "prepare", side_effect=AssertionError("base refreshed")) as prepare, patch.object(
                    base.reference, "_build_reference_transport_inverse", side_effect=AssertionError("inverse rebuilt")
                ) as rebuild:
                    if scenario == "valid":
                        action = GainTwoReferenceAction(base, base)
                        self.assertIs(action.prepare(), action)
                    else:
                        with self.assertRaises(ValueError):
                            GainTwoReferenceAction(base, base)
                    prepare.assert_not_called()
                    rebuild.assert_not_called()
                self.assertIs(base.reference._reference_transport_inverse, expected_cache)
                self.assertIs(base._signature, signature)
                self.assertEqual((cache.data_ptr(), cache._version), state)

    def test_beta_mode_harmonics_backend_and_original_settings_are_frozen(self):
        base, rhs, _, action = self.fixture()
        other, _, _, _ = self.fixture()
        replacements = [(action, "beta", 1), (action, "beta", 3), (action, "beta", 2.0),
                        (action, "mode", "q3"), (action, "mode", "full"),
                        (action, "harmonics", None), (action, "prefix_backend", other),
                        (base, "depth", 3), (base, "relaxation", .9), (base, "eta_scale", .2)]
        for operator in (base.rte, base.reference):
            for name in ("weights", "sigma_t", "phase_matrix", "directions"):
                tensor = getattr(operator, name)
                replacements.append((operator, name, tensor.clone()))
            replacements.append((operator, "setup_generation", operator.setup_generation + 1))
        for owner, name, replacement in replacements:
            with self.subTest(owner=type(owner).__name__, name=name, replacement=str(replacement) if name == "beta" else "guard"):
                self.reject_restore(base, rhs, action, owner, name, replacement)

    def test_mask_layout_mutation_and_missing_cache_reject_without_refresh(self):
        base, rhs, _, action = self.fixture()
        mask = action.q3_mask
        noncontiguous = torch.empty(24, dtype=mask.dtype)[::2]
        noncontiguous.copy_(mask)
        offset = torch.empty(13, dtype=mask.dtype)[1:]
        offset.copy_(mask)
        for replacement in (None, mask.clone(), mask.view_as(mask), noncontiguous, offset):
            self.reject_restore(base, rhs, action, action, "q3_mask", replacement)
        cache = base.reference._reference_transport_inverse
        with patch.object(base.reference, "_build_reference_transport_inverse", side_effect=AssertionError("cache rebuilt")) as rebuild:
            self.reject_restore(base, rhs, action, base.reference, "_reference_transport_inverse", None)
            rebuild.assert_not_called()
        for owner_name, tensor_name in (("action", "q3_mask"), ("reference", "_reference_transport_inverse")):
            base, rhs, _, action = self.fixture()
            owner = action if owner_name == "action" else base.reference
            getattr(owner, tensor_name).add_(0)
            with self.assertRaises(RuntimeError):
                action.prepare()
            with self.assertRaises(RuntimeError):
                action(rhs.to(torch.complex128))

    def test_segmented_backend_graph_and_static_state_identity_are_frozen(self):
        base, rhs, prefix, action = self.fixture(True)
        action(rhs)
        self.assertIs(action.prepare(), action)
        for name, replacement in (("step_graph", frozen.FakeStepReplay(prefix)),
                                  ("static_r", prefix.static_r.view_as(prefix.static_r)),
                                  ("static_z", prefix.static_z.transpose(-1, -2))):
            with self.subTest(name=name):
                self.reject_restore(base, rhs, action, prefix, name, replacement)

    def test_raw_cpu_constructor_binder_and_input_contracts(self):
        _, rhs, _, action = self.fixture()
        with self.assertRaises(ValueError):
            RawGainTwoReferenceGraph(action, rhs)
        for value in (rhs.real, rhs[:, :3], rhs[:, :, :4], rhs[:0], rhs[0]):
            with self.assertRaises(ValueError):
                action(value)
        _, _, _, segmented = self.fixture(True)
        wrapper = object.__new__(RawGainTwoReferenceGraph)
        with self.assertRaises(ValueError):
            wrapper._bind_captured_state(segmented, rhs.clone(), torch.zeros_like(rhs))
        static_input = rhs.clone()
        for output in (static_input, torch.zeros_like(rhs).to(torch.complex128), torch.zeros_like(rhs[:, :3])):
            wrapper = object.__new__(RawGainTwoReferenceGraph)
            with self.assertRaises(ValueError):
                wrapper._bind_captured_state(action, static_input, output)

    def test_raw_replay_and_native_fallback_call_owned_gain_compound(self):
        base, rhs, action, wrapper = self.graph_fixture()
        first = wrapper(rhs)
        self.assert_exact(first, action(rhs).to(torch.complex128))
        saved = first.clone()
        second = wrapper(rhs * complex(.3, -.2))
        self.assert_exact(first, saved)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertNotEqual(first.data_ptr(), wrapper.static_output.data_ptr())
        count = wrapper.graph.calls
        native = rhs.to(torch.complex128) + complex(1e-10, -2e-10)
        actual = wrapper(native)
        self.assert_exact(actual, self.literal(base, base, native))
        self.assertEqual(wrapper.graph.calls, count)
        self.assertGreater(float((actual - base(native)).norm()), 1e-6)

    def test_raw_guards_gain_mode_mask_and_layout_before_replay_or_native_fallback(self):
        _, rhs, action, wrapper = self.graph_fixture()
        for owner, name, replacement in ((action, "beta", 1), (action, "beta", 2.0),
                                         (action, "mode", "q3"), (wrapper, "beta", 1),
                                         (wrapper, "beta", 2.0), (wrapper, "mode", "q3"),
                                         (action, "q3_mask", action.q3_mask.clone()),
                                         (wrapper, "static_output", wrapper.static_output.view_as(wrapper.static_output))):
            original = getattr(owner, name)
            count = wrapper.graph.calls
            setattr(owner, name, replacement)
            try:
                with self.assertRaises(RuntimeError):
                    wrapper.prepare()
                for value in (rhs, rhs.to(torch.complex128)):
                    with self.assertRaises(RuntimeError):
                        wrapper(value)
                    self.assertEqual(wrapper.graph.calls, count)
            finally:
                setattr(owner, name, original)
            self.assert_exact(wrapper(rhs), action(rhs).to(torch.complex128))

    def test_raw_capture_calls_gain_compound_and_preserves_body_error_over_end_error(self):
        _, rhs, _, action = self.fixture()
        stream, events, active = object(), [], {"stream": False}
        initial, ending = RuntimeError("gain body sentinel"), RuntimeError("gain end sentinel")
        test = self

        @contextmanager
        def stream_context(selected):
            test.assertIs(selected, stream)
            active["stream"] = True
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")
                active["stream"] = False

        class Graph:
            def capture_begin(self, **kwargs):
                test.assertTrue(active["stream"])
                test.assertFalse(torch.is_grad_enabled())
                test.assertEqual(kwargs, {"capture_error_mode": "global"})
                events.append("begin")

            def capture_end(self):
                test.assertTrue(active["stream"])
                test.assertFalse(torch.is_grad_enabled())
                events.append("end")
                raise ending

        def compound(selected, value):
            test.assertIs(selected, action)
            test.assertIs(value, rhs)
            test.assertTrue(active["stream"])
            test.assertFalse(torch.is_grad_enabled())
            events.append("compound")
            raise initial

        with patch.object(torch.cuda, "stream", side_effect=stream_context), patch.object(
            GainTwoReferenceAction, "__call__", autospec=True, side_effect=compound
        ) as call:
            with self.assertRaises(RuntimeError) as caught:
                _capture_full_action(Graph(), action, rhs, stream)
            self.assertIs(caught.exception, initial)
            self.assertIs(caught.exception.__cause__, ending)
            call.assert_called_once_with(action, rhs)
        self.assertEqual(events, ["enter", "begin", "compound", "end", "exit"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GainTwoReferenceTests)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RawFullCaptureLifecycleTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
