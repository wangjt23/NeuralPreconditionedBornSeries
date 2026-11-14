"""CPU contracts for the frozen original TSA4 plus reference-defect action.

These N8/L12 fixtures are deterministic unit examples, not project sources or
held-out data. Fake replays test recipe/guard/ownership contracts without CUDA.
The original operator and capture helpers are imported without modification.
"""

from contextlib import ExitStack, contextmanager
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

import test_segmented_tsa as frozen
from direct_segmented_tsa import DirectSegmentedTSA
from direct_full_graph import _capture_full_action
from test_direct_full_graph import RawFullCaptureLifecycleTests
from matched_reference import ReferenceDefectAction, RawReferenceDefectGraph


class FakeCompoundReplay:
    def __init__(self, wrapper):
        self.wrapper, self.calls = wrapper, 0

    def replay(self):
        self.calls += 1
        self.wrapper.static_output.copy_(
            self.wrapper.base(self.wrapper.static_input)
        )


class MatchedReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def fixture(self, harmonics=None, segmented=False):
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
        action = ReferenceDefectAction(base, prefix, harmonics=harmonics)
        return base, rhs, prefix, action

    @staticmethod
    def literal(base, prefix, rhs, harmonics):
        # In particular, do not downcast a native128 field or retain the
        # segmented wrapper's widened64 output before the physical defect.
        g = prefix(rhs).to(rhs.dtype)
        defect = rhs - base.rte.A_op(g)
        correction = base.reference.reference_transport_solve(
            defect, harmonics=harmonics
        )
        return g + correction

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
        self.assert_exact(action(rhs), self.literal(base, base, rhs, action.harmonics))

    def graph_fixture(self, harmonics=None):
        base, rhs, prefix, action = self.fixture(harmonics)
        wrapper = object.__new__(RawReferenceDefectGraph)
        wrapper._bind_captured_state(action, rhs.clone(), action(rhs))
        wrapper.graph = FakeCompoundReplay(wrapper)
        return base, rhs, action, wrapper

    def test_full_and_q3_literal_recipe_parity_in_both_precisions_and_backends(self):
        for harmonics in (None, 1):
            for segmented in (False, True):
                base, rhs, prefix, action = self.fixture(harmonics, segmented)
                for dtype in (torch.complex64, torch.complex128):
                    with self.subTest(harmonics=harmonics, segmented=segmented, dtype=dtype):
                        value = rhs.to(dtype)
                        if dtype == torch.complex128:
                            value = value + complex(1e-10, -2e-10)
                        actual = action(value)
                        self.assertEqual(actual.dtype, dtype)
                        self.assert_exact(actual, self.literal(base, prefix, value, harmonics))
                        self.assertEqual(base.depth, 4)

    def test_segmented_prefix_backcast_matches_plain_original_rounding(self):
        for harmonics in (None, 1):
            base, rhs, prefix, action = self.fixture(harmonics, True)
            self.assertEqual(prefix(rhs).dtype, torch.complex128)
            self.assert_exact(action(rhs), self.literal(base, base, rhs, harmonics))
            self.assertEqual(prefix.step_graph.calls, 6)

    def test_original_recipe_counts_seventeen_spatial_fft_pairs_and_optional_angular_pair(self):
        for harmonics in (None, 1):
            for segmented in (False, True):
                _, rhs, _, action = self.fixture(harmonics, segmented)
                with self.subTest(harmonics=harmonics, segmented=segmented):
                    with patch.object(torch.fft, "fftn", wraps=torch.fft.fftn) as fftn, patch.object(
                        torch.fft, "ifftn", wraps=torch.fft.ifftn
                    ) as ifftn, patch.object(torch.fft, "fft", wraps=torch.fft.fft) as fft, patch.object(
                        torch.fft, "ifft", wraps=torch.fft.ifft
                    ) as ifft:
                        action(rhs)
                    self.assertEqual(fftn.call_count, 17)
                    self.assertEqual(ifftn.call_count, 17)
                    self.assertEqual(fft.call_count, int(harmonics == 1))
                    self.assertEqual(ifft.call_count, int(harmonics == 1))
                    self.assertEqual(action.application_fft_counts,
                                     {"fftn": 17, "ifftn": 17,
                                      "fft": int(harmonics == 1), "ifft": int(harmonics == 1)})

    def test_full_angles_and_spatial_nyquist_are_retained_in_q3_response(self):
        base, rhs, _, action = self.fixture(1)
        actual = action(rhs)
        self.assertEqual(actual.shape, (1, 12, 8, 8))
        spectrum = torch.fft.fftn(actual, dim=(-2, -1))
        self.assertGreater(float(spectrum[:, :, 4, :].norm()), 1e-4)
        self.assertGreater(float(spectrum[:, :, :, 4].norm()), 1e-4)
        # The projector limits the input to R0, not the output of R0.
        spatial = torch.exp(2j * torch.pi * torch.arange(8) / 8)[:, None].expand(8, 8)
        retained = torch.ones_like(rhs) * spatial
        response = base.reference.reference_transport_solve(retained, harmonics=1)
        outside = response - base.reference.angular_project(response, harmonics=1)
        self.assertGreater(float(outside.norm() / response.norm()), .1)
        angles = torch.arange(12, dtype=torch.float64) * (2 * torch.pi / 12)
        omitted = torch.exp(2j * angles).to(rhs.dtype)[None, :, None, None] * spatial
        discarded = base.reference.reference_transport_solve(omitted, harmonics=1)
        full = base.reference.reference_transport_solve(omitted)
        self.assertLess(float(discarded.norm() / full.norm()), 2e-6)

    def test_uniform_weighted_projection_identity_and_nonuniform_counterexample(self):
        base, _, _, _ = self.fixture(1)
        reference = base.reference
        basis = reference.angular_fourier_basis(1, dtype=torch.complex128)
        weights = torch.diag(reference.weights.to(torch.complex128))
        gram = basis.conj().T @ weights @ basis
        manual = basis @ torch.linalg.solve(gram, basis.conj().T @ weights)
        eye = torch.eye(12, dtype=torch.complex128).reshape(1, 12, 12, 1)
        projected = reference.angular_project(eye, 1)[0, :, :, 0]
        torch.testing.assert_close(projected, manual, atol=2e-13, rtol=2e-13)
        basis64 = reference.angular_fourier_basis(1, dtype=torch.complex64)
        weights64 = torch.diag(reference.weights.to(torch.complex64))
        manual64 = basis64 @ torch.linalg.solve(
            basis64.conj().T @ weights64 @ basis64, basis64.conj().T @ weights64
        )
        projected64 = reference.angular_project(eye.to(torch.complex64), 1)[0, :, :, 0]
        torch.testing.assert_close(projected64, manual64, atol=2e-6, rtol=2e-6)
        angle = torch.arange(12, dtype=torch.float64) * (2 * torch.pi / 12)
        nonuniform = torch.diag(((1 + .35 * torch.cos(2 * angle)) / 12).to(torch.complex128))
        weighted = basis @ torch.linalg.solve(
            basis.conj().T @ nonuniform @ basis, basis.conj().T @ nonuniform
        )
        self.assertGreater(float((weighted - projected).abs().max()), .01)

    def test_linear_zero_phase_and_material_batch_broadcast_in_original_precision(self):
        for harmonics in (None, 1):
            _, rhs, _, action = self.fixture(harmonics)
            for dtype, tolerance in ((torch.complex64, 5e-6), (torch.complex128, 3e-13)):
                with self.subTest(harmonics=harmonics, dtype=dtype):
                    value = rhs.to(dtype)
                    other = value.flip(-1) * complex(.23, -.17)
                    first, second = action(value), action(other)
                    additive = action(value + other)
                    self.assertLess(float((additive - first - second).norm() / (first.norm() + second.norm())), tolerance)
                    phase = complex(.37, -.42)
                    phased = action(phase * value)
                    self.assertLess(float((phased - phase * first).norm() / first.norm()), tolerance)
                    zero = action(torch.zeros_like(value))
                    self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))
                    batch = action(torch.cat((value, other), dim=0))
                    self.assertEqual(batch.shape[0], 2)
                    torch.testing.assert_close(batch, torch.cat((first, second), dim=0), atol=tolerance, rtol=tolerance)

    def test_outputs_and_owned128_do_not_alias_inputs_or_later_applications(self):
        for harmonics in (None, 1):
            _, rhs, prefix, action = self.fixture(harmonics, True)
            first, wide = action(rhs), action.owned128(rhs)
            saved, wide_saved = first.clone(), wide.clone()
            second = action(rhs * complex(.2, .3))
            second_wide = action.owned128(rhs.to(torch.complex128))
            self.assertEqual(wide.dtype, torch.complex128)
            self.assertEqual(second_wide.dtype, torch.complex128)
            self.assert_exact(wide, saved.to(torch.complex128))
            self.assert_exact(first, saved)
            self.assert_exact(wide, wide_saved)
            outputs = (first, wide, second, second_wide)
            self.assertEqual(len({value.data_ptr() for value in outputs}), len(outputs))
            for value in outputs:
                for static in (rhs, prefix.static_r, prefix.static_z, prefix.step_output):
                    self.assertNotEqual(value.untyped_storage().data_ptr(), static.untyped_storage().data_ptr())

    def test_native128_uses_promoted_existing_cache_without_reinversion(self):
        base, rhs, _, action = self.fixture(1)
        cache = base.reference._reference_transport_inverse
        pointer, version = cache.data_ptr(), cache._version
        native = rhs.to(torch.complex128) + complex(1e-10, -2e-10)
        with patch.object(base.reference, "_build_reference_transport_inverse", side_effect=AssertionError("cache rebuilt")):
            actual = action(native)
            self.assert_exact(actual, self.literal(base, base, native, 1))
        self.assertIs(base.reference._reference_transport_inverse, cache)
        self.assertEqual((cache.data_ptr(), cache._version, cache.dtype), (pointer, version, torch.complex64))
        rebuilt = torch.linalg.inv(base.reference._reference_transport_matrix().to(torch.complex128))
        self.assertGreater(float((cache.to(torch.complex128) - rebuilt).norm() / rebuilt.norm()), 1e-10)

    def test_constructor_rejects_other_depth_modes_backends_and_nonuniform_weights(self):
        for mode in (0, 2, -1, True, "1"):
            base, _, _, _ = self.fixture()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                ReferenceDefectAction(base, base, harmonics=mode)
        base, rhs, _, _ = self.fixture()
        for depth in (1, 8):
            original = base.depth
            base.depth = depth
            try:
                with self.assertRaises(ValueError):
                    ReferenceDefectAction(base, base)
            finally:
                base.depth = original
        other, _, _, _ = self.fixture()
        with self.assertRaises(ValueError):
            ReferenceDefectAction(base, other)
        with self.assertRaises(ValueError):
            ReferenceDefectAction(base, lambda value: value)
        for operator in (base.rte, base.reference):
            original = operator.weights
            operator.weights = original * torch.linspace(.8, 1.2, 12)
            try:
                with self.assertRaises(ValueError):
                    ReferenceDefectAction(base, base, harmonics=1)
            finally:
                operator.weights = original
        original_dtype = base.rte.float_dtype
        base.rte.float_dtype = torch.float64
        try:
            with self.assertRaises(ValueError):
                ReferenceDefectAction(base, base)
        finally:
            base.rte.float_dtype = original_dtype
        original_cache = base.reference._reference_transport_inverse
        base.reference._reference_transport_inverse = original_cache.to(torch.complex128)
        try:
            with self.assertRaises(ValueError):
                ReferenceDefectAction(base, base)
        finally:
            base.reference._reference_transport_inverse = original_cache

    def test_constructor_preserves_prepared_cache_and_rejects_stale_or_missing_state_without_refresh(self):
        for scenario in ("valid", "stale_sigma_s", "missing_inverse", "mismatched_grid"):
            base, _, _ = frozen.SegmentedTSATests().fixture(4, False)
            original_cache = base.reference._reference_transport_inverse
            cache_state = (original_cache.data_ptr(), original_cache._version)
            original_signature = base._signature
            if scenario == "stale_sigma_s":
                # Equal values still have a new mutation version; constructor
                # must not call prepare to silently synchronize that material.
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
                        action = ReferenceDefectAction(base, base, harmonics=1)
                        self.assertIs(action.reference._reference_transport_inverse, original_cache)
                        self.assertIs(action.prepare(), action)
                    else:
                        with self.assertRaises(ValueError):
                            ReferenceDefectAction(base, base, harmonics=1)
                    prepare.assert_not_called()
                    rebuild.assert_not_called()
                self.assertIs(base.reference._reference_transport_inverse, expected_cache)
                self.assertIs(base._signature, original_signature)
                self.assertEqual((original_cache.data_ptr(), original_cache._version), cache_state)

    def test_frozen_mode_backend_settings_and_original_tensor_replacements_are_rejected(self):
        base, rhs, _, action = self.fixture(1)
        other, _, _, _ = self.fixture()
        replacements = [(action, "harmonics", None), (action, "mode", "full"),
                        (action, "prefix_backend", other), (action, "base4", other),
                        (action, "rte", other.rte), (action, "reference", other.reference)]
        replacements.extend((base, name, value) for name, value in (
            ("depth", 3), ("relaxation", .9), ("eta_scale", .2), ("kind", "reference")
        ))
        for operator in (base.rte, base.reference):
            for name in ("sigma_t", "sigma_s", "directions", "weights", "phase_matrix", "kx", "ky"):
                tensor = getattr(operator, name)
                replacements.extend(((operator, name, tensor.clone()), (operator, name, tensor.view_as(tensor))))
            replacements.append((operator, "setup_generation", operator.setup_generation + 1))
        cache = base.reference._reference_transport_inverse
        replacements.extend(((base.reference, "_reference_transport_inverse", cache.clone()),
                             (base.reference, "_reference_transport_inverse", cache.transpose(-1, -2)),
                             (base.rte, "sigma_t", base.rte.sigma_t.transpose(-1, -2))))
        for name in ("sigma0", "sigma_s_ref"):
            tensor = getattr(base.reference, name)
            replacements.append((base.reference, name, tensor.clone()))
        for owner, name, replacement in replacements:
            with self.subTest(owner=type(owner).__name__, name=name):
                self.reject_restore(base, rhs, action, owner, name, replacement)

    def test_inplace_original_changes_and_missing_cache_reject_without_refresh(self):
        for name in ("weights", "sigma_t", "_reference_transport_inverse"):
            base, rhs, _, action = self.fixture(1)
            tensor = getattr(base.reference, name)
            tensor.add_(0)
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError):
                    action.prepare()
                with self.assertRaises(RuntimeError):
                    action(rhs)
        base, rhs, _, action = self.fixture()
        cache = base.reference._reference_transport_inverse
        base.reference._reference_transport_inverse = None
        try:
            with patch.object(base.reference, "_build_reference_transport_inverse", side_effect=AssertionError("cache rebuilt")):
                with self.assertRaises(RuntimeError):
                    action.prepare()
                with self.assertRaises(RuntimeError):
                    action(rhs.to(torch.complex128))
        finally:
            base.reference._reference_transport_inverse = cache
        self.assert_exact(action(rhs), self.literal(base, base, rhs, None))

    def test_segmented_backend_graph_and_static_layout_are_frozen_while_copy_versions_advance(self):
        for harmonics in (None, 1):
            base, rhs, prefix, action = self.fixture(harmonics, True)
            version = prefix.static_z._version
            self.assert_exact(action(rhs), self.literal(base, base, rhs, harmonics))
            self.assert_exact(action(rhs * .3), self.literal(base, base, rhs * .3, harmonics))
            self.assertGreater(prefix.static_z._version, version)
            self.assertIs(action.prepare(), action)
            replacements = [("step_graph", frozen.FakeStepReplay(prefix)),
                            ("static_r", prefix.static_r.view_as(prefix.static_r)),
                            ("static_z", prefix.static_z.transpose(-1, -2)),
                            ("step_output", prefix.step_output.clone())]
            for name, replacement in replacements:
                with self.subTest(harmonics=harmonics, name=name):
                    self.reject_restore(base, rhs, action, prefix, name, replacement)

    def test_field_validation_and_whole_capture_require_plain_prefix_and_independent_buffers(self):
        _, rhs, _, action = self.fixture(1)
        for value in (rhs.real, rhs[:, :3], rhs[:, :, :4], rhs[:0], rhs[0]):
            with self.subTest(shape=tuple(value.shape), dtype=value.dtype), self.assertRaises(ValueError):
                action(value)
        _, _, _, segmented = self.fixture(1, True)
        wrapper = object.__new__(RawReferenceDefectGraph)
        with self.assertRaises(ValueError):
            wrapper._bind_captured_state(segmented, rhs.clone(), torch.zeros_like(rhs))
        static_input = rhs.clone()
        for output in (static_input, torch.zeros_like(rhs).to(torch.complex128), torch.zeros_like(rhs[:, :3])):
            wrapper = object.__new__(RawReferenceDefectGraph)
            with self.assertRaises(ValueError):
                wrapper._bind_captured_state(action, static_input, output)

    def test_raw_graph_cpu_rejection_compound_replay_native_fallback_and_owned_outputs(self):
        for harmonics in (None, 1):
            base, rhs, action, wrapper = self.graph_fixture(harmonics)
            with self.assertRaises(ValueError):
                RawReferenceDefectGraph(action, rhs)
            first = wrapper(rhs)
            self.assert_exact(first, action(rhs).to(torch.complex128))
            saved = first.clone()
            second = wrapper(rhs * complex(.3, -.2))
            self.assert_exact(first, saved)
            self.assertNotEqual(first.data_ptr(), second.data_ptr())
            self.assertNotEqual(first.data_ptr(), wrapper.static_output.data_ptr())
            native = rhs.to(torch.complex128) + complex(1e-10, -2e-10)
            count = wrapper.graph.calls
            native_output = wrapper(native)
            self.assert_exact(native_output, action(native))
            self.assertEqual(wrapper.graph.calls, count)
            self.assertGreater(float((native_output - base(native)).norm()), 1e-6)

    def test_raw_graph_guards_compound_before_replay_or_native_fallback(self):
        for harmonics in (None, 1):
            _, rhs, action, wrapper = self.graph_fixture(harmonics)
            original = action.harmonics
            action.harmonics = 1 if original is None else None
            try:
                for value in (rhs, rhs.to(torch.complex128)):
                    with self.assertRaises(RuntimeError):
                        wrapper(value)
                    self.assertEqual(wrapper.graph.calls, 0)
            finally:
                action.harmonics = original
            self.assert_exact(wrapper(rhs), action(rhs).to(torch.complex128))
            original_output = wrapper.static_output
            wrapper.static_output = original_output.view_as(original_output)
            try:
                with self.assertRaises(RuntimeError):
                    wrapper(rhs)
            finally:
                wrapper.static_output = original_output
            self.assert_exact(wrapper(rhs), action(rhs).to(torch.complex128))

    def test_raw_capture_body_calls_compound_action_rather_than_prefix(self):
        _, rhs, _, action = self.fixture(1)
        expected = action(rhs)
        events, active = [], {"stream": False}
        stream = object()
        test = self

        @contextmanager
        def selected_stream(value):
            test.assertIs(value, stream)
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

        def compound(selected, value):
            test.assertIs(selected, action)
            test.assertIs(value, rhs)
            test.assertTrue(active["stream"])
            test.assertFalse(torch.is_grad_enabled())
            events.append("compound")
            return expected

        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.cuda, "stream", side_effect=selected_stream))
            call = stack.enter_context(patch.object(ReferenceDefectAction, "__call__", autospec=True, side_effect=compound))
            forbidden = [stack.enter_context(patch(target, side_effect=AssertionError(target))) for target in
                         ("torch.cuda.graph", "gc.collect", "torch.cuda.empty_cache", "torch.cuda.synchronize")]
            self.assertIs(_capture_full_action(Graph(), action, rhs, stream), expected)
            call.assert_called_once_with(action, rhs)
            for operation in forbidden:
                operation.assert_not_called()
        self.assertEqual(events, ["enter", "begin", "compound", "end", "exit"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MatchedReferenceTests)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RawFullCaptureLifecycleTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
