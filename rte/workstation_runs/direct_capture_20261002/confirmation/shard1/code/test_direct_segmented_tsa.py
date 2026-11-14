"""CPU contracts and mocked raw-capture lifecycle for DirectSegmentedTSA.

The frozen segmented tests supply the physical fixture and shared guard checks.
Only the supported eager-first mode is exercised here. Fake replays and mocked
CUDA stream contexts perform no CUDA work and establish no performance result.
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
sys.path.insert(0, str(HERE.parent / "segmented"))
sys.path.insert(0, str(HERE))

import test_segmented_tsa as frozen
import direct_segmented_tsa as direct


class DirectNumericalTests(frozen.SegmentedTSATests):
    """Retain all eleven original facets with the direct subclass fixture."""

    def fixture(self, depth=4, capture_first=False):
        if capture_first:
            raise ValueError("this direct-capture fixture supports eager first only")
        base, rhs, _ = super().fixture(depth, False)
        wrapper = object.__new__(direct.DirectSegmentedTSA)
        wrapper._bind_captured_state(
            base, rhs.clone(), torch.zeros_like(rhs),
            torch.zeros_like(rhs) if depth > 1 else None,
            first_output=None, chunk=1, capture_first=False,
        )
        wrapper.step_graph = frozen.FakeStepReplay(wrapper)
        wrapper.first_graph = None
        return base, rhs, wrapper

    def test_cpu_constructor_and_unsupported_chunks_are_rejected(self):
        base, rhs, _ = self.fixture()
        with self.assertRaises(ValueError):
            direct.DirectSegmentedTSA(base, rhs, chunk=1, capture_first=False)
        for chunk in (0, 2, 4):
            with self.subTest(chunk=chunk):
                wrapper = object.__new__(direct.DirectSegmentedTSA)
                with self.assertRaises(ValueError):
                    wrapper._bind_captured_state(
                        base, rhs.clone(), torch.zeros_like(rhs), torch.zeros_like(rhs),
                        chunk=chunk, capture_first=False,
                    )
        wrapper = object.__new__(direct.DirectSegmentedTSA)
        with self.assertRaises(ValueError):
            wrapper._bind_captured_state(
                base, rhs.clone(), torch.zeros_like(rhs), torch.zeros_like(rhs),
                first_output=torch.zeros_like(rhs), chunk=1, capture_first=True,
            )
        with self.assertRaises(ValueError):
            direct.DirectSegmentedTSA._validate_base(base, 1, True)

    def test_all_requested_depths_and_first_modes_match_original_exactly(self):
        for depth in (1, 4, 8, 16):
            with self.subTest(depth=depth):
                base, rhs, wrapper = self.fixture(depth)
                self.assert_original(base, rhs, wrapper)
                self.assert_original(base, torch.zeros_like(rhs), wrapper)
                self.assertEqual(wrapper.step_graph.calls, 2 * (depth - 1))
                self.assertIsNone(wrapper.first_graph)

    def test_recipe_has_original_four_depth_minus_one_forward_and_inverse_ffts(self):
        for depth in (1, 4, 8, 16):
            with self.subTest(depth=depth):
                _, rhs, wrapper = self.fixture(depth)
                with patch.object(torch.fft, "fftn", wraps=torch.fft.fftn) as forward, patch.object(
                    torch.fft, "ifftn", wraps=torch.fft.ifftn
                ) as inverse:
                    wrapper(rhs)
                    self.assertEqual(forward.call_count, 4 * depth - 1)
                    self.assertEqual(inverse.call_count, 4 * depth - 1)

    def test_each_rhs_resets_state_and_returned_outputs_keep_owned_storage(self):
        base, rhs, wrapper = self.fixture(8)
        first = self.assert_original(base, rhs, wrapper)
        saved = first.clone()
        second = self.assert_original(base, rhs * complex(.37, -.42), wrapper)
        third = self.assert_original(base, rhs, wrapper)
        torch.testing.assert_close(first, saved, atol=0, rtol=0)
        torch.testing.assert_close(third, saved, atol=0, rtol=0)
        self.assertEqual(len({first.data_ptr(), second.data_ptr(), third.data_ptr()}), 3)
        for value in (first, second, third):
            for name in ("static_r", "static_z", "step_output"):
                self.assertNotEqual(value.data_ptr(), getattr(wrapper, name).data_ptr())
        self.assertEqual(wrapper.step_graph.calls, 3 * 7)
        self.assertIsNone(wrapper.first_graph)

    def test_native_complex128_uses_original_without_any_replay_or_depth_change(self):
        base, rhs, wrapper = self.fixture(16)
        native = rhs.to(torch.complex128) + (1e-10 + 2e-10j)
        torch.testing.assert_close(wrapper(native), base(native), atol=0, rtol=0)
        self.assertEqual(wrapper.step_graph.calls, 0)
        self.assertIsNone(wrapper.first_graph)
        self.assertEqual(base.depth, 16)

    def test_static_state_and_outputs_reject_view_replacement_but_copy_versions_advance(self):
        base, rhs, wrapper = self.fixture()
        versions = {name: getattr(wrapper, name)._version for name in ("static_r", "static_z", "step_output")}
        self.assert_original(base, rhs, wrapper)
        self.assert_original(base, rhs * .3, wrapper)
        for name, version in versions.items():
            self.assertGreater(getattr(wrapper, name)._version, version)
            with self.subTest(buffer=name):
                tensor = getattr(wrapper, name)
                self.reject_and_restore(base, rhs, wrapper, wrapper, name, tensor.transpose(-1, -2))


class RawCaptureLifecycleTests(unittest.TestCase):
    def harness(self, begin_error=None, stage_error=None, end_error=None):
        events, active = [], {"stream": False}
        stream, base, static_r, static_z, output = (object() for _ in range(5))
        test = self

        @contextmanager
        def stream_context(selected):
            test.assertIs(selected, stream)
            test.assertFalse(active["stream"])
            active["stream"] = True
            events.append("stream_enter")
            try:
                yield
            finally:
                events.append("stream_exit")
                active["stream"] = False

        class FakeRawGraph:
            def capture_begin(self, *args, **kwargs):
                events.append("begin")
                test.assertTrue(active["stream"])
                test.assertFalse(torch.is_grad_enabled())
                test.assertEqual(args, ())
                test.assertEqual(kwargs, {"capture_error_mode": "global"})
                if begin_error is not None:
                    raise begin_error

            def capture_end(self, *args, **kwargs):
                events.append("end")
                test.assertTrue(active["stream"])
                test.assertFalse(torch.is_grad_enabled())
                test.assertEqual(args, ())
                test.assertEqual(kwargs, {})
                if end_error is not None:
                    raise end_error

        def stage(selected_base, selected_r, selected_z):
            events.append("stage")
            test.assertTrue(active["stream"])
            test.assertFalse(torch.is_grad_enabled())
            test.assertIs(selected_base, base)
            test.assertIs(selected_r, static_r)
            test.assertIs(selected_z, static_z)
            if stage_error is not None:
                raise stage_error
            return output

        @contextmanager
        def mocked():
            with ExitStack() as stack:
                stack.enter_context(patch.object(torch.cuda, "stream", side_effect=stream_context))
                stage_mock = stack.enter_context(patch.object(direct.SegmentedTSA, "_subsequent_stage", side_effect=stage))
                forbidden = [
                    stack.enter_context(patch(target, side_effect=AssertionError("forbidden capture pre-step: " + target)))
                    for target in ("torch.cuda.graph", "gc.collect", "torch.cuda.empty_cache", "torch.cuda.synchronize")
                ]
                try:
                    yield FakeRawGraph(), base, static_r, static_z, stream, output, events, stage_mock
                finally:
                    for operation in forbidden:
                        operation.assert_not_called()
        return mocked()

    def test_success_uses_one_explicit_stream_no_grad_and_raw_begin_end_without_pool(self):
        previous_grad_state = torch.is_grad_enabled()
        with self.harness() as (graph, base, residual, correction, stream, output, events, stage):
            result = direct._capture_subsequent_stage(graph, base, residual, correction, stream)
            self.assertIs(result, output)
            self.assertEqual(events, ["stream_enter", "begin", "stage", "end", "stream_exit"])
            stage.assert_called_once()
        self.assertEqual(torch.is_grad_enabled(), previous_grad_state)

    def test_failed_begin_does_not_execute_stage_or_attempt_capture_end(self):
        error = RuntimeError("begin sentinel")
        with self.harness(begin_error=error) as (graph, base, residual, correction, stream, _, events, stage):
            with self.assertRaises(RuntimeError) as caught:
                direct._capture_subsequent_stage(graph, base, residual, correction, stream)
            self.assertIs(caught.exception, error)
            self.assertEqual(events, ["stream_enter", "begin", "stream_exit"])
            stage.assert_not_called()

    def test_body_exception_and_interrupt_end_capture_inside_stream_without_fallback(self):
        for error in (RuntimeError("stage sentinel"), KeyboardInterrupt("stage interrupt")):
            with self.subTest(error=type(error).__name__):
                with self.harness(stage_error=error) as (graph, base, residual, correction, stream, _, events, stage):
                    with self.assertRaises(type(error)) as caught:
                        direct._capture_subsequent_stage(graph, base, residual, correction, stream)
                    self.assertIs(caught.exception, error)
                    self.assertEqual(events, ["stream_enter", "begin", "stage", "end", "stream_exit"])
                    stage.assert_called_once()

    def test_capture_end_failure_is_propagated_after_successful_stage(self):
        error = RuntimeError("end sentinel")
        with self.harness(end_error=error) as (graph, base, residual, correction, stream, _, events, stage):
            with self.assertRaises(RuntimeError) as caught:
                direct._capture_subsequent_stage(graph, base, residual, correction, stream)
            self.assertIs(caught.exception, error)
            self.assertEqual(events, ["stream_enter", "begin", "stage", "end", "stream_exit"])
            stage.assert_called_once()

    def test_stage_error_remains_primary_when_capture_end_also_fails(self):
        initial, ending = RuntimeError("primary stage error"), RuntimeError("secondary end error")
        with self.harness(stage_error=initial, end_error=ending) as (graph, base, residual, correction, stream, _, events, stage):
            with self.assertRaises(RuntimeError) as caught:
                direct._capture_subsequent_stage(graph, base, residual, correction, stream)
            self.assertIs(caught.exception, initial)
            self.assertIs(caught.exception.__cause__, ending)
            self.assertEqual(events, ["stream_enter", "begin", "stage", "end", "stream_exit"])
            stage.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
