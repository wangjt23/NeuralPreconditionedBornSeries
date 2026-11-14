"""Inherited CPU guards plus mocked raw-capture lifecycle for the full action.

No CUDA constructor or GPU work is performed. Real capture correctness remains
subject to the separate prescribed operator and complete-solver parity gates.
"""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import torch

# Local workspace uses the sibling frozen dependency folder. A staged flat
# directory can supply these same modules alongside this file without changes.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "headroom"))
sys.path.insert(0, str(HERE))

import test_graph_classical_v2 as contracts
import direct_full_graph as direct
from direct_full_graph import DirectFullGraph

contracts.GraphClassicalV2=DirectFullGraph
contracts.GraphClassical=DirectFullGraph


class RawFullCaptureLifecycleTests(unittest.TestCase):
    def harness(self, begin_error=None, body_error=None, end_error=None):
        events, active = [], {"stream": False}
        stream, static_input, output = object(), object(), object()
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

        def apply(selected_input):
            events.append("body")
            test.assertIs(selected_input, static_input)
            test.assertTrue(active["stream"])
            test.assertFalse(torch.is_grad_enabled())
            if body_error is not None:
                raise body_error
            return output

        @contextmanager
        def mocked():
            with ExitStack() as stack:
                stack.enter_context(patch.object(torch.cuda, "stream", side_effect=stream_context))
                forbidden = [
                    stack.enter_context(patch(target, side_effect=AssertionError("forbidden raw capture pre-step: " + target)))
                    for target in ("torch.cuda.graph", "gc.collect", "torch.cuda.empty_cache", "torch.cuda.synchronize")
                ]
                base = Mock(side_effect=apply)
                try:
                    yield FakeRawGraph(), base, static_input, stream, output, events
                finally:
                    for operation in forbidden:
                        operation.assert_not_called()
        return mocked()

    def test_success_orders_begin_body_end_in_same_stream_no_grad_global_without_pool(self):
        prior_grad_state = torch.is_grad_enabled()
        with self.harness() as (graph, base, static_input, stream, output, events):
            result = direct._capture_full_action(graph, base, static_input, stream)
            self.assertIs(result, output)
            self.assertEqual(events, ["stream_enter", "begin", "body", "end", "stream_exit"])
            base.assert_called_once_with(static_input)
        self.assertEqual(torch.is_grad_enabled(), prior_grad_state)

    def test_begin_failure_skips_body_and_capture_end(self):
        initial = RuntimeError("full capture begin sentinel")
        with self.harness(begin_error=initial) as (graph, base, static_input, stream, _, events):
            with self.assertRaises(RuntimeError) as caught:
                direct._capture_full_action(graph, base, static_input, stream)
            self.assertIs(caught.exception, initial)
            self.assertEqual(events, ["stream_enter", "begin", "stream_exit"])
            base.assert_not_called()

    def test_body_and_end_failure_preserves_primary_exception_with_end_chained(self):
        for initial in (RuntimeError("full action sentinel"), KeyboardInterrupt("full action interrupt")):
            with self.subTest(primary_type=type(initial).__name__):
                ending = RuntimeError("full capture end sentinel")
                with self.harness(body_error=initial, end_error=ending) as (graph, base, static_input, stream, _, events):
                    with self.assertRaises(type(initial)) as caught:
                        direct._capture_full_action(graph, base, static_input, stream)
                    self.assertIs(caught.exception, initial)
                    self.assertIs(caught.exception.__cause__, ending)
                    self.assertEqual(events, ["stream_enter", "begin", "body", "end", "stream_exit"])
                    base.assert_called_once_with(static_input)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(contracts.GraphClassicalV2Tests)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RawFullCaptureLifecycleTests))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
