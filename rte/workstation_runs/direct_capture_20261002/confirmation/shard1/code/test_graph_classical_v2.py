"""CPU-only ownership/guard tests; these do not test CUDA graph performance.

The production CUDA constructor is bypassed explicitly. A fake replay applies
the actual classical RTE operator into static output storage, exercising the
same captured-state guards and output-ownership path without GPU work.
"""

import copy
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
from graph_classical_v2 import GraphClassical, GraphClassicalV2


class FakeReplay:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.calls = 0

    def replay(self):
        self.calls += 1
        self.wrapper.static_output.copy_(self.wrapper.base(self.wrapper.static_input))


class GraphClassicalV2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        n = 8
        physical = PeriodicRTE2D(
            shape=(n, n), dx=1. / n, n_angles=12, anisotropy_g=.65,
            eta_scale=0., eta_floor=0., fdtype=torch.float32, device="cpu",
        )
        axis = torch.arange(n, dtype=torch.float32) * (2 * torch.pi / n)
        t = (3. + .21 * axis.cos()[:, None] + .13 * axis.sin()[None, :])[None, None]
        s = (1. + .09 * axis.sin()[:, None] + .07 * axis.cos()[None, :])[None, None]
        generator = torch.Generator(device="cpu").manual_seed(531)
        self.rhs = torch.randn(1, 12, n, n, dtype=torch.complex64, generator=generator)
        physical.setup(t, s, self.rhs)
        self.base = ClassicalTransportStack(physical, "tsa", depth=2, eta_scale=0.)
        self.wrapper = object.__new__(GraphClassicalV2)
        static_input = self.rhs.clone()
        static_output = self.base(static_input)
        self.wrapper._bind_captured_state(self.base, static_input, static_output)
        self.wrapper.graph = FakeReplay(self.wrapper)

    def assert_unchanged_apply(self):
        actual = self.wrapper(self.rhs)
        expected = self.base(self.rhs).to(torch.complex128)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def assert_replacement_rejected_then_restored(self, owner, name, replacement):
        original = getattr(owner, name)
        before = self.wrapper.graph.calls
        setattr(owner, name, replacement)
        try:
            with self.assertRaises(RuntimeError):
                self.wrapper(self.rhs)
            self.assertEqual(self.wrapper.graph.calls, before)
        finally:
            setattr(owner, name, original)
        self.assert_unchanged_apply()

    def test_alias_and_production_constructor_do_not_enable_cpu_graph_capture(self):
        self.assertIs(GraphClassical, GraphClassicalV2)
        with self.assertRaises(ValueError):
            GraphClassicalV2(self.base, self.rhs)
        self.assert_unchanged_apply()

    def test_material_and_inverse_transpose_views_reject_before_replay_then_restore(self):
        for owner, name in (
            (self.base.rte, "sigma_t"), (self.base.rte, "sigma_s"),
            (self.base.reference, "sigma_t"), (self.base.reference, "sigma_s"),
            (self.base.reference, "_reference_transport_inverse"),
        ):
            with self.subTest(owner="rte" if owner is self.base.rte else "reference", tensor=name):
                original = getattr(owner, name)
                replacement = original.transpose(-1, -2)
                self.assertEqual(original.data_ptr(), replacement.data_ptr())
                self.assertEqual(original.shape, replacement.shape)
                self.assertNotEqual(original.stride(), replacement.stride())
                self.assert_replacement_rejected_then_restored(owner, name, replacement)

    def test_equal_storage_tensor_identity_and_equal_value_clone_replacements_reject(self):
        for owner, names in (
            (self.base.rte, ("sigma_t", "sigma_s", "directions", "kx", "ky", "phase_matrix", "weights")),
            (self.base.reference, ("sigma_t", "sigma_s", "directions", "kx", "ky", "phase_matrix", "weights", "sigma0", "_reference_transport_inverse")),
        ):
            for name in names:
                original = getattr(owner, name)
                for replacement in (original.view_as(original), original.clone()):
                    with self.subTest(tensor=name, same_storage=replacement.data_ptr() == original.data_ptr()):
                        self.assertIsNot(original, replacement)
                        self.assertTrue(torch.equal(original, replacement))
                        self.assert_replacement_rejected_then_restored(owner, name, replacement)

    def test_base_and_operator_identity_changes_reject_even_when_tensors_are_identical(self):
        # Shallow copies retain all tensor identities/settings. A pointer-only
        # tensor guard would miss these changes in captured Python operators.
        self.assert_replacement_rejected_then_restored(self.wrapper, "base", copy.copy(self.base))
        self.assert_replacement_rejected_then_restored(self.base, "rte", copy.copy(self.base.rte))
        self.assert_replacement_rejected_then_restored(self.base, "reference", copy.copy(self.base.reference))

    def test_settings_and_generation_changes_reject_and_depth_restore_resumes(self):
        changes = [(self.base, name, replacement) for name, replacement in (
            ("kind", "reference"), ("depth", 3), ("eta_scale", .1), ("relaxation", .9),
        )]
        for operator in (self.base.rte, self.base.reference):
            changes.extend((operator, name, replacement) for name, replacement in (
                ("shape", (16, 8)), ("dx", operator.dx * 1.1),
                ("n_angles", 10), ("anisotropy_g", .5),
                ("float_dtype", torch.float64), ("complex_dtype", torch.complex128),
                ("eta_scale", .1), ("eta_floor", .002), ("diffusion_floor", .003),
                ("setup_generation", operator.setup_generation + 1),
            ))
        for owner, name, replacement in changes:
            with self.subTest(setting=name):
                self.assert_replacement_rejected_then_restored(owner, name, replacement)

    def test_in_place_tensor_version_changes_are_rejected_before_replay(self):
        tensor = self.base.rte.phase_matrix
        previous_version = tensor._version
        tensor.add_(.001)
        self.assertGreater(tensor._version, previous_version)
        for rhs in (self.rhs, self.rhs.to(torch.complex128)):
            with self.subTest(dtype=rhs.dtype):
                with self.assertRaises(RuntimeError):
                    self.wrapper(rhs)
        self.assertEqual(self.wrapper.graph.calls, 0)

    def test_nonzero_storage_offset_views_with_matching_values_are_rejected(self):
        for owner, name in (
            (self.base.rte, "sigma_t"),
            (self.base.reference, "_reference_transport_inverse"),
            (self.wrapper, "static_input"), (self.wrapper, "static_output"),
        ):
            with self.subTest(tensor=name):
                original = getattr(owner, name)
                backing = torch.empty(original.numel() + 1, dtype=original.dtype)
                replacement = backing.as_strided(original.shape, original.stride(), storage_offset=1)
                replacement.copy_(original)
                self.assertEqual(replacement.shape, original.shape)
                self.assertEqual(replacement.stride(), original.stride())
                self.assertTrue(torch.equal(replacement, original))
                self.assertEqual(replacement.storage_offset(), 1)
                self.assertNotEqual(original.storage_offset(), replacement.storage_offset())
                self.assertEqual(self.wrapper._tensor_signature(replacement)[-1], 1)
                self.assert_replacement_rejected_then_restored(owner, name, replacement)

    def test_missing_inverse_is_rejected_without_property_access_or_cache_rebuild(self):
        reference = self.base.reference
        original = reference._reference_transport_inverse
        reference._reference_transport_inverse = None
        try:
            with patch.object(reference, "_build_reference_transport_inverse", side_effect=AssertionError("guard attempted cache rebuild")) as rebuild:
                with self.assertRaises(RuntimeError):
                    self.wrapper(self.rhs)
                rebuild.assert_not_called()
                self.assertIsNone(reference._reference_transport_inverse)
                self.assertEqual(self.wrapper.graph.calls, 0)
        finally:
            reference._reference_transport_inverse = original
        self.assert_unchanged_apply()

    def test_captured_cache_has_a_strong_reference_after_owner_replacement(self):
        reference = self.base.reference
        original = reference._reference_transport_inverse
        retained = weakref.ref(original)
        reference._reference_transport_inverse = original.clone()
        del original
        gc.collect()
        self.assertIsNotNone(retained())
        with self.assertRaises(RuntimeError):
            self.wrapper(self.rhs)
        reference._reference_transport_inverse = retained()
        self.assert_unchanged_apply()

    def test_static_buffer_layout_changes_reject_but_copy_versions_can_advance(self):
        initial_versions = (self.wrapper.static_input._version, self.wrapper.static_output._version)
        self.assert_unchanged_apply()
        self.assert_unchanged_apply()
        self.assertGreater(self.wrapper.static_input._version, initial_versions[0])
        self.assertGreater(self.wrapper.static_output._version, initial_versions[1])
        for name in ("static_input", "static_output"):
            with self.subTest(buffer=name):
                original = getattr(self.wrapper, name)
                self.assert_replacement_rejected_then_restored(self.wrapper, name, original.transpose(-1, -2))

    def test_returned_outputs_have_owned_storage_and_survive_subsequent_replay(self):
        first = self.wrapper(self.rhs)
        snapshot = first.clone()
        self.assertEqual(first.dtype, torch.complex128)
        self.assertNotEqual(first.data_ptr(), self.wrapper.static_output.data_ptr())
        second_rhs = self.rhs * complex(.37, -.42)
        second = self.wrapper(second_rhs)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(first, snapshot, atol=0, rtol=0)
        torch.testing.assert_close(second, self.base(second_rhs).to(torch.complex128), atol=0, rtol=0)
        self.assertEqual(self.wrapper.graph.calls, 2)

    def test_native_complex128_fallback_preserves_original_precision_and_skips_replay(self):
        rhs = self.rhs.to(torch.complex128) + (1e-10 + 2e-10j)
        expected = self.base(rhs)
        actual = self.wrapper(rhs)
        self.assertEqual(actual.dtype, torch.complex128)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(self.wrapper.graph.calls, 0)

    def test_wrong_shape_and_device_reject_before_replay(self):
        for value in (self.rhs[..., :-1], torch.empty_like(self.rhs, device="meta")):
            with self.subTest(shape=tuple(value.shape), device=str(value.device)):
                with self.assertRaises(ValueError):
                    self.wrapper(value)
                self.assertEqual(self.wrapper.graph.calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
