"""Small CPU-only protocol checks. No model, solver, or dataset training."""

import copy
import math
import unittest

import torch

import train_tail_multiresolution as study


def paired_banks(count=3, contexts=True):
    result = []
    original_ids = torch.tensor([2, 0]) if contexts else torch.arange(count)
    for n in (64, 128):
        sigma_t = torch.arange(1, count + 1, dtype=torch.float32)[:, None, None, None].expand(count, 1, n, n).clone()
        sigma_s = sigma_t * .75
        data = dict(sigma_t=sigma_t, sigma_s=sigma_s)
        for key in study.SOURCE_PARAMETERS:
            data[key] = torch.arange(count)[:, None]
        t = sigma_t[original_ids] / .1
        s = t - .1 * (sigma_t[original_ids] - sigma_s[original_ids])
        bank = dict(q=torch.zeros(len(original_ids), 12, n, n), t=t, s=s)
        if contexts:
            bank["indices"] = original_ids.clone()
        result.append(dict(bank=bank, data=data))
    return result


class ProtocolTests(unittest.TestCase):
    def test_original_eligible_ids_and_nested_materials(self):
        coarse, fine = paired_banks()
        receipt = study.check_paired_identity(coarse, fine, contexts=True)
        self.assertEqual(receipt["original_ids"], [2, 0])
        self.assertEqual(receipt["original_parent_count"], 3)
        fine["bank"]["indices"] = torch.tensor([0, 2])
        with self.assertRaisesRegex(ValueError, "eligible original"):
            study.check_paired_identity(coarse, fine, contexts=True)

    def test_same_mislabelling_cannot_pass(self):
        coarse, fine = paired_banks()
        for saved in (coarse, fine):
            saved["bank"]["indices"] = torch.tensor([0, 2])
        with self.assertRaisesRegex(ValueError, "saved original IDs"):
            study.check_paired_identity(coarse, fine, contexts=True)

    def test_source_parameters_and_full_validation_are_guarded(self):
        coarse, fine = paired_banks(16, contexts=False)
        self.assertEqual(study.check_paired_identity(coarse, fine, contexts=False, expected_count=16)["count"], 16)
        changed = copy.deepcopy(fine)
        changed["data"]["geometry_base64"][0] += 1
        with self.assertRaisesRegex(ValueError, "source/geometry"):
            study.check_paired_identity(coarse, changed, contexts=False, expected_count=16)
        with self.assertRaisesRegex(ValueError, "ALL 17"):
            study.check_paired_identity(coarse, fine, contexts=False, expected_count=17)

    def test_sampling_preserves_reference_draw_order(self):
        original = torch.tensor([2, 8, 13, 21])
        plan = study.build_sampling_plan(original, 7, 4, 27002)
        generator = torch.Generator().manual_seed(27002 + 72)
        for update in range(7):
            ids = torch.randint(4, (4,), generator=generator)
            angle = torch.rand(4, generator=generator, dtype=torch.float64) * (2 * math.pi)
            self.assertTrue(torch.equal(plan["ids"][update], ids))
            self.assertTrue(torch.equal(plan["phase_angles"][update], angle))
            self.assertTrue(torch.equal(plan["generator_states"][update + 1], generator.get_state()))
        repeat = study.build_sampling_plan(original, 7, 4, 27002)
        self.assertEqual(plan["identity_phase_sha256"], repeat["identity_phase_sha256"])

    def test_equal_presentations_only_change_alternate_grid(self):
        plan = study.build_sampling_plan(torch.arange(8), 1000, 4, 27002)
        a, b = (study.sampling_rows(plan, arm) for arm in study.ARMS)
        self.assertEqual(len(a), 4000)
        self.assertEqual(len(b), 4000)
        self.assertEqual(sum(row["grid"] == 128 for row in b), 2000)
        self.assertTrue(all(row["grid"] == 64 for row in a))
        for row_a, row_b in zip(a, b):
            self.assertEqual({k: v for k, v in row_a.items() if k != "grid"},
                             {k: v for k, v in row_b.items() if k != "grid"})

    def test_selection_uses_all_cases_at_both_grids(self):
        rows = [dict(grid=grid, index=index, failed=int(grid == 128 and index == 3),
                     classical_calls=grid / 64, injection_calls=2, true_residual=1e-6)
                for grid in (64, 128) for index in range(16)]
        failures, calls, residual = study.selection_score(rows)
        self.assertEqual(failures, 1)
        self.assertEqual(calls, 2.5)
        self.assertAlmostEqual(residual, 1e-6)
        with self.assertRaisesRegex(ValueError, "every validation ID"):
            study.selection_score(rows[:-1])
        rows[0]["true_residual"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            study.selection_score(rows)


if __name__ == "__main__":
    unittest.main()
