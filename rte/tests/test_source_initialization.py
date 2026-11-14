import unittest

import torch

from rte.source_initialization import characteristic_project, constant_initial, solve_from_initial
from rte.run_hard_case_experiments import make_case


class SourceInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_original_source_tolerance_and_zero(self):
        rhs = torch.ones(1, 2, 2, 2, dtype=torch.complex128)
        action = lambda x: 2*x
        initial = rhs*(.5-1e-7)
        result, status = solve_from_initial(action, rhs, lambda x: x/2, initial, tolerance=1e-5)
        self.assertEqual(status['iterations'], 0)
        self.assertEqual(status['failed'], 0)
        result, status = solve_from_initial(action, rhs, lambda x: x/2, rhs*20, tolerance=1e-9)
        torch.testing.assert_close(action(result), rhs)
        self.assertEqual(status['failed'], 0)
        result, status = solve_from_initial(action, torch.zeros_like(rhs), lambda x: x/2, initial)
        torch.testing.assert_close(result, torch.zeros_like(rhs))

    def test_mass_and_characteristic_projection(self):
        op = make_case('mixed', 8, 12, torch.float64, 0.)[0]
        rhs = torch.randn_like(op.source)
        value = constant_initial(op, rhs)
        torch.testing.assert_close(op.A_op(value).sum(), rhs.sum(), atol=1e-10, rtol=1e-10)
        projected = characteristic_project(op, rhs, modes=16)
        torch.testing.assert_close(characteristic_project(op, projected, modes=16), projected)
        torch.testing.assert_close(characteristic_project(op, rhs, modes=64), rhs)


if __name__ == '__main__':
    unittest.main()
