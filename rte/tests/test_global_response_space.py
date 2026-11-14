import unittest

import torch

from rte.audit_global_response_space import VerifiedResponseSpace


class GlobalResponseSpaceTests(unittest.TestCase):
    def test_exact_on_verified_space_and_residual_identity(self):
        torch.manual_seed(3)
        matrix = torch.randn(12, 12, dtype=torch.complex128)+4*torch.eye(12)
        action = lambda x: (matrix @ x.flatten(1).T).T.reshape_as(x)
        base = lambda x: x/4
        columns = torch.randn(12, 3, dtype=torch.complex128)
        model = VerifiedResponseSpace(action, base, columns, (3, 2, 2))
        q = model.q.T.reshape(3, 3, 2, 2)
        torch.testing.assert_close(action(model(q)), q, atol=1e-12, rtol=1e-12)
        r = torch.randn(2, 3, 2, 2, dtype=torch.complex128)
        projected = (r.flatten(1).T-model.q @ (model.q.mH @ r.flatten(1).T)).T.reshape_as(r)
        torch.testing.assert_close(r-action(model(r)), projected-action(base(projected)), atol=1e-12, rtol=1e-12)

    def test_rank_deficiency(self):
        z = torch.arange(1., 13.).to(torch.complex128)[:, None]
        model = VerifiedResponseSpace(lambda x: 2*x, lambda x: x/3,
            torch.cat((z, 2*z), dim=1), (3, 2, 2))
        self.assertEqual(model.rank, 1)
        self.assertLess(model.verification_error, 1e-12)


if __name__ == '__main__':
    unittest.main()
