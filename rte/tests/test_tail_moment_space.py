import unittest

import torch

from rte.tail_moment_space import moment_components, combine_moment_space


class TailMomentTests(unittest.TestCase):
    def test_partition_and_projected_optimum(self):
        torch.set_num_threads(1)
        torch.manual_seed(72)
        z = torch.randn(2, 12, 3, 3, dtype=torch.complex128, requires_grad=True)
        parts = moment_components(z)
        torch.testing.assert_close(parts.sum(1), z)
        target = torch.randn_like(z)
        q = torch.randn(2, 3, 12, 3, 3, dtype=torch.complex128)
        for b in range(2):
            real = torch.cat((q[b].flatten(1).real, q[b].flatten(1).imag), 1).T
            orthogonal = torch.linalg.qr(real).Q.T
            q[b] = torch.complex(orthogonal[:, :108], orthogonal[:, 108:]).reshape(3, 12, 3, 3)
        residual = target-(q*(q.conj()*target[:, None]).real.flatten(2).sum(2)[:, :, None, None, None]).sum(1)
        correction, ratio = combine_moment_space(lambda value:value, z, residual, q)
        for b in range(2):
            columns = torch.cat((q[b], parts[b])).flatten(1).T
            real = torch.cat((columns.real, columns.imag))
            r = torch.cat((residual[b].flatten().real, residual[b].flatten().imag))
            coef = torch.linalg.lstsq(real, r).solution
            expected = (r-real@coef).square().sum()/r.square().sum()
            torch.testing.assert_close(ratio[b], expected)
        ratio.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_exact_teacher_and_zero_proposal(self):
        z = torch.randn(2, 12, 4, 4, dtype=torch.complex128)
        correction, ratio = combine_moment_space(lambda x:3*x, z, 3*z, None)
        torch.testing.assert_close(correction, z, atol=1e-10, rtol=1e-10)
        self.assertLess(float(ratio.max()), 1e-20)
        correction, ratio = combine_moment_space(lambda x:x, z*0, z, None)
        self.assertEqual(float(correction.abs().sum()), 0.)
        torch.testing.assert_close(ratio, torch.ones_like(ratio))


if __name__ == '__main__':
    unittest.main()
