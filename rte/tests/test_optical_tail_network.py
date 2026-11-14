import unittest

import torch

from rte.optical_tail_network import OpticalAxisAttention, OpticalTailNetwork
from rte.tail_response_network import TailResponseNetwork


class OpticalTailTests(unittest.TestCase):
    def test_zero_upgrade_and_scale_equivariance(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        base = TailResponseNetwork(width=8, blocks=1, modes=2)
        torch.nn.init.normal_(base.output.weight, std=.02)
        net = OpticalTailNetwork(width=8, blocks=1, modes=2)
        net.load_state_dict(base.state_dict())
        for n in (16, 32):
            t, s = torch.ones(1, 1, n, n)*10, torch.ones(1, 1, n, n)*9.99
            q, ref = [torch.randn(1, 12, n, n, dtype=torch.complex64) for _ in range(2)]
            torch.testing.assert_close(net(t, s, q, ref), base(t, s, q, ref), atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(net(t, s, q*1e-9, ref*1e-9)/1e-9, net(t, s, q, ref), atol=2e-6, rtol=2e-6)
            net(t, s, q, ref).abs().square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_periodic_translation_of_optical_attention(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        for axis in (0, 1):
            net = OpticalAxisAttention(8, axis)
            torch.nn.init.normal_(net.output.weight, std=.02)
            hidden, geometry = torch.randn(2, 8, 12, 16), torch.randn(2, 4, 12, 16)
            t = torch.rand(2, 1, 12, 16)*10+1
            shift = lambda x:torch.roll(x, (3, 5), (-2, -1))
            torch.testing.assert_close(net(shift(hidden), shift(geometry), shift(t)),
                shift(net(hidden, geometry, t)), atol=2e-6, rtol=2e-6)


if __name__ == '__main__':
    unittest.main()
