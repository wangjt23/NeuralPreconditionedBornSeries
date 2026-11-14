import unittest

import torch

from rte.tail_response_network import TailResponseNetwork


class TailResponseNetworkTests(unittest.TestCase):
    def test_reference_initialization_tiny_scaling_and_gradient(self):
        torch.set_num_threads(1)
        torch.manual_seed(96)
        net = TailResponseNetwork(width=8, blocks=1, modes=2)
        t, s = torch.ones(2, 1, 16, 16)*10, torch.ones(2, 1, 16, 16)*9.99
        r, ref = [torch.randn(2, 12, 16, 16, dtype=torch.complex64) for _ in range(2)]
        torch.testing.assert_close(net(t, s, r, ref), ref)
        torch.nn.init.normal_(net.output.weight, std=.01)
        z = net(t, s, r, ref)
        torch.testing.assert_close(net(t, s, r*1e-9, ref*1e-9)/1e-9, z, atol=2e-6, rtol=2e-6)
        self.assertEqual(float(net(t, s, r*0, ref*0).abs().sum()), 0.)
        z.abs().square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))


if __name__ == '__main__':
    unittest.main()
