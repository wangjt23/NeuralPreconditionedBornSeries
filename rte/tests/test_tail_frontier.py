import torch

from rte.augmented_fgmres import augmented_fgmres
from rte.probe_tail_frontier import frontier_fgmres


def test_no_proposal_matches_default_and_rejected_zero_is_safe():
    torch.manual_seed(1919)
    matrix = torch.randn(20, 20, dtype=torch.complex128)/5+2*torch.eye(20)
    rhs = torch.randn(1, 1, 4, 5, dtype=torch.complex128)
    action = lambda x: (matrix@x.flatten()).reshape_as(x)
    expected, a = augmented_fgmres(action, rhs, lambda x:x, tolerance=1e-9)
    for inject in (None, lambda x,r:torch.zeros_like(r)):
        result, b = frontier_fgmres(action, rhs, lambda x:x, inject=inject, tolerance=1e-9)
        torch.testing.assert_close(result, expected)
        assert a['iterations'] == b['iterations']
        assert not b['failed']


def test_first_classical_application_after_injection_uses_saved_residual():
    torch.manual_seed(1919)
    matrix = torch.randn(24, 24, dtype=torch.complex128)/6+2*torch.eye(24)
    rhs = torch.randn(1, 1, 4, 6, dtype=torch.complex128)
    action = lambda x: (matrix@x.flatten()).reshape_as(x)
    calls, captured = [], []
    def precondition(r):
        calls.append(r.clone())
        return r
    def inject(x, r):
        captured.append(r.clone()/r.norm())
        return torch.linalg.solve(matrix, r.flatten()).reshape_as(r)+.1*r
    _, status = frontier_fgmres(action, rhs, precondition, inject=inject, tolerance=1e-8)
    assert status['injection_accepted'] == 1
    assert len(calls) > 4
    torch.testing.assert_close(calls[4], captured[0])
    assert not status['failed']
