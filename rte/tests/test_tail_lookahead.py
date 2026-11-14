import torch

from rte.augmented_fgmres import augmented_fgmres, projected_direction_loss
from rte.tail_lookahead import continuation_loss


def problem():
    torch.manual_seed(1920)
    matrix = torch.randn(24, 24, dtype=torch.complex128)/6+2*torch.eye(24)
    rhs = torch.randn(1, 1, 4, 6, dtype=torch.complex128)
    action = lambda x: (matrix@x.flatten()).reshape_as(x)
    return action, rhs


def test_continuation_matches_deployed_fgmres_and_single_projected_loss():
    action, rhs = problem()
    contexts = []
    _, prefix = augmented_fgmres(action, rhs, lambda x:x, maxiter=4,
        tolerance=1e-14, capture_prefix=contexts.append)
    context = contexts[0]
    residual = context['residual']
    basis = context['image_basis'].transpose(0, 1)
    proposal = .8*residual+.2*torch.randn_like(residual)
    ratio = continuation_loss(action, lambda x:x, proposal, residual, basis, 2)
    expected = projected_direction_loss(action, proposal, residual, basis)
    torch.testing.assert_close(ratio[:, 0], expected, rtol=1e-9, atol=1e-12)
    _, status = augmented_fgmres(action, rhs, lambda x:x, inject=lambda x,r:proposal,
        maxiter=7, tolerance=1e-14)
    for j in range(3):
        actual = (status['trace'][5+j]/prefix['true_residual'])**2
        torch.testing.assert_close(ratio[0,j], torch.tensor(actual, dtype=ratio.dtype), rtol=1e-8, atol=1e-12)


def test_continuation_gradient_and_zero_proposal_are_finite():
    action, rhs = problem()
    contexts = []
    augmented_fgmres(action, rhs, lambda x:x, maxiter=4, tolerance=1e-14, capture_prefix=contexts.append)
    c = contexts[0]
    basis = c['image_basis'].transpose(0, 1)
    z = torch.randn_like(rhs, requires_grad=True)
    function = lambda value: continuation_loss(action, lambda x:x, value, c['residual'], basis, 2)
    assert torch.autograd.gradcheck(function, (z,), fast_mode=True)
    value = function(torch.zeros_like(z))
    assert torch.isfinite(value).all()
    torch.testing.assert_close(value, torch.ones_like(value))


def test_transport_mixed_precision_continuation_matches_physical_trace():
    from rte.mixed_precision_transport import high_precision_copy
    from rte.run_hard_case_experiments import make_case
    from rte.transport_baselines import ClassicalTransportStack
    low = make_case('mixed', 8, 12, torch.float32, 0.)[0]
    high = high_precision_copy(low)
    base = ClassicalTransportStack(low, 'tsa', 2, 0.)
    precondition = lambda x:base(x.to(torch.complex64)).to(torch.complex128)
    rhs = torch.randn_like(high.source)
    contexts = []
    _, prefix = augmented_fgmres(high.A_op, rhs, precondition, maxiter=4,
        tolerance=1e-14, capture_prefix=contexts.append)
    c = contexts[0]
    proposal = base.reference.reference_transport_solve(c['residual'].to(torch.complex64)).to(torch.complex128)
    ratio = continuation_loss(high.A_op, precondition, proposal, c['residual'],
        c['image_basis'].transpose(0, 1), 2)
    _, status = augmented_fgmres(high.A_op, rhs, precondition,
        inject=lambda x,r:proposal, maxiter=7, tolerance=1e-14)
    expected = torch.tensor([(r/prefix['true_residual'])**2 for r in status['trace'][5:]], dtype=ratio.dtype)
    torch.testing.assert_close(ratio[0], expected, rtol=2e-4, atol=1e-10)


def test_batched_continuation_is_independent_per_medium():
    action, rhs = problem()
    contexts = []
    augmented_fgmres(action, rhs, lambda x:x, maxiter=4, tolerance=1e-14, capture_prefix=contexts.append)
    c = contexts[0]
    basis = c['image_basis'].transpose(0, 1)
    proposal = torch.randn_like(rhs)
    r = torch.cat((c['residual'], 2*c['residual']))
    z = torch.cat((proposal, 2*proposal))
    qb = torch.cat((basis, basis))
    batched_action = lambda x:torch.cat([action(row[None]) for row in x])
    result = continuation_loss(batched_action, lambda x:x, z, r, qb, 2)
    torch.testing.assert_close(result[0], result[1])
