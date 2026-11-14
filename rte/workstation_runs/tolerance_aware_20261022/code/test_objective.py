"""Algebraic guards for the causal objective change; no physical solves."""
import math,unittest
import torch
from objective import components,original_prefix_factor,gradient_comparison,BETA,TAU

class ObjectiveTests(unittest.TestCase):
    def test_original_rhs_threshold_and_normalization_invariance(self):
        r=torch.tensor([[3.+0j,4.+0j],[0.+0j,2.+0j]],dtype=torch.complex128)
        qnorm=torch.tensor([10.,4.],dtype=torch.float64)
        eta=original_prefix_factor(r,qnorm)
        self.assertTrue(torch.equal(eta,torch.tensor([.5,.5],dtype=torch.float64)))
        rho=torch.tensor([[4e-10],[4e-10]],dtype=torch.float64)
        _,excess=components(rho,eta,'tolerance')
        self.assertLess(float(excess.abs().max()),1e-12)
        scaled=original_prefix_factor(r*7.,qnorm*7.)
        self.assertTrue(torch.allclose(eta,scaled,atol=1e-15,rtol=0))

    def test_merely_changing_log_units_is_a_gradient_null(self):
        rho=torch.tensor([[.1,.01,.001],[.2,.02,.002]],dtype=torch.float64,requires_grad=True)
        eta=torch.tensor([.02,.00002],dtype=torch.float64)
        old,d=components(rho,eta,'two_step')
        a=torch.autograd.grad(old,rho,retain_graph=True)[0]
        b=torch.autograd.grad(d.mean(),rho)[0]
        self.assertTrue(torch.allclose(a,b,atol=1e-13,rtol=1e-13))

    def test_new_gradient_equals_frozen_sigmoid_weight(self):
        log_rho=torch.tensor([[-3.,-4.,-5.],[-8.,-9.,-10.]],dtype=torch.float64,requires_grad=True)
        rho=log_rho.exp();eta=torch.tensor([.001,.00001],dtype=torch.float64)
        new,d=components(rho,eta,'tolerance')
        actual=torch.autograd.grad(new,log_rho)[0]
        expected=torch.sigmoid(d.detach()/BETA)/rho.numel()
        self.assertTrue(torch.allclose(actual,expected,atol=1e-14,rtol=1e-13))
        self.assertGreater(float(actual.max()-actual.min()),.05)

    def test_oversolved_samples_have_less_pressure(self):
        rho=torch.tensor([[1e-6],[1e-6]],dtype=torch.float64)
        eta=torch.tensor([1e-4,1e-1],dtype=torch.float64)
        _,d=components(rho,eta,'tolerance')
        self.assertLess(float(torch.sigmoid(d[0]/BETA)),.01)
        self.assertGreater(float(torch.sigmoid(d[1]/BETA)),.95)

    def test_nonfinite_gradient_cannot_pass_identifiability_gate(self):
        for invalid in [float('nan'),float('inf')]:
            with self.assertRaises(ValueError):gradient_comparison(torch.tensor([1.,2.]),torch.tensor([invalid,1.]))

    def test_solver_gate_rejects_missing_required_grid(self):
        from benchmark_tolerance import gate
        with self.assertRaises(ValueError):gate([],'tolerance',{128:['reference_tail'],256:['tsa16_zero']})

if __name__=='__main__':unittest.main()
