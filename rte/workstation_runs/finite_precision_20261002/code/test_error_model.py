import math,unittest
from fractions import Fraction
from error_model import ceil_sqrt,stored_scales,conditional_action_bound,prospective_budget

class ErrorModelTests(unittest.TestCase):
 def test_exact_sqrt_upper_and_outward_coercivity(self):
  q=Fraction(2);v=ceil_sqrt(q);self.assertGreaterEqual(v*v,q);self.assertLess((v-Fraction(1,1<<80))**2,q)
  s=stored_scales([[1.,0.],[0.,1.]],[-2.,1.],[-2.,1.],[3.,4.],[1.,2.],[[1.,0.],[0.,1.]],[.5,.5])
  self.assertEqual(s['scattering_norm_upper'],.5);self.assertEqual(s['coercivity_lower'],2.5)
  self.assertGreaterEqual(s['operator_norm_upper'],7.)
 def test_symbol_and_rounding_terms_survive_zero_fft64(self):
  s=stored_scales([[1.,0.]],[-2.,1.],[-2.,1.],[3.],[1.],[[1.]],[1.])
  b=conditional_action_bound(s,64,0.,0.)
  self.assertGreater(b['streaming'],0.);self.assertGreater(b['final_additions'],0.)
  self.assertGreater(b['total'],b['symbol_formation'])
 def test_forward_bound_is_monotone_without_fitting(self):
  s=stored_scales([[1.,0.]],[-2.,1.],[-2.,1.],[3.],[1.],[[1.]],[1.])
  a=conditional_action_bound(s,32,1e-7,1e-7);b=conditional_action_bound(s,32,1e-6,1e-6)
  self.assertGreater(b['total'],a['total']);self.assertFalse(b['certified_backend_contract'])
 def test_tolerance_linked_budget_is_safe_and_conditional(self):
  s=stored_scales([[1.,0.]],[-2.,1.],[-2.,1.],[3.],[1.],[[1.]],[1.]);p=prospective_budget(s)
  self.assertLessEqual(Fraction.from_float(p['allowed_outer_absolute_error_per_field_norm']),Fraction(1,100)*Fraction(1,100000)*Fraction(2)/(1+Fraction(1,100000)))
  eta=p['equal_forward_inverse_fft64_contract_required'];self.assertIsNotNone(eta)
  self.assertLessEqual(conditional_action_bound(s,64,eta,eta)['total'],p['allowed_outer_absolute_error_per_field_norm'])
  self.assertFalse(p['empirical_reference_checks_can_certify_this_contract'])
 def test_noncoercive_context_is_not_admitted(self):
  s=stored_scales([[1.,0.]],[-2.,1.],[-2.,1.],[1.],[2.],[[1.]],[1.])
  with self.assertRaises(AssertionError):prospective_budget(s)

if __name__=='__main__':unittest.main(verbosity=2)
