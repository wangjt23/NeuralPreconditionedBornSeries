"""Tiny CPU checks for the precise shared coarse-column interpolation."""
import unittest
import torch
from gpu_direction_jobs import prolong
class ProlongTests(unittest.TestCase):
 def test_retained_mode_and_normalization(self):
  x=torch.tensor([1,1j,-1,-1j],dtype=torch.complex128)
  field=(x[:,None]+.25*x[None,:])[None,None]
  fine=prolong(field,8)
  self.assertLess(float(torch.linalg.vector_norm(fine[...,::2,::2]-field)),1e-13)
  self.assertLess(float(torch.linalg.vector_norm(fine[...,0,::2]-field[...,0,:])),1e-13)
 def test_identical_strict_nyquist_removal(self):
  stripe=torch.tensor([1,-1,1,-1],dtype=torch.complex128)
  field=stripe[None,None,:,None].expand(1,2,4,4).clone()
  self.assertEqual(float(torch.linalg.vector_norm(prolong(field,8))),0.)
 def test_complex_linearity_and_owned_result(self):
  x=torch.arange(16,dtype=torch.float64).reshape(1,1,4,4).to(torch.complex128)
  y=x.flip(-1).clone();a=.25+.5j;b=-.75+.125j
  result=prolong(a*x+b*y,8);truth=a*prolong(x,8)+b*prolong(y,8)
  self.assertLess(float(torch.linalg.vector_norm(result-truth)),1e-12)
  saved=result.clone();prolong(y,8)
  self.assertTrue(torch.equal(result,saved))
if __name__=='__main__':unittest.main()
