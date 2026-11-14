"""Independent exact/adversarial CPU tests, with no RTE/GPU/data imports."""
from fractions import Fraction as F
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import interval_fft as p
import stagewise_fft as s
from test_interval_fft import fixture, exact_action


def f(x):
    return F.from_float(float(x))


class IndependentStagewiseAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=shutil.which('c++')or shutil.which('g++')
        if compiler is None:
            raise RuntimeError('Existing C++ compiler required')
        cls.tmp=tempfile.TemporaryDirectory(prefix='rte_stage_ball_audit_')
        lib=Path(cls.tmp.name)/('audit.dylib'if sys.platform=='darwin'else'audit.so')
        command=[compiler,'-O2','-std=c++17','-fno-fast-math','-ffp-contract=off',
                 '-shared','-fPIC',str(Path(__file__).with_name('cpu_interval_primitives.cpp')),
                 '-o',str(lib)]
        subprocess.run(command,check=True,capture_output=True,text=True)
        cls.guard=s.load_native_cpu_backend(lib)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_zero_center_does_not_erase_incoming_radius(self):
        zero=np.zeros((1,2,4,4),dtype=np.complex128)
        E=F(1,8)
        for inverse,expected in [(False,4*E),(True,E/4)]:
            ball=s.fft2_ball(s.StagewiseBall(zero,float(E)),inverse=inverse)
            self.assertGreaterEqual(f(ball.radius_upper),expected)
            self.assertTrue(np.all(ball.center==0))
            previous=float(E)
            for row in ball.stages:
                self.assertEqual(row['incoming_radius_upper'],previous)
                previous=row['radius_upper']
            self.assertEqual(ball.radius_upper,previous)

    def test_exact_two_point_old_operands_and_inverse_axis_scaling(self):
        x=np.array([[.5+.25j,-.125+.5j],[.75-.25j,.25-.125j]],dtype=np.complex128)
        for inverse in [False,True]:
            ball=s.fft_ball(x,axis=0,inverse=inverse)
            error=F(0)
            for j in range(2):
                for k,sign in [(0,1),(1,-1)]:
                    for component in ['real','imag']:
                        expected=(f(getattr(x[0,j],component))+sign*f(getattr(x[1,j],component)))/(2 if inverse else 1)
                        error+=(expected-f(getattr(ball.center[k,j],component)))**2
            self.assertLessEqual(error,f(ball.radius_upper)**2)
            self.assertFalse(np.shares_memory(x,ball.center))
            self.assertTrue(any(r['kind']=='inverse_normalization'for r in ball.stages) == inverse)

    def test_exact_nonnormal_heterogeneous_physical_action_ball(self):
        args=fixture(); truth=exact_action(*args); before=args[0].copy()
        ball=s.physical_action_ball(*args)
        error=F(0)
        for a,i,j in np.ndindex((3,4,4)):
            target=truth[a][i][j];z=ball.center[0,a,i,j]
            error+=(target[0]-f(z.real))**2+(target[1]-f(z.imag))**2
        self.assertLessEqual(error,f(ball.radius_upper)**2)
        self.assertTrue(np.array_equal(args[0],before))
        self.assertEqual([r['kind']for r in ball.stages].count('inverse_normalization'),2)
        self.assertEqual([r['kind']for r in ball.stages].count('stored_D64_multiplication'),1)

    def test_ball_norm_and_boundary_are_outward(self):
        center=np.array([3.+4.j],dtype=np.complex128)
        ball=s.StagewiseBall(center,.25)
        lo,hi=s.ball_norm_bounds(ball)
        self.assertLessEqual(f(lo),F(19,4))
        self.assertGreaterEqual(f(hi),F(21,4))
        center[0]=0
        self.assertEqual(ball.center[0],3.+4.j)
        zero=s.StagewiseBall(np.zeros(2,dtype=np.complex128),.25)
        self.assertEqual(s.ball_norm_bounds(zero),(0.,.25))

    def test_individual_accuracy_does_not_imply_paired_field_gate(self):
        # H=diag(1,1/64), q=(1,0), with the same value in four cells.
        shape=(1,2,2,2);q=np.zeros(shape,dtype=np.complex128);q[:,0]=1.
        baseline=q.copy();candidate=q.copy()
        baseline[:,1]=2.**-14;candidate[:,1]=-2.**-14
        symbol=np.zeros(shape,dtype=np.complex128)
        t=np.ones((1,1,2,2));scattering=t*(63/64)
        phase=np.array([[0.,0.],[0.,1.]]);weights=np.ones(2)
        for x in [baseline,candidate]:
            r=s.certify_residual(q,x,symbol,t,scattering,phase,weights)
            self.assertTrue(r['accuracy_accepted_without_paired_field'])
            self.assertTrue(r['accepted']);self.assertFalse(r['paired_field_gate_established'])
        pair=s.paired_field_check(candidate,baseline)
        self.assertFalse(pair['paired_field_accepted'])
        self.assertTrue(pair['paired_field_rigorous_rejection'])
        combined=s.certify_residual(q,candidate,symbol,t,scattering,phase,weights,baseline=baseline)
        self.assertTrue(combined['accuracy_accepted_without_paired_field'])
        self.assertFalse(combined['accepted'])
        self.assertTrue(combined['paired_field_gate_required'])

    def test_represented_gain_pairs_and_baseline_denominator(self):
        baseline=np.array([1.+.25j,-.5+.125j,2.-.25j],dtype=np.complex128)
        same=s.paired_field_check(baseline,baseline)
        self.assertTrue(same['paired_field_accepted'])
        positive=s.paired_field_check((1+2.**-20)*baseline,baseline)
        negative=s.paired_field_check((1+2.**-8)*baseline,baseline)
        self.assertTrue(positive['paired_field_accepted'])
        self.assertFalse(positive['paired_field_rigorous_rejection'])
        self.assertFalse(negative['paired_field_accepted'])
        self.assertTrue(negative['paired_field_rigorous_rejection'])
        undefined=s.paired_field_check(np.zeros_like(baseline),np.zeros_like(baseline))
        self.assertFalse(undefined['paired_field_accepted'])
        self.assertIsNone(undefined['paired_field_relative_upper'])

    def test_public_paired_checker_requires_native_guard(self):
        with patch.object(p,'_NATIVE',None):
            with self.assertRaises(RuntimeError):
                s.paired_field_check(np.ones(2,dtype=np.complex128),np.ones(2,dtype=np.complex128))

    def test_exact_radius_input_is_not_silently_rounded_down(self):
        value=F(1,3)
        try:
            ball=s.StagewiseBall(np.zeros(2,dtype=np.complex128),value)
        except TypeError:
            return  # An explicit represented-binary64-only contract is valid.
        self.assertGreaterEqual(f(ball.radius_upper),value)


if __name__=='__main__':
    unittest.main(verbosity=2)
