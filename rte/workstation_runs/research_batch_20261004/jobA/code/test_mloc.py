"""Tiny independent Mloc algebra/ownership tests; no project or GPU inputs."""
import unittest
import importlib.util
import numpy as np
from mloc import LocalCollisionScaledReference,LocalCollisionReplacement,InvalidLocalCollisionScale


def inputs():
    phase=np.array([[1.,.25],[.5,.75]],dtype=np.float64)
    weights=np.array([.5,.25],dtype=np.float64)
    t=np.array([[[[2.,2.25],[1.75,2.5]]]],dtype=np.float64)
    s=np.array([[[[.25,.5],[.125,.375]]]],dtype=np.float64)
    rhs=np.array([[[[1.+.25j,-.5+.125j],[.75-.25j,.5+.5j]],
                   [[.25-.125j,.125+.5j],[-.25+.75j,1.-.125j]]]],dtype=np.complex128)
    return phase,weights,t,s,rhs


class MlocNumpyTests(unittest.TestCase):
    def test_full_stored_noncirculant_collision_against_dense_solve(self):
        phase,w,t,s,r=inputs();tr=float(t.mean());sr=float(s.mean())
        m=LocalCollisionScaledReference(phase,w,t,s,tr,sr,lambda x:x.copy())
        P=phase*w[None];Cref=tr*np.eye(2)-sr*P
        truth=np.empty_like(r)
        for i,j in np.ndindex((2,2)):
            C=t[0,0,i,j]*np.eye(2)-s[0,0,i,j]*P
            truth[0,:,i,j]=np.linalg.solve(C,Cref@r[0,:,i,j])
        np.testing.assert_allclose(m(r),truth,rtol=2e-14,atol=2e-14)
        self.assertFalse(np.allclose(P[1],np.roll(P[0],1)))

    def test_general_complex_eigenvalues_are_retained(self):
        phase=np.array([[.2,-.9],[.9,.2]],dtype=np.float64);w=np.ones(2)
        _,_,t,s,r=inputs()
        m=LocalCollisionScaledReference(phase,w,t,s,2.,.25,lambda x:x.copy())
        self.assertGreater(np.max(np.abs(m.eigenvalues.imag)),.8)
        Cref=2*np.eye(2)-.25*phase
        for i,j in np.ndindex((2,2)):
            expected=np.linalg.solve(t[0,0,i,j]*np.eye(2)-s[0,0,i,j]*phase,Cref@r[0,:,i,j])
            np.testing.assert_allclose(m(r)[0,:,i,j],expected,rtol=2e-14,atol=2e-14)

    def test_fixed_material_complex_linearity_and_owned_output(self):
        phase,w,t,s,r=inputs();reference=lambda x:x*.75
        m=LocalCollisionScaledReference(phase,w,t,s,2.,.25,reference)
        z=m(r);saved=z.copy();q=r[:,::-1].copy();alpha=.25+.5j;beta=-.75+.125j
        np.testing.assert_allclose(m(alpha*r+beta*q),alpha*m(r)+beta*m(q),rtol=2e-14,atol=2e-14)
        m(q);np.testing.assert_array_equal(z,saved)
        t[:]=10.;phase[:]=0.;w[:]=0.
        np.testing.assert_array_equal(m(r),saved)
        self.assertFalse(np.shares_memory(z,r))

    def test_exact_intended_defect_identity_with_dense_streaming(self):
        phase,w,t,s,r=inputs();P=phase*w[None];tr=float(t.mean());sr=float(s.mean())
        # Four spatial cells, with a real skew matrix on each angular channel.
        D=np.array([[0.,1.,0.,-.5],[-1.,0.,.25,0.],[0.,-.25,0.,.75],[.5,0.,-.75,0.]])
        S=np.kron(np.eye(2),D).astype(np.complex128)
        C=np.zeros((8,8));Cref=np.kron(tr*np.eye(2)-sr*P,np.eye(4))
        for i in range(4):
            for a in range(2):
                for b in range(2):
                    C[a*4+i,b*4+i]=t.ravel()[i]*(a==b)-s.ravel()[i]*P[a,b]
        R=np.linalg.inv(S+Cref)
        ref=lambda x:(R@x.reshape(-1)).reshape(x.shape)
        m=LocalCollisionScaledReference(phase,w,t,s,tr,sr,ref)
        y=ref(r).reshape(-1);z=m(r).reshape(-1)
        np.testing.assert_allclose(r.reshape(-1)-(S+C)@z,-S@(z-y),rtol=3e-13,atol=3e-13)
        self.assertGreater(np.linalg.norm(S@(z-y)),1e-4)

    def test_bad_eigenbasis_and_invalid_inputs_fail_without_repair(self):
        _,w,t,s,r=inputs();defective=np.array([[1.,1.],[0.,1.]])
        with self.assertRaises(InvalidLocalCollisionScale):
            LocalCollisionScaledReference(defective,np.ones(2),t,s,2.,.25,lambda x:x)
        phase,w,t,s,r=inputs()
        with self.assertRaises(TypeError):
            LocalCollisionScaledReference(phase.astype(np.float32),w,t,s,2.,.25,lambda x:x)
        m=LocalCollisionScaledReference(phase,w,t,s,2.,.25,lambda x:x)
        with self.assertRaises(ValueError):m(r.astype(np.complex64))
        with self.assertRaises(InvalidLocalCollisionScale):
            bad=r.copy();bad.flat[0]=complex(np.nan,0);m(bad)


@unittest.skipUnless(importlib.util.find_spec('torch')is not None,'Existing Torch absent on auxiliary host')
class MlocTorchTests(unittest.TestCase):
    def test_torch_general_kernel_matches_independent_dense_numpy(self):
        import torch
        phase,w,t,s,r=inputs()
        arrays=[torch.tensor(v)for v in (phase,w,t,s,r)]
        m=LocalCollisionScaledReference(*arrays[:4],float(t.mean()),float(s.mean()),lambda x:x.clone())
        control=LocalCollisionScaledReference(phase,w,t,s,float(t.mean()),float(s.mean()),lambda x:x.copy())
        np.testing.assert_allclose(m(arrays[4]).numpy(),control(r),rtol=3e-13,atol=3e-13)
        self.assertLessEqual(m.diagnostics['condition_2'],100.)

    def test_convenience_reference_physics_and_formula(self):
        import torch
        phase,w,t,s,r=inputs()
        class High:
            sigma_t=torch.tensor(t);sigma_s=torch.tensor(s)
            phase_matrix=torch.tensor(phase);weights=torch.tensor(w)
            def _stream_symbol(self,batch,dtype,device):
                data=torch.tensor([[[[0.,1.],[-2.,-.5]],[[0.,-.25],[.5,-1.]]]],device=device)
                return (1j*data).to(dtype)
        high=High();m=LocalCollisionReplacement(high);rhs=torch.tensor(r)
        y=m.reference_apply(rhs);z=m(rhs)
        def S(v):return torch.fft.ifftn(m.symbol*torch.fft.fftn(v,dim=(-2,-1)),dim=(-2,-1))
        # Explicit contraction avoids an accidentally diagonal angular oracle.
        scatter=lambda v:torch.einsum('ac,bchw->bahw',high.phase_matrix.to(v.dtype),v*high.weights.reshape(1,2,1,1))
        H=lambda v:S(v)+high.sigma_t*v-high.sigma_s*scatter(v)
        ref_image=S(y)+m.tref*y-m.sref*scatter(y)
        np.testing.assert_allclose(ref_image.numpy(),r,rtol=3e-13,atol=3e-13)
        np.testing.assert_allclose((rhs-H(z)).numpy(),(-S(z-y)).numpy(),rtol=5e-13,atol=5e-13)
        self.assertLess(m.diagnostics['reference_matrix_closure_max'],1e-10)


if __name__=='__main__':unittest.main(verbosity=2)
