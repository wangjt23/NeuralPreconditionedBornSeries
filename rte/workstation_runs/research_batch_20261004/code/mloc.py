"""Prespecified fixed-material local-collision scaling after one reference.

Nominal intended action M=C(x)^-1 C_ref R_ref, C=tI-sP, P=phase*weights_columns.
Exact arithmetic gives I-HM=-S(C^-1 C_ref-I)R_ref if R_ref=(S+C_ref)^-1.
Computed caches/eigendecompositions are floating approximations, not certificates.
No source-dependent coefficients, learning, symmetrization, clipping or tuning.
NumPy supports tiny local tests; Torch tensors select the existing FWI backend.
"""
import numpy as np


class InvalidLocalCollisionScale(ValueError):
    pass


class LocalCollisionScaledReference:
    """Frozen FP64 factors; reference is a caller-owned homogeneous inverse.

    Inputs: phase(L,L), weights(L), t/s(B,1,H,W), reference_t/reference_s(B or1).
    Real inputs must already be binary64; call fields must be complex128.
    Caller must bind R_ref to exactly the same reference material and streaming
    target and charge all setup, reference work, checks, and physical images.
    """
    CONDITION_CAP=100.
    RECONSTRUCTION_CAP=1e-12
    EIGEN_RESIDUAL_CAP=1e-12

    def __init__(self,phase,weights,t,s,reference_t,reference_s,reference):
        if not callable(reference):
            raise TypeError('reference must be an explicit frozen callable')
        self.reference=reference
        self.torch=None
        if type(phase).__module__.startswith('torch'):
            import torch
            self.torch=torch
        xp=self.torch if self.torch is not None else np
        self.xp=xp
        real_dtype=xp.float64
        complex_dtype=xp.complex128
        inputs=[phase,weights,t,s]
        if any(v.dtype!=real_dtype for v in inputs):
            raise TypeError('Material/angular inputs must be represented binary64')
        if self.torch is not None:
            if any(v.device!=phase.device for v in inputs):
                raise ValueError('Frozen tensors must share device')
            rt=xp.as_tensor(reference_t,dtype=real_dtype,device=phase.device).clone().detach()
            rs=xp.as_tensor(reference_s,dtype=real_dtype,device=phase.device).clone().detach()
            copy=lambda v:v.clone().detach()
            cv=lambda v:v.to(complex_dtype)
        else:
            rt=np.array(reference_t,dtype=np.float64,copy=True)
            rs=np.array(reference_s,dtype=np.float64,copy=True)
            copy=lambda v:np.array(v,copy=True)
            cv=lambda v:v.astype(np.complex128)
        L=len(weights)
        if L<1 or tuple(phase.shape)!=(L,L)or t.ndim!=4 or tuple(t.shape)!=tuple(s.shape)or t.shape[1]!=1:
            raise ValueError('Material/angular layout mismatch')
        B,H,W=t.shape[0],t.shape[-2],t.shape[-1]
        count=lambda v:int(v.numel())if self.torch is not None else int(v.size)
        if min(B,H,W)<1 or count(rt)not in (1,B)or count(rs)not in (1,B):
            raise ValueError('Reference material must be scalar or match frozen batch')
        rt,rs=rt.reshape(-1,1,1,1),rs.reshape(-1,1,1,1)
        alltrue=lambda x:bool(x.all())
        if not all(alltrue(xp.isfinite(v))for v in inputs+[rt,rs]):
            raise InvalidLocalCollisionScale('Nonfinite frozen input')
        if not(alltrue(t>0)and alltrue(s>=0)and alltrue(rt>0)and alltrue(rs>=0)):
            raise InvalidLocalCollisionScale('Positive t and nonnegative s required')
        self.t,self.s=copy(t),copy(s)
        self.phase,self.weights=copy(phase),copy(weights)
        self.reference_t,self.reference_s=rt,rs
        self.shape=(B,L,H,W)
        # Retain the complete actually stored matrix. This multiplication and
        # all subsequent operations are computed FP64, not exact products.
        P=cv(self.phase*self.weights.reshape(1,L))
        lam,V=xp.linalg.eig(P)
        Vinv=xp.linalg.inv(V)
        norm=lambda x:xp.linalg.norm(x)
        scalar=lambda x:float(x.item())if hasattr(x,'item')else float(x)
        pnorm=scalar(norm(P))
        denominator=pnorm if pnorm>0 else 1.
        reconstructed=(V*lam.reshape(1,L))@Vinv
        eigen_residual=P@V-V*lam.reshape(1,L)
        reconstruction=scalar(norm(P-reconstructed))/denominator
        residual=scalar(norm(eigen_residual))/(denominator*scalar(norm(V)))
        condition=scalar(xp.linalg.cond(V))
        if not all(np.isfinite(v)for v in (reconstruction,residual,condition))or condition>self.CONDITION_CAP or reconstruction>self.RECONSTRUCTION_CAP or residual>self.EIGEN_RESIDUAL_CAP:
            raise InvalidLocalCollisionScale('Eigenbasis conditioning/reconstruction gate failed')
        local_denominator=cv(self.t)-cv(self.s)*lam.reshape(1,L,1,1)
        numerator=cv(rt)-cv(rs)*lam.reshape(1,L,1,1)
        minimum=scalar(xp.min(xp.abs(local_denominator)))
        if minimum<=0 or not alltrue(xp.isfinite(local_denominator)):
            raise InvalidLocalCollisionScale('Singular/nonfinite local collision denominator')
        filters=numerator/local_denominator
        if not alltrue(xp.isfinite(filters)):
            raise InvalidLocalCollisionScale('Nonfinite prepared scaling')
        self.P,self.eigenvalues,self.V,self.Vinv=P,lam,V,Vinv
        self.filters=filters
        self.diagnostics=dict(condition_2=condition,reconstruction_relative=reconstruction,
            eigen_residual_relative=residual,min_denominator_magnitude=minimum,
            condition_cap=self.CONDITION_CAP,reconstruction_cap=self.RECONSTRUCTION_CAP,
            eigen_residual_cap=self.EIGEN_RESIDUAL_CAP,
            backend='torch'if self.torch is not None else'numpy',
            target='Computed full stored column-weighted P; nominal C^-1 Cref Rref',
            exact_or_interval_certificate=False,parameter_sweep=False)

    def local_scale(self,value):
        if value.dtype!=self.xp.complex128 or tuple(value.shape)!=self.shape:
            raise ValueError('Call field must match frozen shape and complex128 dtype')
        if self.torch is not None and value.device!=self.phase.device:
            raise ValueError('Call device differs from frozen material')
        if not bool(self.xp.isfinite(value).all()):
            raise InvalidLocalCollisionScale('Nonfinite call field')
        coefficients=self.xp.einsum('ma,bahw->bmhw',self.Vinv,value)
        output=self.xp.einsum('am,bmhw->bahw',self.V,coefficients*self.filters)
        if not bool(self.xp.isfinite(output).all()):
            raise InvalidLocalCollisionScale('Nonfinite scaled output')
        return output  # Both contractions allocate owned output storage.

    def __call__(self,rhs):
        if rhs.dtype!=self.xp.complex128 or tuple(rhs.shape)!=self.shape:
            raise ValueError('RHS must match frozen shape and complex128 dtype')
        value=self.reference(rhs)
        return self.local_scale(value)


class LocalCollisionReplacement:
    """FWI Torch convenience wrapper using one prescribed mean-material Rref.

    This source-dependent *application* has material-only frozen coefficients;
    it does not acquire a source-dependent inverse or learn any parameters.
    Original high must supply actual complex128 D64 and binary64 material/phase.
    Caller owns complete setup/timing and original-H acceptance diagnostics.
    """
    def __init__(self,high):
        import torch
        self.torch=torch
        if high.sigma_t.dtype!=torch.float64 or high.sigma_s.dtype!=torch.float64:
            raise TypeError('Original high physical material must be binary64')
        if high.phase_matrix.dtype!=torch.float64 or high.weights.dtype!=torch.float64:
            raise TypeError('Original high phase and weights must be binary64')
        self.t=high.sigma_t.detach().clone();self.s=high.sigma_s.detach().clone()
        self.phase=high.phase_matrix.detach().clone();self.weights=high.weights.detach().clone()
        B=self.t.shape[0];L=len(self.weights);H,W=self.t.shape[-2:]
        self.shape=(B,L,H,W)
        self.tref=self.t.mean((-2,-1),keepdim=True)
        self.sref=self.s.mean((-2,-1),keepdim=True)
        self.symbol=high._stream_symbol(B,torch.complex128,self.t.device).detach().clone()
        if tuple(self.symbol.shape)!=(B,L,H,W)or not bool(torch.isfinite(self.symbol).all())or not bool((self.symbol.real==0).all()):
            raise InvalidLocalCollisionScale('Actual high D64 must be finite and purely imaginary')
        P=(self.phase*self.weights.reshape(1,L)).to(torch.complex128)
        diagonal=self.symbol.permute(0,2,3,1)+self.tref.reshape(B,1,1,1)
        matrix=torch.diag_embed(diagonal)-self.sref.reshape(B,1,1,1,1)*P.reshape(1,1,1,L,L)
        self.inverse=torch.linalg.inv(matrix)
        if not bool(torch.isfinite(self.inverse).all()):
            raise InvalidLocalCollisionScale('Nonfinite homogeneous reference cache')
        # Pointwise closure is a floating diagnostic, never an inverse proof.
        eye=torch.eye(L,dtype=torch.complex128,device=self.t.device)
        closure=matrix@self.inverse-eye
        relative=torch.linalg.matrix_norm(closure,ord='fro')/float(L)**.5
        reference_closure=float(relative.max().item())
        if not np.isfinite(reference_closure)or reference_closure>1e-10:
            raise InvalidLocalCollisionScale('Homogeneous FP64 cache closure gate failed')
        self.local=LocalCollisionScaledReference(self.phase,self.weights,self.t,self.s,
            self.tref.reshape(B),self.sref.reshape(B),self.reference_apply)
        self.diagnostics=dict(self.local.diagnostics,
            reference_material_policy='Fixed actual spatial mean of t/s; no sweep',
            reference_matrix_closure_max=reference_closure,reference_matrix_closure_cap=1e-10,
            reference_inverse_bytes=self.inverse.numel()*self.inverse.element_size(),
            local_filter_bytes=self.local.filters.numel()*self.local.filters.element_size(),
            reference_dtype='complex128',phase_eigenvalues_preserved_complex=True,
            full_original_phase_used=True,stored_original_stream_symbol_used=True,
            homogeneous_reference_action='Actual captured D64 plus computed mean collision',
            physical_action_formula='Mloc=C(x)^-1 Cref Rref; intended exact algebra only')
        del matrix,closure,relative

    def reference_apply(self,rhs):
        torch=self.torch
        if rhs.dtype!=torch.complex128 or tuple(rhs.shape)!=self.shape or rhs.device!=self.t.device:
            raise ValueError('Reference RHS must match frozen complex128 shape/device')
        spectrum=torch.fft.fftn(rhs,dim=(-2,-1),norm='backward')
        vectors=spectrum.permute(0,2,3,1).unsqueeze(-1)
        output=(self.inverse@vectors).squeeze(-1).permute(0,3,1,2)
        return torch.fft.ifftn(output,dim=(-2,-1),norm='backward')

    def __call__(self,rhs):
        return self.local(rhs)
