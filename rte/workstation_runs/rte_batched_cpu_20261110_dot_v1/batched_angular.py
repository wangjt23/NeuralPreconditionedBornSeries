"""Independent mode-batched angular representation; no conjugate-mode reuse."""
import time
import torch

MODES = list(range(-8, 9))

class BatchedCyclicFactor:
    def __init__(self, diag, upper, lower, first_last, last_first):
        self.m, self.n, _ = diag[0].shape
        self.l = len(diag)
        self.lower, self.lu, self.piv, self.Z = lower, [], [], []
        start = time.perf_counter(); current = diag[0]; stages = []
        for i in range(self.l):
            tick = time.perf_counter()
            lu, piv = torch.linalg.lu_factor(current)
            self.lu.append(lu); self.piv.append(piv)
            lu_seconds = time.perf_counter() - tick; tick = time.perf_counter()
            if i + 1 < self.l:
                z = torch.linalg.lu_solve(lu, piv, upper[i])
                self.Z.append(z)
                current = diag[i+1] - lower[i] @ z
            stages.append(dict(angle_stage=i, batched_LU_seconds=lu_seconds, solve_and_update_seconds=time.perf_counter()-tick))
        chain = time.perf_counter() - start; tick = time.perf_counter()
        U = torch.zeros((self.m,self.l,self.n,2*self.n), dtype=diag[0].dtype, device=diag[0].device)
        U[:,0,:,:self.n] = first_last
        U[:,-1,:,self.n:] = last_first
        self.W = self.open_solve(U)
        H = torch.eye(2*self.n, dtype=U.dtype, device=U.device)[None] + torch.cat((self.W[:,-1],self.W[:,0]),dim=1)
        self.corner_lu, self.corner_piv = torch.linalg.lu_factor(H)
        self.setup = dict(chain_seconds=chain, corner_seconds=time.perf_counter()-tick, chain_stages=stages)
        tensors = self.lower+self.lu+self.piv+self.Z+[self.W,self.corner_lu,self.corner_piv]
        self.bytes = sum(v.numel()*v.element_size() for v in tensors)

    def open_solve(self, b):
        y = []
        for i in range(self.l):
            value = b[:,i] if i == 0 else b[:,i] - self.lower[i-1] @ y[-1]
            y.append(torch.linalg.lu_solve(self.lu[i], self.piv[i], value))
        x = [None]*self.l; x[-1] = y[-1]
        for i in range(self.l-2,-1,-1): x[i] = y[i] - self.Z[i] @ x[i+1]
        return torch.stack(x, dim=1)

    def __call__(self, b):
        y = self.open_solve(b)
        c = torch.linalg.lu_solve(self.corner_lu, self.corner_piv, torch.cat((y[:,-1],y[:,0]),dim=1))
        return y - self.W @ c[:,None]

class BatchedAngularModes:
    def __init__(self, low, high):
        from layered_inverse import background_from_material
        start = time.perf_counter(); self.shape = low.source.shape
        axis,t,s,_ = background_from_material(low.sigma_t,low.sigma_s)
        if axis != 0: raise ValueError('This qualification only covers the frozen normal axis0')
        self.t,self.s = t[0,0,:,0].double(),s[0,0,:,0].double()
        self.n,self.l,self.m = len(self.t),high.n_angles,len(MODES)
        device = high.source.device
        self.indices = torch.tensor([k % self.n for k in MODES],device=device,dtype=torch.long)
        self.nk = high.kx.flatten(); self.tk = high.ky.flatten()[self.indices]
        self.normal,self.tangent = high.directions[:,0],high.directions[:,1]
        self.K = (high.phase_matrix*high.weights[None]).to(torch.complex128)
        F = torch.fft.fft(torch.eye(self.l,dtype=torch.complex128,device=device),dim=0,norm='ortho')
        Cn = F @ torch.diag(self.normal.to(torch.complex128)) @ F.conj().T
        Ct = F @ torch.diag(self.tangent.to(torch.complex128)) @ F.conj().T
        Kf = F @ self.K @ F.conj().T
        I = torch.eye(self.n,dtype=torch.complex128,device=device)
        D = torch.fft.ifft(1j*self.nk[:,None]*torch.fft.fft(I,dim=0),dim=0)
        S,T = torch.diag(self.s).to(torch.complex128),torch.diag(self.t).to(torch.complex128)
        def block(i,j):
            return Cn[i,j]*D[None] + 1j*self.tk[:,None,None]*Ct[i,j]*I[None] - Kf[i,j]*S[None] + (T[None] if i==j else 0.)
        diag = [block(i,i) for i in range(self.l)]
        upper = [block(i,i+1) for i in range(self.l-1)]
        lower = [block(i+1,i) for i in range(self.l-1)]
        first_last,last_first = block(0,self.l-1),block(self.l-1,0)
        assembly = time.perf_counter()-start
        self.factor = BatchedCyclicFactor(diag,upper,lower,first_last,last_first)
        self.setup = dict(assembly_seconds=assembly,**self.factor.setup,total_seconds=time.perf_counter()-start)
        self.profiles = []
        self.bytes = self.factor.bytes + sum(v.numel()*v.element_size() for v in [self.t,self.s,self.indices,self.nk,self.tk,self.normal,self.tangent,self.K])

    def original(self, q):
        x = q.reshape(self.m,self.l,self.n,-1)
        dx = torch.fft.ifft(1j*self.nk[None,None,:,None]*torch.fft.fft(x,dim=2),dim=2)
        v = self.normal[None,:,None,None]*dx + 1j*self.tk[:,None,None,None]*self.tangent[None,:,None,None]*x + self.t[None,None,:,None]*x - self.s[None,None,:,None]*torch.einsum('lm,kmnr->klnr',self.K,x)
        return v.reshape_as(q)

    def raw(self, q):
        transformed = torch.fft.fft(q.reshape(self.m,self.l,self.n,-1),dim=1,norm='ortho')
        return torch.fft.ifft(self.factor(transformed),dim=1,norm='ortho').reshape_as(q)

    def repaired(self, q, tolerance=1e-10, max_corrections=3):
        if q.ndim != 3 or q.shape[:2] != (self.m,self.l*self.n): raise ValueError('Expected [17,1536,columns]')
        active = (q!=0).any(dim=1)
        x = torch.zeros_like(q); counts = torch.zeros_like(active,dtype=torch.int64); trace = []
        if not bool(active.any()):
            return x,dict(success=True,corrections=counts.tolist(),residual_trace=[torch.zeros_like(active,dtype=q.real.dtype).tolist()],zero_columns=int((~active).sum()),batched_raw_calls=0,batched_original_calls=0)
        scale = q.abs().amax(dim=1)
        # Ones replace denominators ONLY for mathematical zero columns, never small nonzero columns.
        safe_scale = torch.where(active,scale,torch.ones_like(scale))
        den = (q/safe_scale[:,None]).norm(dim=1)
        safe_den = torch.where(active,den,torch.ones_like(den))
        x = torch.where(active[:,None],self.raw(q),x)
        for count in range(max_corrections+1):
            r = q-self.original(x)
            ratio = torch.where(active,(r/safe_scale[:,None]).norm(dim=1)/safe_den,torch.zeros_like(den))
            trace.append(ratio.tolist()); pending = active & (~torch.isfinite(ratio) | (ratio>tolerance))
            if not bool(pending.any()):
                return x,dict(success=True,corrections=counts.tolist(),residual_trace=trace,zero_columns=int((~active).sum()),batched_raw_calls=1+count,batched_original_calls=1+count)
            if count < max_corrections:
                x += self.raw(torch.where(pending[:,None],r,torch.zeros_like(r)))
                counts += pending
        return x,dict(success=False,corrections=counts.tolist(),residual_trace=trace,zero_columns=int((~active).sum()),batched_raw_calls=1+max_corrections,batched_original_calls=1+max_corrections)

    def apply(self, r):
        if r.shape[0] != 1: raise ValueError('Physical RHS remain sequential, batch size1')
        f = torch.fft.fft(r,dim=-1,norm='ortho')
        q = f.index_select(-1,self.indices).permute(3,1,2,0).reshape(self.m,self.l*self.n,1)
        z,st = self.repaired(q,tolerance=1e-10,max_corrections=3); self.profiles.append(st)
        if not st['success']: raise RuntimeError('Batched original-G repair target failed')
        out = torch.zeros_like(f)
        out.index_copy_(-1,self.indices,z.reshape(self.m,self.l,self.n,1).permute(3,1,2,0))
        return torch.fft.ifft(out,dim=-1,norm='ortho')
