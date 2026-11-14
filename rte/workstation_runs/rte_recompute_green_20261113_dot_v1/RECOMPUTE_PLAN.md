# Exact factor-storage elision: proof, scoped obstacles, one fixed comparison

## Scope and alternatives

Keep the actual N128, k=0 background and original residual target 1e-10/max 3 repairs. Keep the previously fixed actual-frequency skew-preserving derivative tree and absolute error budget. No rank, leaf, precision or mode sweep. No physical RHS or full dense Green inverse. Source/background hashes must equal previous operator-first work.

Three concepts were audited before this check:

1. **Selected executable attribution check:** eliminate persistent W_L,W_R,M_L,M_R by exact recomputation, retaining only Schur LU and leaf LU. This changes factor storage/application, not Gtilde, so it retains the proven model q. It cannot be called a low-rank improvement or elimination of all Schur storage.
2. **Eliminate Schur LU too:** use matrix-free interface iterations. Each S-vector product needs two child inverse actions. Original-G accretivity alone is not an interface convergence bound: nonunitary interface-coordinate changes can change Euclidean conditioning without changing G. The old root norm already gives ||I-S||2 >= (||S||F-sqrt(R))/sqrt(R) >1, ruling out a Euclidean norm-contractive identity/Neumann certificate. This does not prove spectral-radius divergence or rule out another interface preconditioner. Not implemented.
3. **Exploit angular closure/nested bases:** in actual stored data the twelve normal velocities are distinct and every phase entry is nonzero. Polynomial interpolation in the diagonal velocity matrix gives each E_ii; E_ii K E_jj / K_ij gives E_ij. Thus these generators span the full 12-by-12 matrix algebra. Primitive near-angular-banding does not imply a fixed low-dimensional exact angular closure after inversion or Schur products. This does not rule out approximate tensor compression. Nested HSS/ULV bases are a genuinely different representation, but restriction of a root rank33 basis to 32 cells can be full row rank; old sibling ranks14/8 cannot be reused without checking inherited interfaces. We will inspect four fixed restrictions, not build an HSS solver.

## Exact low-storage identity

For Gtilde=[[A,U_L V_R*],[U_R V_L*,B]], define M_L=V_L*A^-1*U_L, M_R=V_R*B^-1*U_R and S=I-M_L M_R. Store S LU, not W or M. For an input (qL,qR):

1. yR=B^-1 qR.
2. zL=A^-1(qL-U_L V_R* yR).
3. a=S^-1 V_L* zL.
4. tR=B^-1 U_R a.
5. xR=yR-tR; xL=zL+A^-1 U_L V_R* tR.

This is algebraically identical to the previous cached factor. Tensor lifts/projections use scalar normal factors and the actual diagonal velocity; they never retain expanded U or W. Two left and two right child solves are required. For this exact evaluation circuit, the complete depth4 tree has 256 leaf visits rather than16. That is not a 16x total arithmetic or timing prediction, because cached W/M multiplications are removed.

The persistent tensor layout predicts about7.18 MB plus the actual velocity vector. It still stores Schur LU. Construction must not silently materialize full W/M and free them later: build each Schur matrix in a fixed block of12 columns, one scalar-basis coordinate times all angles. Use child inverse actions to compute I-M_L M_R on those columns. Charge all streamed setup work, peak RSS and factor storage.

The old q~3.38e-4 bound remains a model theorem for exact Gtilde^-1. Floating recomputation introduces its own per-input E=I-Gtilde Bhat, and the same q+(1+q)||E|| accounting and mandatory original residual checks apply. No claim of a uniform machine certificate.

## Fixed comparison

Three sequential fresh CPU0 processes, one torch/OMP/MKL/OpenBLAS thread, CUDA disabled:

- cached previous Green factor (attribution control),
- new recomputing Green factor,
- current AngularBandMode (decision control).

Each pays its own physical/operator setup and method-specific construction. Angular-band does not pay a derivative-tree build it does not use. Cached/recomputing use identical derivative factors, checked by byte hashes. No three factors share a worker or count each other's retained tensors as their memory. Shared prototype imports and process startup are reported, not hidden.

Same3 fixed normalized complex random columns + exact zero +1e-200 nonzero column. Three unchanged repeated checked applications per factor, all included, no warmup exclusion. Raw approximate-operator defects are separate diagnostics. Save the accepted cached result as a small reference for later equality checks, never feed it to construction. Record exact owned tensor bytes, process high-water RSS, current RSS phases, setup, each application and original checks. Instrument one separate raw application to verify recursive visit counts, and charge that extra diagnostic. Do not turn visit counts into runtime claims.

Cached process additionally pays for the following source-independent fixed structure diagnostics: actual angular coefficient distinctness/nonzero phase and commutator; original root S norm and the Neumann norm obstruction; angular-Fourier off-band energy of root W and S-I (diagnostic only, not a proposed inverse compression); four predetermined32-cell restrictions of root scalar bases. These are not source selection or parameter scans.

A deterministic small recursive algebra test must compare cached vs recomputed vs original dense solve before the real comparison. Production code unchanged. Preserve any failed run. Stop after these three processes, regardless of whether memory savings win. No mode expansion, tensor compression or new interface iteration follows automatically.

## Provenance

The classical HSS/ULV alternative is described by Chandrasekaran, Gu and Pals, SIAM J. Matrix Anal. Appl. 28 (2006), DOI10.1137/S0895479803436652. Its small nested-rank premises have not been established here. This is not the older Hermitian/skew-Hermitian splitting also called HSS in this RTE repository.
