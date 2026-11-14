# Selected coordinate representation: angular cyclic blocks with original-G repair

## Debate round 1: angular versus normal Fourier coordinates
Actual stored angular coefficients are not perfectly circulant/equispaced: node errors reach2.88e-7rad and the angular cyclic-band projection changes the full mode block by about1.26e-7 relative Frobenius norm. Therefore an angular block-tridiagonal solve alone is not a bitwise exact replacement. However, the measured discarded-operator norm bound is about2.1e-4 against original coercivity0.04996, giving a conditional real-arithmetic residual-repair contraction bound below0.00435 for modes0,1,8.

The normal material profiles are qualitatively different: all128 Fourier coefficients are nonzero and half-bandwidth16/32 drops5.73%/1.70% of coefficient L2. The two-layer transition is not a few-harmonic material. Reject normal narrow-band direct factorization as an exact representation and do not implement it.

## Debate round 2: how to exploit the selected structure without changing the equation
Transform using the actual stored12-angle matrices. Keep diagonal and cyclic nearest-neighbor blocks from those transformed matrices, including their small non-ideal coefficients. Factor the resulting open block chain with Ny-by-Ny block LU and restore its two periodic corner blocks with a2Ny Woodbury correction. This Woodbury correction handles angular wrap boundaries, not a fictitious rank1 scattering kernel.

Apply the original G_k matrix-free using the stored normal FFT frequencies, original12-angle full phase/weights, original directions and layer material. First verify this action against the dense1536 block. After the cyclic-band solve, allow at most3 residual corrections with that original action, target1e-10, all charged. Do not relax the target or silently drop failed vectors. Dense LU remains the reference action and cost control. Reuse the exact prior seed20261110 and three complex RHS columns for each mode[0,1,8]. No full17-mode build, whole-PDE timing or GPU yet.

The possible gain comes from replacing one1536-dimensional factor with12 factors of128 plus a256-dimensional corner system and a small original-operator repair. Failure modes include unstable block elimination/corner correction, greater-than-expected stored-geometry defect, or repair/application costs eliminating the benefit. Pass only if every tested original-G residual meets1e-10 and the dense action agrees; report the representation as tolerance-certified, not bitwise exact.
