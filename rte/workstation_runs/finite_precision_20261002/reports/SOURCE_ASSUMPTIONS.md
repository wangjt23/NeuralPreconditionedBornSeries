# Source authority and unresolved arithmetic contracts

Read-only inspection used existing local cached original sources and preserved authored code; no original source was copied/exported, no operator/action was run, and no field/data was generated. Line numbers below refer to that existing snapshot. Source hashes make this inspection identifiable, not a claim that source alone establishes a compiled-kernel error bound.

| Cached file | Inspected contract | SHA256 |
|---|---|---|
| `remote_reference/rte/models.py` | lines182–203: weights→angular einsum→σs scattering; FP32/FP64 stored-geometry stream assembly; backward-normalized FFT pair; final term addition/subtraction | `6cd887b332ed48409bcffa2d2bad6ca9654c97ff7071c13a6443278f583b04d1` |
| `remote_reference/rte/mixed_precision_transport.py` | line9: original buffers/material/source promoted; line40: actual residual128→64 model→128 cast path | `9c6793ed0bffad5bf803253ac0ada097026c653c02f6498e82fb0974dfc9765a` |
| `remote_reference/rte/transport_baselines.py` | lines52/64: TSA primitive and fixed-stage defect iteration; actual low A inside factor; reference inverse cache fixed by preparation | `05f6984eb24962409ff25e42313e9c1ac740df86b59ef7b498faff715cfd62ec` |
| `remote_reference/rte/augmented_fgmres.py` | line32: real-stacked coefficients; lines55/101: split residual/candidate; line138: final original-RHS residual refresh | `21ddfb628d35738ca88844d6c8587b0b8809956ed000bc7a859237d12b6bc742` |
| `remote_reference/rte/npbs_unet_ablation.py` | original complex traced solver line2372 and real wrapper line2252; CLI-only TF32 controls elsewhere do not establish this runner's runtime state | `4da8fb3a2998a02b45d508f7e8387b12c99849cfc06ff8363327b667aa17d5f2` |
| `deliverables/complex_krylov/complex_augmented_fgmres.py` | lines27/42: authored complex SVD/MGS companion; lines108/176: split/candidate; line232: fresh final physical residual | `d517306d65f796b923c08735be235f7d56b615614e8a068fcd35349a592bfe4e` |

Preserved old policy SHA256 `acd73134c63839726d3b699c60e795b8da90d14709a9529fb94e905ce59ebd93` and result-settings SHA256 `f00e6a06a58d3f9b1b0221179edffd3adaf79ca2ba8d6d415770c65d3129e5a1` remain unchanged. They establish an executed mandatory-entry failure and zero full solves, not the contracts below.

| Needed quantity/assumption | What source/primary documentation supports | What is not supplied |
|---|---|---|
| Fixed nominal complex-linear physical H | FFT multiplier and fixed collision algebra; promoted original data | Exact bitwise floating linearity, continuum accuracy or real-subspace invariance |
| L-versus-H symbol difference Δs | Both use original stored geometry, assembled in differing precisions | A declaration that the completed FP32 symbol was simply promoted or equals physical H |
| β,a,b,Λ,m on each stored material | Finite-dimensional definitions; skew streaming; Hermitian-part collision estimate | Certified rounded evaluations/enclosures or numerical values newly computed here |
| εF/εI for32 and64 | cuFFT qualitative logarithmic accuracy growth; fixed-plan/version/GPU determinism; PyTorch normalization | Numeric norm-specific worst-case constants, known twiddle accuracy, exact actual bundled cuFFT version/plan kernel path |
| μD,μt,μw,μs,μP,μA | Source operation graph; NVIDIA scalar/FMA rounding descriptions | Verified effective einsum/GEMM precision, reduction ordering/operation count, backend flags, overflow/subnormal/flush handling for every relevant input |
| Arnoldi f0,F,E and candidate dx | Actual directions retained; FP64 outer action; one-pass MGS; final refresh | A universal orthogonality guarantee, finite coefficient norm bound, a complete trajectory from the stopped run, or free raw-image capture |
| Final-residual/norm arithmetic | Fresh original-H64/source check and original evaluated acceptance | Interval residual/source/field norm enclosures or an exact-nominal tolerance proof at a rounded boundary |

The standard model gamma_k(u) requires k*u<1 and ordinary correctly rounded arithmetic without unaccounted range effects. It is not licensed for an undocumented fused/reduced-precision contraction just from `complex64` storage. The conditional model permits explicit additive underflow terms if needed; removing them requires a justified range domain. Tensor-core use is not inferred from hardware alone and is not accused as the observed cause.

No empirical coefficient is fitted to low_H64's worst5.036133085e−6 receipt. The source-based homogeneity diagnostic remains the original engineering gate; operation-error/conditioning formulas describe why a different residual-related question requires different justified quantities. Further numerical evaluation or a newly designed gate would need separate authorization and a protocol fixed before its measurements.
