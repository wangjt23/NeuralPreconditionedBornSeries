# CBS and Shifted-Laplacian Preconditioning for 2-D Helmholtz

This report documents the implemented preconditioners, the benchmark setup, the numerical results, and the conclusions from the comparison work. It is intentionally self-contained: the key algorithms, "inner solve" terminology, discretizations, run commands, figures, and interpretation are all included here.

The most important point is that CBS and finite-difference shifted Laplacian are naturally matched to different discrete operators:

- CBS is native to the repository's original pseudo-spectral/FFT Helmholtz discretization.
- Finite-difference shifted Laplacian is native to a five-point finite-difference Helmholtz discretization.

For that reason, the final comparisons are separated by discretization. Cross-discretization runs were useful for debugging and tuning, but they are not used as the main leaderboard.

## Executive Summary

For the pseudo-spectral/FFT Helmholtz operator, CBS is the correct native baseline. On the main `L=2`, `n=160`, `ppw=12` case, FGMRES with a tuned CBS preconditioner converges in `144` outer iterations, and the nested CBS-inner variant converges in `38` outer iterations. The FFT shifted-Laplacian V-cycle also converges, but is much slower on CPU in this implementation.

For the finite-difference Helmholtz operator, the FD shifted-Laplacian family is the native comparison. On the same physical case, FD-SL V-cycle and matrix-free FD-SL converge, but the sparse shifted-Laplacian LU preconditioner is much faster. A direct sparse LU factorization of the unshifted FD Helmholtz matrix is faster still on this 2-D problem size; it is a direct solver reference, not a preconditioner design.

The main practical conclusion is:

- Use CBS results only on the pseudo-spectral operator.
- Use FD-SL results only on the finite-difference operator.
- The current 2-D CPU problems are still small enough that sparse direct LU remains very competitive. Iterative preconditioners should show clearer benefits for larger 2-D grids, 3-D problems, GPU matrix-free runs, or many-RHS/parameter-sweep settings where direct factorization cost or memory dominates.

## Files Added or Modified

Relevant implementation files:

- `helmholtz/models/cbs.py`: original CBS pseudo-spectral operator and stationary CBS update. This file remains the source of the original CBS discretization.
- `helmholtz/models/shifted_laplacian.py`: shifted-Laplacian implementations:
  - `ShiftedLaplacianVcycle2d`
  - `FiniteDifferenceShiftedLaplacianVcycle2d`
  - `MatrixFreeShiftedLaplacian2d`
  - `SparseShiftedLaplacian2d`
- `helmholtz/models/gmres.py`: right-preconditioned flexible GMRES.
- `helmholtz/compare_cbs_shifted_laplacian.py`: native pseudo-spectral CBS comparison.
- `helmholtz/compare_fd_shifted_laplacian.py`: native finite-difference shifted-Laplacian comparison.
- `helmholtz/inference2d.py`: integration hooks for shifted-Laplacian preconditioner model names.

## Common Physical Problem

The benchmark uses a deterministic moderate-contrast velocity field:

```text
c(x,y) = 1
       + 0.08 sin(2 pi x) cos(2 pi y)
       - 0.08 exp(-((x - 0.35)^2 + (y - 0.55)^2) / 0.025)
       + 0.07 exp(-((x - 0.72)^2 + (y - 0.38)^2) / 0.018)
```

The squared slowness field is

```text
kappa(x,y) = 1 / c(x,y)^2.
```

The main reported case is:

```text
physical domain     L x L, L = 2
physical grid       n x n, n = 160
grid spacing        dx = L / n = 0.0125
boundary layer      PBL, width = 16 grid cells
padded grid         192 x 192
source              unit point source at (n/4, n/2)
ppw                 12
frequency           mean(c) / (ppw * dx) = 6.6513574365840675
velocity range      [0.8528643755555572, 1.1332342936382955]
kappa range         [0.7786828018689907, 1.37480167214025]
kappa contrast      1.7655477542851308
precision           float64 / complex128
hardware            CPU
outer solver        FGMRES
outer tolerance     1e-6 relative residual
outer restart       100
outer maxiter       500
```

The `L=2`, `n=160` case has four times the physical area of the earlier `L=1`, `n=80` case while preserving `dx`, frequency, and points per wavelength.

## Solver Terminology

The outer linear system is

```text
A u = b.
```

All Krylov experiments use right-preconditioned FGMRES. The preconditioner application is written as

```text
z = P(r),
```

where `P` approximates `A^{-1}` or another useful inverse-like operator. In right preconditioning, FGMRES builds the solution from preconditioned basis vectors. This is why flexible GMRES is used: some preconditioners, especially nested Krylov preconditioners, are not fixed linear maps in the strict classical sense.

### What "Inner Solve" Means

An "inner solve" is a small Krylov solve performed inside one outer preconditioner application.

For example, in `FGMRES + CBS inner(4)`, the outer solver still solves

```text
A u = b.
```

At every outer preconditioner call, instead of applying the CBS kernel once, the code approximately solves

```text
A z = r
```

with exactly `4` inner FGMRES iterations. That inner FGMRES itself uses the CBS kernel as its preconditioner. The outer solver receives `z` as the result of the preconditioner application. This is not a direct solve; it is a nested fixed-budget iterative preconditioner.

For `FD-SL matrix-free`, the inner solve is different. The outer solver still solves the physical FD Helmholtz system

```text
A_FD u = b,
```

but the preconditioner approximately solves the shifted system

```text
M_beta z = r
```

with a fixed number of inner FGMRES iterations. Here `M_beta` is the shifted-Laplacian operator, not the original Helmholtz operator.

## Discretization A: Pseudo-Spectral/FFT Helmholtz

This is the repository's original CBS discretization in `ConvergentBornSeries2d`.

The padded-domain pseudo-spectral operator is

```text
A_PS u = dx^2 [ F^{-1}(|p|^2 F u) - k(x)^2 u ],
```

where:

- `F` is the FFT over the padded grid.
- `|p|^2` is the spectral symbol of the negative Laplacian.
- `k(x)^2 = omega^2 kappa(x)` includes the complex PBL modification.
- The FFT is periodic on the padded domain, while outgoing behavior is approximated by the complex absorbing boundary.

### CBS Preconditioner

CBS uses a homogeneous reference wavenumber `k0` and damping `epsilon`:

```text
G0        = (|p|^2 - k0^2 - i epsilon)^(-1)
V         = k(x)^2 - k0^2 - i epsilon
gamma     = i V / epsilon
P_CBS(r)  = gamma G0 (r / dx^2).
```

The stationary CBS update is

```text
u <- u - gamma [ u - G0 (V u + b / dx^2) ].
```

The FGMRES version applies `P_CBS(r)` as a right preconditioner.

The tuned CBS variant allows simple scalar tuning:

```text
k0       <- k0_scale * k0
epsilon  <- epsilon_scale * epsilon
gamma    <- gamma_scale * gamma
```

On the main pseudo-spectral run, `epsilon_scale=1.25` was the best of the small tested set.

### CBS Inner Krylov

`FGMRES + CBS inner(4)` uses this preconditioner application:

```text
Given outer residual r:
  approximately solve A_PS z = r
  with 4 inner FGMRES iterations,
  using P_CBS as the inner right preconditioner.
Return z to the outer FGMRES.
```

This is a fair CBS-side optimization because it uses the same outer Krylov framework that was also considered for shifted Laplacian.

### FFT Shifted-Laplacian V-Cycle

For comparison on the pseudo-spectral operator, the shifted-Laplacian V-cycle uses

```text
M_beta,PS u = dx^2 [ F^{-1}(|p|^2 F u) - (1 + i beta) k(x)^2 u ].
```

The hierarchy uses full-weighting restriction, bilinear prolongation, and damped Jacobi smoothing. The main pseudo-spectral run used:

```text
beta          = 1.5
levels        = [(192,192), (96,96), (48,48), (24,24), (12,12)]
pre_smooth    = 6
post_smooth   = 6
num_cycles    = 3
```

## Pseudo-Spectral Numerical Results

Command:

```bash
cd helmholtz
/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python compare_cbs_shifted_laplacian.py \
  --n 160 \
  --domain-length 2 \
  --boundary-width 16 \
  --skip-cbs-stationary \
  --include-cbs-inner \
  --cbs-inner-maxiter 4 \
  --cbs-epsilon-scale 1.25 \
  --shift 1.5 \
  --sl-discretization fft \
  --pre-smooth 6 \
  --post-smooth 6 \
  --num-cycles 3 \
  --krylov-maxiter 500 \
  --out-dir ./outputs/ps_cbs_sl_n160_L2
```

Figure:

![Pseudo-spectral CBS and FFT SL comparison](outputs/ps_cbs_sl_n160_L2/cbs_vs_sl_vcycle.png)

Results:

| Method | Discretization | Preconditioner | Outer iterations | Final relative residual | Wall time |
|---|---|---|---:|---:|---:|
| FGMRES + CBS | pseudo-spectral | tuned CBS kernel, `epsilon_scale=1.25` | 144 | `9.544e-07` | `2.479s` |
| FGMRES + CBS inner(4) | pseudo-spectral | 4 inner FGMRES iterations with CBS kernel | 38 | `8.935e-07` | `0.859s` |
| FGMRES + SL V-cycle | pseudo-spectral | FFT shifted-Laplacian V-cycle | 334 | `9.600e-07` | `22.519s` |

Summary file:

```text
helmholtz/outputs/ps_cbs_sl_n160_L2/summary.json
```

Interpretation:

CBS is best matched to this operator. The CBS kernel uses the same FFT Green's-function structure as the pseudo-spectral operator, so it is inexpensive and effective. The nested CBS-inner preconditioner is the strongest pseudo-spectral method tested: it reduces outer iterations from `144` to `38` and wall time from `2.479s` to `0.859s`.

The FFT shifted-Laplacian V-cycle converges but is much slower in this implementation. Its cost comes from repeated multigrid smoothing and coarse-grid work, while CBS uses direct FFT diagonalization of the homogeneous Green operator.

## Discretization B: Five-Point Finite-Difference Helmholtz

The finite-difference physical operator is

```text
A_FD u = (-Delta_h u) - dx^2 k(x)^2 u.
```

The zero-exterior five-point negative Laplacian is

```text
(-Delta_h u)_{i,j} = 4 u_{i,j}
                   - u_{i-1,j} - u_{i+1,j}
                   - u_{i,j-1} - u_{i,j+1}.
```

The same padded PBL-modified `k(x)^2` is used. The difference from the pseudo-spectral operator is the Laplacian discretization and boundary action at the padded edge.

### Shifted-Laplacian Operator

The shifted-Laplacian preconditioner is based on

```text
M_beta z = r,
M_beta = (-Delta_h) - dx^2 (1 + i beta) k(x)^2.
```

The complex shift `i beta k(x)^2` damps the indefinite Helmholtz operator, making approximate inversion easier. Smaller shifts can better approximate the true operator but may be harder for multigrid; larger shifts are more stable for smoothing but less accurate as a preconditioner.

### FD-SL V-Cycle

The FD-SL V-cycle uses:

- rediscretized FD shifted operators on coarser grids,
- full-weighting restriction,
- bilinear prolongation,
- Jacobi or Vanka smoothing,
- optional exact coarse solve on the coarsest level.

The final FD run used Vanka smoothing:

```text
beta          = 0.75
levels        = [(192,192), (96,96), (48,48), (24,24), (12,12)]
pre_smooth    = 2
post_smooth   = 2
num_cycles    = 2
Vanka patch   = 2 x 2
Vanka stride  = 2
Vanka weight  = 0.8
```

The Vanka smoother forms overlapping local `2 x 2` patch systems. For each patch, it solves the local shifted-Laplacian block for a residual correction, scatters corrections back, averages overlaps, and damps by `vanka_weight`. The implementation caches local patch inverses, so repeated smoothing sweeps do not repeatedly call `torch.linalg.solve` for the same local matrices.

### FD-SL Matrix-Free Inner Solve

The matrix-free shifted-Laplacian preconditioner performs a fixed-budget inner solve:

```text
Given outer residual r:
  approximately solve M_beta z = r
  with inner FGMRES, using only PyTorch stencil applications.
Return z to outer FGMRES.
```

For the reported run:

```text
beta                       = 0.25
inner FGMRES max iterations = 12
inner preconditioner        = Jacobi
```

No sparse matrix is assembled for this method. It applies the shifted stencil directly in PyTorch.

### FD-SL Sparse LU

The sparse shifted-Laplacian preconditioner explicitly assembles the shifted FD matrix

```text
M_beta = (-Delta_h) - dx^2 (1 + i beta) k(x)^2
```

and factors it once with SciPy sparse LU. Each preconditioner application solves

```text
M_beta z = r
```

using the stored LU factors. This is still a shifted-Laplacian preconditioner, not a direct solve of the original Helmholtz matrix, because `beta=0.25`.

### Exact FD Helmholtz LU Reference

The exact FD Helmholtz LU reference assembles and factors the unshifted physical FD operator

```text
A_FD = (-Delta_h) - dx^2 k(x)^2.
```

It applies `A_FD^{-1}` directly. Therefore FGMRES converges in one iteration up to numerical precision. This is a direct solver baseline, not an iterative preconditioner design and not a shifted-Laplacian method.

It is included to show whether the problem size is large enough for iterative methods to beat direct sparse factorization. For the current 2-D CPU sizes, direct LU remains very strong.

## Finite-Difference Numerical Results

Command:

```bash
cd helmholtz
/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python compare_fd_shifted_laplacian.py \
  --n 160 \
  --domain-length 2 \
  --boundary-width 16 \
  --skip-cbs \
  --include-identity \
  --smoother vanka \
  --vanka-stride 2 \
  --vanka-weight 0.8 \
  --pre-smooth 2 \
  --post-smooth 2 \
  --num-cycles 2 \
  --vcycle-shift 0.75 \
  --mf-shift 0.25 \
  --mf-inner-maxiter 12 \
  --mf-inner-preconditioner jacobi \
  --sparse-shift 0.25 \
  --include-fd-lu-reference \
  --krylov-maxiter 500 \
  --out-dir ./outputs/fd_sl_only_n160_L2
```

Figure:

![Finite-difference SL comparison](outputs/fd_sl_only_n160_L2/fd_cbs_vs_sl.png)

Results:

| Method | Discretization | Preconditioner | Outer iterations | Final relative residual | Setup | Solve | Total |
|---|---|---|---:|---:|---:|---:|---:|
| FGMRES + identity | FD | none | 500 | `9.616e-03` | `0.000s` | `9.505s` | `9.505s` |
| FGMRES + FD-SL V-cycle | FD | V-cycle, Vanka, `beta=0.75` | 250 | `9.811e-07` | `0.004s` | `8.566s` | `8.570s` |
| FGMRES + FD-SL MF | FD | matrix-free inner FGMRES, `beta=0.25` | 230 | `9.801e-07` | `0.003s` | `10.918s` | `10.921s` |
| FGMRES + FD-SL sparse LU | FD | sparse LU of shifted operator, `beta=0.25` | 54 | `9.041e-07` | `0.412s` | `0.933s` | `1.345s` |
| FGMRES + FD Helmholtz LU | FD | direct sparse LU of `A_FD` | 1 | `1.746e-14` | `0.218s` | `0.019s` | `0.237s` |

Summary file:

```text
helmholtz/outputs/fd_sl_only_n160_L2/summary.json
```

Interpretation:

FD-SL sparse LU is the strongest shifted-Laplacian variant tested on the FD operator. The pure PyTorch V-cycle and matrix-free variants converge, but their CPU stencil/smoothing cost is high. The exact unshifted FD LU direct solve is fastest because this remains a moderate 2-D problem size.

## Scaling Diagnostic

A larger diagnostic run was performed with:

```text
L = 4
n = 320
dx = 0.0125
padded grid = 352 x 352
```

This is four times the linear domain size of the original `L=1`, `n=80` run, and sixteen times the physical area. It was run only for key methods.

Figure:

![Finite-difference scaling diagnostic](outputs/fd_scaling_n320_L4_key_methods/fd_cbs_vs_sl.png)

Results:

| Method | Iterations | Final relative residual | Total time |
|---|---:|---:|---:|
| FGMRES + CBS tuned, applied to FD diagnostic | 264 | `9.696e-07` | `11.934s` |
| FGMRES + CBS inner(4), applied to FD diagnostic | 71 | `9.537e-07` | `4.696s` |
| FGMRES + FD-SL sparse LU | 94 | `9.788e-07` | `7.940s` |
| FGMRES + FD Helmholtz LU | 1 | `2.790e-14` | `1.399s` |

Summary file:

```text
helmholtz/outputs/fd_scaling_n320_L4_key_methods/summary.json
```

This diagnostic still shows direct FD LU ahead for single-RHS 2-D CPU solves. It also confirms that the direct solver comparison should not be overinterpreted: sparse direct factorization remains affordable at these sizes.

## Analysis During Development

The comparison evolved in several stages.

1. Initial shifted-Laplacian V-cycle on the pseudo-spectral operator underperformed CBS. The main reason is discretization mismatch and method fit: CBS is an FFT Green's function method for the same pseudo-spectral structure as the original operator, while a standard geometric SL V-cycle does not exploit that operator as directly.

2. A finite-difference Helmholtz operator was added to test shifted Laplacian in a compatible setting. Once the physical operator and shifted preconditioner both used the five-point stencil, FD-SL convergence improved. This confirmed that part of the early underperformance was due to comparing SL against a non-native discretization.

3. CBS also needed fair tuning. Scaling `epsilon` improved CBS, and adding a CBS inner-Krylov preconditioner improved it substantially. This was important because applying advanced Krylov ideas to SL but not to CBS would bias the comparison.

4. Vanka smoothing was added for SL and implemented in a cached PyTorch form. It helped the V-cycle relative to the earlier Jacobi-only smoother, but it did not make the pure PyTorch V-cycle competitive with CBS on the pseudo-spectral operator or with sparse factorization on the FD operator.

5. Sparse shifted-Laplacian LU was added as an advanced SL reference. It is much stronger than pure V-cycle or matrix-free SL on the FD operator, but still slower than direct unshifted FD LU for the tested 2-D sizes.

6. Domain enlargement was clarified. Increasing `n` alone while setting `dx=1/n` refines a fixed unit domain. The scripts now support `--domain-length` so that `n=160, L=2` preserves `dx` and PPW while increasing physical area by four.

7. Final reporting separates pseudo-spectral and finite-difference results. This avoids the misleading statement "CBS versus FD-SL" as a single fair table when the two methods are being evaluated against different residual operators.

## Final Conclusions

1. CBS should be judged on the repository's original pseudo-spectral/FFT operator. On that operator, tuned CBS and especially CBS-inner FGMRES are the strongest methods tested.

2. Shifted Laplacian should be judged on a compatible finite-difference operator unless an FFT/pseudo-spectral shifted-Laplacian method is specifically designed and optimized for the pseudo-spectral discretization.

3. The "inner solve" variants are nested fixed-budget Krylov preconditioners. For CBS inner, the inner system is the original pseudo-spectral Helmholtz operator and the inner preconditioner is the CBS kernel. For FD-SL matrix-free, the inner system is the shifted FD Helmholtz operator.

4. The direct FD Helmholtz LU reference is a true direct sparse solver for `A_FD`, not a preconditioner. It remains fastest in the tested 2-D CPU regime, meaning the test size is still not large enough to demonstrate a direct-solver memory or time bottleneck.

5. For future work, the most meaningful next comparisons would be:
   - larger 2-D grids until direct LU memory becomes limiting,
   - 3-D finite-difference Helmholtz where direct LU becomes much less attractive,
   - GPU matrix-free SL and CBS experiments,
   - multiple right-hand sides where setup amortization changes the sparse/direct tradeoff,
   - stronger SL smoothers/coarsening strategies from Helmholtz multigrid literature.

## Verification

Python files were checked with:

```bash
/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python -m py_compile \
  helmholtz/models/shifted_laplacian.py \
  helmholtz/compare_fd_shifted_laplacian.py \
  helmholtz/compare_cbs_shifted_laplacian.py \
  helmholtz/verify_shifted_laplacian.py \
  helmholtz/inference2d.py
```

The checked files compiled successfully.

