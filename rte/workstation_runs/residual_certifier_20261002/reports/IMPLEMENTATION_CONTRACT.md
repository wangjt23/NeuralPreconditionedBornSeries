# Input-specific interval FFT certifier — implementation contract

This helper supplies an input-specific enclosure of the exact nominal discrete map formed from the original captured assembled complex128 D64, stored real collision/angular buffers and frozen source/candidate bits. It does not supply a GPU error contract, continuum estimate, solver convergence result or performance advantage. No original source, project field, GPU calculation or full solve was used in its local development.

## API and execution

Compile `cpu_interval_primitives.cpp` with the existing system compiler using `-O2 -std=c++17 -fno-fast-math -ffp-contract=off -shared -fPIC`; use the platform's shared-library suffix. Record compiler version/command and source/library digests. Call `load_native_cpu_backend(path)` before `certify_residual(q,x,D64,t,s,phase,weights)`. Without that backend, certification fails closed. The backend is process-local; it changes no floating control mode, package, allocator or CUDA setting.

Shapes are q/x B×L×Nx×Ny, t/s B×1×Nx×Ny, phase L×L, weights L, D64 B×L×Nx×Ny or1×L×Nx×Ny. Spatial lengths must be powers of two≥2. D64 must be complex128 with exactly zero real part. Inputs must remain frozen for the call. Binary32 operands are promoted exactly where permitted; arbitrary integer/extended-precision input narrowing is rejected. Optional `margin_lower` must be positive and no greater than the independently derived conservative coefficient margin.

`fft_box`/`fft2_box` and interval primitives expose conditional fixture operations; they alone do not claim native mode checks. `certify_residual` requires the native guard before work and again after it, records both checks, and rejects arithmetic-control changes. Guard flags32/64 repeat mode checks. They are not independent empirical proofs. Sticky exception flags may change during ordinary arithmetic; control modes are neither reset nor changed. Actual compiler/CPU IEEE semantics remain the explicit trust boundary. NumPy is used for exact copying/indexing/comparisons and adjacent-value `nextafter`; certification's every endpoint add/multiply uses the authored native kernels. It uses no FFT/BLAS/trigonometric/reduction backend as a certificate.

## Enclosure argument

Each native real add/multiply is a separately rounded binary64 primitive; one adjacent value in each direction encloses its exact real value, including gradual underflow, under the declared trust contract. Four corner products followed by outward extrema enclose real interval multiplication. Four such interval products and two interval additions/subtractions enclose complex multiplication. Nonfinite/overflow endpoints fail closed. Permutation, reshape, bit reversal and sign change are exact.

The twiddle provider obtains a rational pi bracket from the alternating arctangent series and Machin identity. It reduces angles to magnitude≤pi/4, encloses sine/cosine at an exact rational midpoint with alternating-series tails, and adds the angular radius using derivative bounds≤1. Quadrant changes are exact; rational interval width is<1e−30 before directed binary64 conversion. Axis roots remain exact. Forward signs are negative and inverse signs positive.

Induction through bit reversal and each radix2 butterfly encloses the exact DFT at every stage. Inverse normalization multiplies by the exact power-of-two reciprocal, with outward multiplication. Applying both spatial axes gives backward-normalized2D transforms. Multiplying by the captured symbol and enclosing every exact stored collision product/angular sum therefore encloses the same nominal physical map. Angular contraction uses an explicit loop over the actual phase/weight order, with no ideal quadrature replacement.

Squared intervals crossing zero have lower bound zero and upper bound the larger squared endpoint; other squares use outward endpoint minima/maxima. Explicit outward pairwise addition encloses their sum. Scalar square roots use integer/Fraction dyadic brackets and directed conversion, requiring no floating sqrt contract. The coefficient margin is `min(t-s*sqrt(||P||1||P||inf))`, P=phase·diag(weights), with exact rational stored products, upward rational sqrt and positive-margin validation. Pure-imaginary streaming is skew-Hermitian in the spatial Euclidean DFT geometry. Cauchy–Schwarz gives this collision coercivity lower bound without symmetry/eigenvalue assumptions on P.

## Fixed acceptance and scalar receipts

The exact residual tolerance is1/100000 and the fixed reserve fraction1/100. A represented center is clipped inside every residual interval component. E is the norm upper bound of the vector residual box minus that center, not the difference of residual norm endpoints. Exact Fraction cross-products require `C_U <= (99/100)*tau*Q_L`, `E_U <= (1/100)*tau*Q_L`, and direct `R_U <= tau*Q_L`. C/Q norm uncertainty is already included through upper/lower bounds; no extra norm-width allowance is added. Center/source norm widths and vector-radius bounds are reported separately.

With positive exact margin m, solution error is at most R_U/m. The relative solution-error bound is `R_U/(m*X_L-R_U)` when that denominator is positive; acceptance requires it≤1/1000. Zero candidates or nonpositive solution denominators yield no solution bound. Zero source-norm lower bound returns failed relative acceptance and no relative residual value, rather than dividing by zero or accepting it. Used/derived margins are returned as exact rational strings and used numerator/denominator, permitting scalar acceptance reconstruction.

Timers separately record guards, coefficient margin, source/candidate norms, interval action, residual norm, center/radius reserve and exact acceptance, plus whole call time. Twiddle setup incurred on a first action is included in that action; compiling/loading or any explicit prewarming must additionally be charged by the executor. Original fields remain private; a consuming runner may export hashes and scalar receipts.

## Local validation and scope

Ten native interval tests passed with the existing Apple clang21.0.0, ARM64 FPCR guard127 and Python3.9.6/NumPy2.0.2. Twelve independent twiddle tests passed. Fixtures compare radix2 results with a dense exact-Fraction4×4 DFT and a heterogeneous/nonsymmetric weighted angular operator; they exercise subnormal and exact-corner arithmetic, normalization, shape/finite/overflow failures, axis/output ownership, rational solution-bound reconstruction, fixed reserve and zero source/candidate outcomes. Root8/16 round trips test larger radix stages without project fields.

These finite fixtures do not prove an arbitrary compiler or all hardware operations. Deployment requires the actual FWI native build/guard and staged CPU tests, with compiler provenance and bounded entry review. No local project-field experiment, remote call, GPU run or full solve was performed by this implementation agent.
