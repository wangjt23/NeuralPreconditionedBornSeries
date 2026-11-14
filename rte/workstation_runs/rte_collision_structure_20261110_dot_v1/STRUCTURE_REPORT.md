# RTE: actual collision obstruction and angular-block qualification

## Current decision

The proposed rank-Ny isotropic Woodbury identity is not valid for the actual g=.65 kernel. An alternative angular-coordinate representation has passed the bounded three-mode original-G qualification. It is not bitwise exact and is not yet a faster complete solver. Do not change the original equation, g, RHS workload or tolerances.

## 1. Actual source, not the isotropic docstring

The executed _make_phase_matrix constructs a normalized angular Poisson kernel when g=.65. scattering_op uses the complete phase-times-weights matrix. Transport/total is angle-block-diagonal; sigma_s is a positive spatial diagonal; the boundary is periodic Fourier with no special inflow rows.

The stored12x12 angular matrix K has singular values from 0.149984633 to 1.000000039. Exact rational Gaussian elimination on the stored binary coefficients proves rank12, not merely a numerical-rank threshold. All128 scattering-profile values are positive (minimum6.500), so rank(K tensor S)=1536.

For the stated split G=T-U V^T, if K=L R^T has rank r, one may set U=L tensor S and V^T=R^T tensor I. The exact Woodbury formula is T^{-1}+T^{-1}U(I-V^T T^{-1}U)^{-1}V^T T^{-1}. Here r=12, so its reduced system still has1536 unknowns. It does not provide the desired128-dimensional correction.

Even moving arbitrary angular-block-diagonal terms into T cannot restore rank128: the disjoint angular rows0-5/columns6-11 submatrix is exactly invertible. Its corresponding off-diagonal spatial block has rank768 and is unaffected by any angular-block-diagonal reassignment. The exact determinant numerators/denominators and stored angular matrix are preserved in exact_rank_01/results.json.

## 2. The isotropic counterfactual is not the current operator

The actual stored g=0 kernel does have rank1. Its rank-Ny Woodbury action matches the corresponding dense isotropic action to4.6e-13 on the9 fixed-mode vectors and uses6.56MB rather than37.75MB of stored factors. But applying that same inverse to the actual g=.65 equation gives relative residuals0.996-2.053. This is an operator change, not a harmless exact representation. These counterfactual checks are diagnostic only and were never inserted into the original PDE solver.

## 3. Select one coordinate system using actual defects

Angular weights are bitwise constant and nodes nominally equally spaced, but stored float32 geometry is not ideal. Maximum angle error is2.88e-7rad, phase circulant defect3.23e-7 relative. The angular FFT collision off-diagonal norm is2.00e-7 relative, and non-nearest transport couplings are about1e-7. Hence cyclic block-tridiagonal projection is approximate.

Normal coefficients are not narrow-band: all128 Fourier coefficients of sigma_t/sigma_s are nonzero. Half-bandwidth16 loses5.73% coefficient L2, and32 still loses1.70%. No normal narrow-band solver was implemented.

For the angular projection, full G-block relative Frobenius defects are about1.26e-7 at k=0,1,8. A Kronecker norm bound puts discarded operator norm at2.07e-4 to2.16e-4 versus original coercivity0.04996. The resulting conditional real-arithmetic residual-correction contraction estimate is0.00415-0.00435. These are computed matrix bounds, not a bitwise identity or a replacement for original-G checks.

## 4. Selected implementation and governing identity

Use the actual transformed angular coefficient matrices, retaining diagonal and cyclic nearest-neighbor blocks including their small non-ideal entries. Factor the open12-block chain using128x128 block LU. Restore its two corner couplings with a256-dimensional Woodbury system. This Woodbury operation repairs periodic angular corners, not fictitious rank1 scattering.

Let B be the cyclic-band operator and E=G-B. Apply x0=B^{-1}q, then x<-x+B^{-1}(q-Gx), with at most3 corrections and original-G relative residual target1e-10. The original G action uses the stored normal FFT frequencies, actual directions, full12-angle phase/weights and material coefficients. Its agreement with the dense original block was checked before using it for repair.

## 5. Fixed qualification results

Modes[0,1,8], seed20261110, the same3 complex RHS columns per mode as the rank audit. Original matrix-free action agrees with dense G within 4.52e-16. Raw cyclic-band inverse has worst original residual 1.13e-05, so it must not be called exact. Every vector required only1 correction and then had original-G residual <= 2.75e-12; maximum relative difference from dense solution was 4.2e-12. No target relaxation or failed-vector deletion.

| k | Dense assembly+LU | Band full setup | Dense apply+check,3 RHS median | Band apply+repair+check median |
|---:|---:|---:|---:|---:|
| 0 | 0.349655s | 0.092127s | 0.011227s | 0.010960s |
| 1 | 0.336253s | 0.081323s | 0.010929s | 0.011265s |
| 8 | 0.327781s | 0.059839s | 0.010484s | 0.010664s |

Dense factor storage is37,754,880bytes/mode. Band factors plus logical coefficient data use16,272,584bytes/mode. Shared physical inputs and temporary peak allocation are not claimed as included in this factor-storage comparison. Setup is materially smaller, but application with repair is roughly comparable on this CPU qualification; do not claim application acceleration. Timings are small-block measurements, not end-to-end solver performance.

Recorded main-script wall time for rank, coordinate, exact-rank and band qualifications totals5.878s. CPU0/one thread only, GPU disabled, no full-PDE solve or training. A/B/C packages remain unchanged.

## Next bounded gate, not run

Build all17 actual tangent modes, including the previously untested negative modes, without assuming conjugate reuse under the stored Nyquist/geometry convention. First retain the same original-G repair target and max3 cap. Only if every block qualifies, compare one fully paid CPU0 four-RHS batch against compact4 and the prior dense-G candidate under the unchanged outer/prefix/P8/inner8 contract. This tests whether setup savings survive complete application/repair costs. Stop on any unrepaired block or original-equation failure; do not loosen thresholds or add RHS.

No GPU test is implied by this CPU result. GB owns the GPU; any future GPU action waits for a new coordinated natural window. The prior dense-G GPU NO-GO remains intact, and a new representation would need its own synchronized setup/application gate.

## Evidence and reproducibility

- DEBATE_AND_PROTOCOL.md and ANGULAR_BAND_DEBATE.md: competing representations, explicit assumptions and stop rules.
- audit_01/results.json and exact_rank_01/results.json: actual kernel rank, exact rational determinants and counterfactual diagnostics.
- coordinates_01/results.json: actual angular defects and normal Fourier tails.
- angular_band_01/results.json: every mode/vector residual, dense comparison, correction trace, setup/application/storage cost and RHS hashes.
- analyze_structure.py checks source hashes and regenerates this report and analysis.json.
- Source functions were imported from the existing FWI repository; production source hashes are in the receipts. No production default changed.
