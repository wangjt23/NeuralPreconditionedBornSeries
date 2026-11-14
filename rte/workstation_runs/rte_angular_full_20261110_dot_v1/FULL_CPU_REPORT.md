# Seventeen mode angular block qualification on CPU

The stored angular operator admits an approximately cyclic block representation whose defect can be corrected against the original operator. On the frozen four-source development workload, this representation preserves the useful inner P8 solve while reducing its construction cost. Three fresh paired CPU confirmations beat the paid compact4 control by 20.76%, 21.34%, and 22.58%. This is a same-instance classical solver result, not a neural gain, material-generalization result, GPU result, or proof of a bitwise exact factorization.

## Fixed problem and acceptance

The original periodic Fourier RTE is unchanged: N=128, 12 stored angular directions, g=0.65, the existing two-layer material with inclusion amplitude 3, and source IDs [0, 4, 7, 14] in that order. All physical coefficients, angular weights, phase matrix and Fourier frequencies come from the original stored implementation. No idealized directions or isotropic replacement are used. Source hashes and production hashes match sealed CPU confirmation A.

The paid decision control is compact4 injection after one TSA64 call. Both P8 candidates keep exactly tangential modes -8 through 8 and solve the same projected original operator P8 A P8, from zero, with inner tolerance 1e-4 and at most 8 iterations. The outer solver retains tolerance 1e-5 and maximum 100 iterations. Every accepted result must also satisfy the evaluated original-equation relative error bound <=1e-3 and relative field difference <=1e-4 against the ordinary native accepted solution. The four native reference solutions are checks only, never candidate initial states or directions.

The structured block residual target is 1e-10 against original G, with at most 3 defect corrections. The raw angular band is only an approximate representation of the actual stored matrix. It retains the actual transformed diagonal and cyclic nearest-neighbor blocks, not analytically regenerated ideal coefficients. The cyclic closure uses a rank-256 corner update to twelve 128-by-128 block factors. It does not assume that the g=0.65 collision matrix is rank one.

## Predeclared protocol and resource use

`FULL_GATE_DEBATE.md` records the two decision rounds before timing. Screening used fresh construction in the fixed orders compact4, dense P8, structured P8 and then structured P8, dense P8, compact4. Only after the structured method beat both controls in total cost and beat dense setup in both orders did the predeclared three additional compact4/structured pairs run, in AB, BA, AB order. No parameter or threshold changed after seeing results.

The process PID was 1270333, pinned to CPU0 with one thread and CUDA disabled. Schur's numerical batch had exited before admission. GB training on CPU32/34 and the A6000 continued, so these are shared-host paired CPU observations, not strictly isolated CPU performance measurements. The job exited successfully and CPU0 was released. Total research runtime was 255.602 seconds, including 24.423 seconds for the four ordinary reference solutions. The sum of all twelve observed candidate batch wall times was 217.999 seconds. Remaining research overhead includes receipt writing, garbage collection, loading and bookkeeping.

## Complete costs and first source

All times below are seconds. Every candidate rebuilds the material/acceptance objects, TSA and its factors. Dense and structured P8 additionally pay for independent original-A0 qualification of all 17 modes. Online cost includes all inner applications, FFTs, original-G checks and defect repairs, outer work and final original-A acceptance checks. There is no device transfer in this CPU experiment. Preparation is not amortized over extra sources.

| Fresh batch | compact4 complete four RHS | dense P8 complete four RHS | structured P8 complete four RHS |
| --- | ---: | ---: | ---: |
| Screening forward | 20.736757 | 18.490616 | 16.266673 |
| Screening reverse | 20.603038 | 17.642856 | 15.782028 |
| Confirmation 1 | 20.312527 | not repeated | 16.095606 |
| Confirmation 2 | 20.312084 | not repeated | 15.977966 |
| Confirmation 3 | 20.069270 | not repeated | 15.537745 |

The structured improvement over dense is 12.03% and 10.55% in the two same-batch screening orders. Its confirmation median is 15.977966 seconds versus compact4's 20.312084 seconds, a 21.34% reduction. The observed whole-batch wall medians are 15.992239 and 20.326235 seconds, respectively; including the small orchestration gap does not reverse the result. There are only three confirmation pairs, so the individual values are the evidence, not a population-level confidence claim.

| Confirmation batch | compact4 first RHS with full setup | structured P8 first RHS with full setup |
| --- | ---: | ---: |
| 1 | 7.839461 | 5.374299 |
| 2 | 7.804880 | 5.429194 |
| 3 | 7.803499 | 5.143662 |

These first-source costs are measured source 0 solve/check time plus the entire fresh setup, not one quarter of the four-source total. Unlike the old dense implementation, the new representation already pays off on the first source of this frozen instance.

## Attribution and storage

The same-batch screening medians separate setup from online work:

| Method | physical and TSA setup | factor construction | original A0 qualification | all setup | online four RHS |
| --- | ---: | ---: | ---: | ---: | ---: |
| compact4 | 0.155243 | 3.053274 | not an exact block inverse | 3.208518 | 17.461379 |
| dense P8 | 0.139876 | 5.059477 | 0.448292 | 5.647645 | 12.419091 |
| structured P8 | 0.132876 | 1.071039 | 0.529303 | 1.733217 | 14.291133 |

Construction savings cover the slower structured application and correction path. The result is not an application-speed improvement over dense LU. The stored factor/coefficient counts are compact4 302,137,344 bytes, dense 641,832,960 bytes and structured 276,633,928 bytes. The structured logical count is 56.90% below dense and 8.44% below compact4. This is not a peak resident-memory measurement; shared coefficient views may be counted more than once. Total factor construction is recorded here. The earlier three-mode qualification D separately records block assembly, chain and corner factor costs; this full run did not separately persist those subphase timings for every one of the 17 modes.

All compact4 solves use outer TSA calls [4,3,4,3]. Every dense and structured P8 batch uses outer calls [2,2,2,2] and inner iterations [5,5,5,4]. Thus the changed timing comes from the inverse representation, not extra inner iterations, a different subspace, more RHS reuse or a relaxed acceptance threshold.

## Original operator validation

All 48 candidate solves pass. The largest original-A relative residual is 9.080373e-6, the largest evaluated relative error bound is 3.945352e-5 and the largest relative field difference is 2.648812e-7. No high-accuracy teacher was constructed.

There are 119 independent original-A0 checks: all 17 modes for each of the two dense and five structured batches. The maximum relative residual is 3.858052e-12. Each check embeds the candidate mode solution into the physical-space field, calls the original high-precision A and removes the actual inclusion perturbation. It does not merely re-evaluate the banded surrogate. All candidate batches use identical qualification vectors, verified by per-mode hashes.

During the actual structured solves, all 1,615 mode inversions require exactly one defect correction. Their maximum raw-band original-G residual is 1.807100e-5; the maximum repaired residual is 3.930610e-12. The correction target remains 1e-10 and the maximum permitted corrections remains 3. Agreement in calls, projected-inner convergence, independent original-A0 checks and final original-A acceptance supports preservation of the intended P8 operation to the stated numerical tolerance. It does not make the representation bitwise exact.

The separately completed zero-column qualification covers modes -8, 0 and 8, all-zero blocks and mixed zero/nonzero columns, including nonzero columns scaled by 1e-200. Strict zero columns return exactly zero with zero residual. Nonzero columns use scale-normalized relative residuals without an absolute floor. The largest mixed-case repaired residual is 2.079102e-12. This fixes the earlier small-probe zero-division boundary without changing its sealed source or results.

## Scope and next decision

This result covers one fixed development material, grid, angle set and four sources on CPU0. It does not establish performance on amplitude 6, another material, a different angular discretization, N=256, reserved samples, or a GPU. CPU seconds must not be compared to the earlier A6000 seconds to infer acceleration. The earlier dense GPU no-go remains valid for that implementation.

Before another numerical round, debate the following distinct questions. First, a single predeclared amplitude 3-to-6 CPU perturbation could test whether the new representation preserves the old bounded stress result without changing its sources or thresholds. This is a coupling-strength perturbation, not broad material generalization. Second, a GPU port might remove dense construction cost but still lose to compact4 because small sequential block solves and repair checks launch many kernels. Third, implementation-only profiling could identify that launch structure without making any new GPU performance claim.

No further experiment is started by this report. A GPU experiment requires explicit coordinated natural release after GB training, evaluation and timing. Its first gate must pay all setup and original-operator validation on the same machine; stop if structured setup already exceeds compact4's complete four-RHS cost, if a block fails the unchanged correction limit or original acceptance, or if complete cost has no advantage. Do not add RHS, retune inner8, loosen tolerances or use reserved data to manufacture a win.

## Evidence and reproduction

`full_01/` contains immutable run receipts: settings, rows, batches, ordinary reference checks, all independent mode qualifications and every runtime repair trace. `zero_columns_01/results.json` contains the boundary qualification. `run_full_cpu.py` and `angular_modes.py` are the executed sources. `analyze_full.py` validates source hashes, original thresholds, all cost identities, all modes, paired ordering and profile counts, and prints `analysis.json`. Re-run the analyzer for static verification; do not re-run the numerical entry point in the existing output directory.

Executed source SHA256 values:

- run_full_cpu.py: 2450fa79b22f0640fd36cd5cd7571b37a6c219ae9130eab341b901f67de2d5de
- angular_modes.py: d152afea90179efa8342a4f12631979951503785330f3aab08d4b7f1fe32a481
- Dense source inherited from A: b560e467e0b81a8293adfa2fc71bd5570ec875a797fea67696083f1430ba3a77

Sealed earlier A/B/C/D packages are unchanged. D's package SHA256 is c857ad48db8727e929fc7c0ed69bc7dadfffcdaa33c3b9e67cec477a2472c32a. The new package's manifest and ZIP receipt provide its own complete provenance. Runtime inputs still require the existing project and original source bank identified in settings; the small evidence package does not duplicate the full dataset.
