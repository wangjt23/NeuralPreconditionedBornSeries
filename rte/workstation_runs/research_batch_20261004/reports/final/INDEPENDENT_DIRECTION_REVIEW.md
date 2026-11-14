# Independent bounded-batch direction and driver review

Reviewed 2026-10-04 UTC, before FWI execution. This review authorizes the proposed **action-only diagnostics**, conditional on the existing FWI CPU preflight passing all seven Mloc tests. It does not admit a new full solver, a learned method, a changed tolerance, or a deployment claim. No SSH, GPU computation, project-source export, training, new source generation, or full solve was performed by this reviewer.

## Two distinct discriminators

**C: local collision scaling after one homogeneous reference.** With the full stored column-weighted phase matrix (P), define (C(x)=t(x)I-s(x)P), (C_0=\bar t I-\bar s P), and the homogeneous reference (R_0=(S+C_0)^{-1}). The prescribed nominal map is

\[
M_{\mathrm{loc}}=C(x)^{-1}C_0R_0,\qquad
I-HM_{\mathrm{loc}}=-S(C(x)^{-1}C_0-I)R_0.
\]

This fixes the local collision defect and leaves a precisely identifiable streaming penalty. It differs from the existing shifted HSS ordering in `remote_reference/rte/collision_transport.py`, and from gain-two or angular-restricted response controls. The new kernel retains the complete stored matrix, general complex eigenvalues, and an explicitly bounded eigenbasis condition; it does not infer a circulant spectrum from a phase row, discard eigenvalue imaginary parts, tune a shift, or repair a denominator. Its reference uses the actual captured D64 symbol and fixed spatial mean material. These are nominal algebraic properties; eigendecomposition, cache inversion and FFT applications remain floating computations.

The driver independently measures the useful local identity on its actual outputs (v=R_0r), (z=M_{\mathrm{loc}}r):

\[
r-Hz=(r-Sv-C_0v)-S(z-v)-(Cz-C_0v).
\]

The three terms distinguish reference closure, streaming penalty and local collision closure. The code calls the ledger a commutator diagnostic, but this displayed expression is the measured identity; no unmeasured commutator approximation is required. A large streaming term would explain why an accurate local inverse fails as a replacement without proving every material-adapted map ineffective.

**D: one full-angle coarse response column.** Compare the saved N64 teacher and saved N64 homogeneous reference for the same stored source/material parameters, using identical removal of the coarse negative-Nyquist row and column and identical normalized Fourier prolongation. Score each column on the original source and on the defect after one strong preconditioner action. This tests the information in a source-aligned spatial coarse response, rather than another angular Q3 space or a population POD basis. The parent-parameter identities bind the same development case; they do not assert that fine and coarse discrete operators or geometries are identical. Interpolation is checked against the filtered coarse teacher, not against the unfiltered teacher or an unknown continuous solution.

The teacher column is an information oracle, not a reusable learned residual action. Its application is constant as a function of the tested residual, and its real best-line coefficient uses the current physical image. Historical teacher acquisition cost is explicitly unknown and cannot be counted as zero. Positive headroom would justify investigating an affordable means of obtaining such a column; it would not establish affordable learning or generalization.

## Static driver and API checks

Reviewed `code/gpu_direction_jobs.py` SHA256 `9bc7c683f1ef6377584195c16f9bbf4cd79a418b1c1fd1ab049650cabc122b4e`, `code/batch_gpu_common.py` SHA256 `d5c77eb7273cfc26bb51619ef49a7760cc69c558e80f895b062ce1aa61ef8ab9`, and the draft policy SHA256 `8078855113bf8bed27f6807acc5b5b6a10e30aed38bcd891ca8cc95d0fd51a67`. The eventual sealed policy must pin the actually staged copies. The driver currently imports `mloc.LocalCollisionReplacement`, matching the authored convenience API.

`load_case` reconstructs the original low operator by casting saved promoted material/source back to FP32, then independently requires bit identities for **all** captured high source, streaming symbol, material, phase and weights. Those exact comparisons guard against coefficient changes; a failed round trip cannot be excused by a tolerance. The strong comparator is the original TSA8 at N128 or TSA16 at N256 through the existing direct segmented recipe. The second input is the defect after **one** strong action, not a four-step Arnoldi prefix.

Every candidate and actual constructed field is persisted and read back before original-H evaluation. The source field is `alpha*x`; the defect correction is `b+alpha*x`. Each is evaluated with the original full physical H, and its image is compared to the corresponding image reconstruction with a fixed (10^{-10}) relative discrepancy guard. This is an ordinary FP64 consistency check, not a certificate of the nominal map. Persisting inputs does not, by itself, persist all H output bits or prove future FFT replay bit identity.

For four unique N128/N256, ID0/3 cases, expected completion totals are:

| Job | Scored rows | Original H | Strong M | Extra streaming S | Persisted fields |
|---|---:|---:|---:|---:|---:|
| C | 24 | 52 | 28 | 16 | 60 |
| D | 16 | 36 | 4 | 0 | 36 |

The current H cap60 includes both; all full-solve/training/new-seed counts remain zero. Approximate field payloads are 450 MiB for C and 270 MiB for D, excluding small serialization headers, within the 512 MiB per-job snapshot cap. GPU memory/time caps are runtime failure guards, not estimates guaranteeing completion. The existing fail-closed protocol retains unfinished or negative jobs rather than extending budgets or changing the maps.

## Economics and inference boundary

Three synchronized application timings per map are component medians. Whole setup, input readback, transfers, persistence, physical images, line checks and process startup are separately paid by the job. A component advantage cannot establish an end-to-end solve advantage. The real best-line score establishes an attained improvement in one trial direction, not a best possible subspace, convergence rate or iteration bound.

The retained learned target is 10% against the strongest eligible contemporary traditional portfolio, with all setup, acquisition, verification and solve costs charged. In symbols, a future complete-cost candidate must satisfy (S_c+A_c+V_c+T_c\leq0.9(S_b+A_b+V_b+T_b)) under the specified use count; no source reuse is assumed. The earlier direct-classical means (176.926 ms N128 and 130.258 ms N256 complete algorithm cost) and warm means (49.896 and 102.716 ms) are historical context, not a current hardware floor or a clock-matched denominator for these action jobs. Current checker cost must be measured; adding a common checker to both arms increases the absolute savings required for the same percentage gate.

C's residual/cost screen and D's information screen remain engineering discriminators. Neither supersedes the 10% learned gate or the unchanged physical residual, solution-error and paired-field acceptance. Jobs A/B validate and optimize the checker on newly persisted bounded fixtures; paired manufactured fixtures provide engineering validation, not actual solver comparison evidence. No full comparison is admitted in this batch, including if all checker predicates pass.

## Authored kernel preflight

`mloc.py`: SHA256 `c73d22314665204b5a6657aa00bbafeb19850a54296ee36a01a4e94cdc920bd0`. `test_mloc.py`: SHA256 `9c072beb9e98d7ec1d0c8d9929d727c948c794b4535daec25ca75c214c2182b2`.

Five independent NumPy CPU tests passed: full noncirculant local dense solves, retained complex eigenvalues, complex-linearity/output ownership, a heterogeneous dense streaming defect identity, and rejection of defective/invalid inputs. Two Torch tests were explicitly skipped because Torch is absent on this auxiliary host; both must pass in the existing FWI CPU interpreter before C/D. The convenience-test phase contraction now explicitly promotes the stored phase to the complex field dtype, matching Torch's required einsum operand types. All four reviewed Python files compiled locally. No original project file was edited.
