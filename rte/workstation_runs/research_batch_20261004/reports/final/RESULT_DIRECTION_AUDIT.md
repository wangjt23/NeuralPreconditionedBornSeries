# Independent result audit: bounded directions C and D

Completed 2026-10-04 UTC from the downloaded scalar receipts, frozen authored code and policy. **Both fixed action-only screens fail.** The results do not justify a new full solve, training, a parameter sweep or a learned-advantage claim. No further experiment was run by this reviewer. Jobs A/B are outside this direction audit; their checker results must be reported separately.

The stdlib-only [auditor](audit_directions.py) independently regenerates [RESULT_DIRECTION_AUDIT.json](RESULT_DIRECTION_AUDIT.json):

```sh
python3 deliverables/batch_20261004/audit_directions.py
```

It checks all scalar payloads for finiteness, complete case/map coverage, application medians, exact expected counters, recorded snapshot counts/bytes/dtypes, fixed gates and code/policy provenance. All 17 local payload hashes match the frozen policy and the root's staged preflight receipt; policy SHA256 is `a01ba4cc03b14b0f7f8a78281f82fe98e8610c7329fbb176fd2fc9d50548e07b`. The preflight receipt records 47 tests and 34 subtests passing. The audit JSON contains each raw receipt's SHA256 and all per-case setup costs.

## C: local collision scaling is cheap but misses the useful physical direction

The table uses the **actual constructed field**, evaluated separately by original full H. Means are over the two fixed development IDs, not over independent source samples.

| Grid/input | Strong mean residual ratio | Mloc mean residual ratio | Mloc/strong application cost | Residual gate | Cost gate |
|---|---:|---:|---:|---|---|
| N128 source | .282863 | .980212 | .297805 | fail | fail |
| N128 one-action defect | .267396 | .980220 | .282115 | fail | fail |
| N256 source | .084072 | .989520 | .107854 | fail | pass |
| N256 one-action defect | .119359 | .986031 | .110519 | fail | pass |

The policy compares **mean squared** ratios, rather than squaring the displayed mean ratio: Mloc may exceed strong by at most .02, while its mean component application cost must be at most .25 of strong in every grid/input stratum. Its squared-residual excesses are .878690, .878888, .970206 and .954350. All four residual strata fail decisively. Mloc mean component medians are .940–.990 ms at N128 and 1.933–2.008 ms at N256; cheap application alone does not establish useful replacement quality or convergence.

The actual-output identity

\[
r-Hz=(r-Sv-C_0v)-S(z-v)-(Cz-C_0v),\quad v=R_0r,\ z=M_{\rm loc}r
\]

closes to at most (1.12933\times10^{-12}\) relative to the input norm. Reference and local collision closure are tiny (maxima (9.52538\times10^{-13}\) and (3.98735\times10^{-13}\)). The streaming material penalty has norm **6.27460–98.1918 times** the input norm and matches the unscaled actual residual norm to approximately (1.42\times10^{-14}\). Eigenbasis conditioning is 1.05777, reconstruction is (1.24560\times10^{-15}\), and every frequency-cache closure is below (7.47\times10^{-14}\), comfortably inside the unchanged guards.

Thus the observed negative is attributable to the surviving streaming/material term for this prescribed ordering, rather than a visibly inaccurate local inverse or homogeneous cache. The identity is a measured FP64 diagnostic, not an interval proof. It does not prove that all material-adapted maps, other orderings or a coupled multi-column space are ineffective.

## D: source-aligned coarse information does not transfer to this defect

| Grid/input | Analytic-column mean ratio | Teacher-column mean ratio | Mean squared-residual gain | Fixed .02 information gate |
|---|---:|---:|---:|---|
| N128 source | .960350 | .715219 | +.402619 | pass |
| N128 one-action defect | .998945 | .999629 | −.00136875 | fail |
| N256 source | .960725 | .784212 | +.301702 | pass |
| N256 one-action defect | .999850 | .999694 | +.000311401 | fail |

The coarse teacher carries more source-response information than the matched analytic column. It still leaves much more source residual than the strong action in C, whose recorded source fields are bit-identical across C/D. On the defect after one strong action, neither fixed source column supplies the required improvement; the teacher column was not acquired for that defect. D therefore fails its all-strata information gate.

This is a **plain attained real-line** correction. No actual Arnoldi AZ projector or full response-space optimization was used. The corrected field is `alpha*x` at the source and `b+alpha*x` at the defect, with fixed b; its coefficient minimizes the measured one-column residual over real scalars. The result does not establish a general coarse-range impossibility, optimal coefficients for a larger space, a reusable inverse action, OOD robustness or a learned construction.

The stored source/geometry parameter identities pass across coarse/fine parents; both columns use identical strict negative-Nyquist removal and Fourier normalization. Sampling the prolongation returns the **filtered** coarse teacher to (8.34\times10^{-17}\) relative discrepancy. The filtering changes the original teacher by .0003444–.0003970 relative norm, so all-N64-coefficient or exact fine/coarse-operator equivalence is not claimed. Saved historical teacher acquisition cost remains **unknown, not zero**.

## Counts, original-H checks and costs

| Job | Scored rows | H actions | Strong M actions | Extra S actions | Saved fields | Snapshot bytes |
|---|---:|---:|---:|---:|---:|---:|
| C | 24 | 52 | 28 | 16 | 60 | 471,933,840 |
| D | 16 | 36 | 4 | 0 | 36 | 283,160,304 |

Each row has one candidate H evaluation and one separately constructed-field H evaluation. Four initial strong-source H evaluations supply the remaining actions. Every candidate and combined field was persisted/read back before its physical check. Maximum constructed-image reconstruction gaps are (7.22777\times10^{-12}\) (C) and (6.58655\times10^{-12}\) (D), below the unchanged (10^{-10}\) guard; actual and image-fit residual ratios differ by at most (1.36\times10^{-14}\). Thus these negatives are supported by actual rounded combined-field H checks, rather than image fitting alone. They remain ordinary floating diagnostics, not residual/solution certificates.

C completed with exit0 in 25.03498 s driver time / **25.74613 s external process**; D completed with exit0 in 19.77543 / **20.71103 s**. Neither timed out. Peak GPU reserved memory was 1,797,259,264 / 847,249,408 bytes, below the 4 GiB cap. Both snapshot totals are below 512 MiB. Full solves, training and new-seed generation are all zero.

Application medians exclude the separately recorded setup, disk/transfer/readback, persistence, line fitting and H checks. C candidate setup totals .180461 s; D loading/prolongation setup totals 1.722311 s, before unknown historical teacher acquisition. D's approximately 14 microsecond application measurements return an already prepared column; they are not acquisition or solve costs. First-context setup/kernel costs, CPU worker contention and startup remain in complete receipts. The whole process is the correct bounded-job cost; sums of diagnostic categories do not reconstruct an unmeasured deployable throughput or process-cold solver ranking. Common preflight additionally records .355232 s compilation, 13.509225 s test-process wall and 36.489903 s preservation, distinct from C/D process clocks.

## Verification limits and stopping boundary

This reviewer checked scalar provenance and local authored hashes, not remote binary arrays or a fresh FWI preservation seal. The manifests attest that the new candidate files remain on FWI; their bytes were not exported or independently rehashed here. The frozen code and successful completion support its actual buffer-identity/readback guards, while future bit replay of H outputs is unproved. These are two stored development cases per grid with three component timing repeats, not a statistical generalization study.

Retain both negatives and stop new direction research this round. A/B may establish a useful checker result independently, but C/D do not admit full-solver comparison or modify the unchanged 10% learned target, physical accuracy gates or sealed holdout. No range impossibility or hardware time floor follows from these failures.
