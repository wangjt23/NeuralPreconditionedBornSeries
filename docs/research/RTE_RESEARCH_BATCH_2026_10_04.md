# RTE FWI batch results — 4 October 2026

The resumed four-job batch completed successfully. It establishes a valid N256 engineering certificate with the unchanged accuracy policy and an equivalent, cheaper checker specialization. Neither screened physical correction justified continuation. No training or full solver campaign was run.

## Workspace and execution

Authorized host: SSH alias FWI, actual hostname `FWI-Standard-PC-i440FX-PIIX-1996`, user wjt. Repository: `/data/wjt/NeuralPreconditionedBornSeries`, branch main, commit `028caea3f35d100225f656e2b1e910c1366aaafe`. RTE means steady radiative transfer equation. This work concerns the existing periodic two-dimensional Fourier/S1 discrete complex-field problem with 12 angles, g=.65 and epsilon=.1. It makes no continuum, boundary-condition, positivity or OOD claim.

New isolated run: `/data/wjt/NeuralPreconditionedBornSeries/rte/workstation_runs/research_batch_20261004`. Existing stored development IDs0/3 from cohort4626 only. No new seed, training, teacher acquisition, heldout8626/reserved9626 numerical evaluation, original-source/field export, Git modification, dependency installation or unrelated job cancellation occurred. Other-project CPU production work was present and was left untouched. GPU setup and screens ran sequentially alongside at most two single-threaded CPU checker workers.

Submission was at 16:28:43 Asia/Shanghai; completion at 16:31:36. All workers and supervisor exited, with no timeout. The batch supervisor took 173.804 seconds including the initial snapshot process. Submission/preflight took another 51.880 seconds, including 36.490 seconds preservation checking, .355 seconds compilation and 13.509 seconds target tests. The complete FWI preflight passed **47 tests and 34 subtests**, including both actual Torch CPU tests for the new material-correction kernel.

The final preservation check verified721 files:576 prior files,42 staged files,the native binary,six frozen fixtures and96 candidate fields. The original nine modified tracked files, branch and commit were unchanged. No owned job remained; GPU use returned to37MiB/0%. This post-run byte-only audit took23.832 seconds, separately from experiment time.

## Results by job

| Job | Question and result | Decision |
|---|---|---|
| A, stagewise certificate | All eight certificates passed: four saved N64/N128 teachers and four newly frozen N256 manufactured b/a fields. Six positive paired controls passed and six negative paired controls were rigorously rejected. | Valid engineering admission under the explicit CPU arithmetic contract; no original-source N256 full-solver admission. |
| B, upper-only norm specialization | Four certificates on exactly A's N256 bytes gave identical numerical bounds, stage radii and decisions. Matched helper calls totaled115.888s versus A's136.168s, a descriptive14.893% reduction. | Preserve this specialization. Performance remains a small concurrent-load screen, not a controlled cold-throughput or solver speedup. |
| C, cheap local material scaling | `M=C(x)^-1 Cref Rref` used the full stored angular matrix and one FP64 mean-reference application. All four quality/cost strata failed the combined screen. | Stop this particular replacement; cheap local inversion does not control the streaming/material penalty. |
| D, reused coarse teacher column | The existing N64 full-angle teacher improved source fitting over the matched homogeneous column, but supplied almost no useful correction after one strong classical action. The all-strata information gate failed. | Preserve the source-information positive and frontier negative. No training or coarse-solver campaign is justified by this column. |

A and B certify the exact nominal H defined by captured original D64 and exact stored material, phase, quadrature, source and candidate bits. They carry one outward Euclidean radius through each FFT layer with `E_next <= sqrt(2)*E+e`, pay both inverse normalizations and local collision/add/subtraction errors, and use independently enclosed mathematical roots. Native binary64 primitives require nearest rounding, disabled FTZ/DAZ and the recorded non-fast-math/non-contraction compiler contract. This trust boundary is explicit; the tests do not prove all compiler/silicon arithmetic, and no global GPU FFT contract is asserted.

The fixed gates remain residual1e-5, center budget99% of that residual tolerance, enclosure-radius budget1%, relative solution bound1e-3 and independently checked paired-field tolerance1e-4. Acceptance is audited using exact rational cross-products. Worst N256 relative enclosure radius was2.46045e-11 versus the1e-7 allowance. The perturbed fields had residual upper bounds about9.537e-7 and relative solution bounds below7.926e-6; direct paired relative differences were about9.537e-7. Individual solution bounds were never substituted for the paired criterion.

The N256 fields are **engineering controls**: b is a captured TSA16 application to the original physical source, the certification RHS is the represented GPU H(b), and a is the represented `(1+2^-20)b`. Their negative paired control uses `(1+2^-8)b`. All six fixture files and every new C/D candidate were persisted, hashed and read back **before** checking. The October2 N256 **NO_CERTIFICATE** results remain unchanged; their old transient arrays were not persisted and are not a bitwise replay baseline. Passing these new cases does not certify a solve of the original physical RHS.

## Physical screen evidence

C used the full stored column-weighted phase matrix, retained complex eigenvalues, and failed closed on conditioning, reconstruction and reference-cache closure safeguards. It did not substitute a first-row circulant approximation, symmetrize the phase, clip denominators or tune parameters. Its nominal identity is `I-HM=-S(C^-1 Cref-I)Rref`; an independently recorded floating diagnostic separates reference/local closure from that streaming penalty.

| Grid and input | C material-scaled mean remaining residual | Strong TSA mean remaining residual | D teacher-column mean remaining residual | D homogeneous-column mean remaining residual |
|---|---:|---:|---:|---:|
| N128, original source | .9802 | .2829 | .7152 | .9604 |
| N128, one strong-action defect | .9802 | .2674 | .9996 | .9989 |
| N256, original source | .9895 | .0841 | .7842 | .9607 |
| N256, one strong-action defect | .9860 | .1194 | .9997 | .9998 |

These are actual original-H evaluations of separately constructed and persisted fields after a real line fit. They are ordinary FP64 diagnostics, not interval solution-error certificates or complete-solver convergence results. The defect is after one TSA8/16 application, not an actual Krylov trajectory or AZ-projected continuation.

C's streaming/material penalty was6.275–98.192 times the input norm, while reference and local collision closures were below9.526e-13 and3.988e-13. The diagnostic identity discrepancy was at most1.130e-12. This is decisive against this particular local scaling, without proving that all heterogeneous corrections fail. C's warm application component was .94–2.01ms, versus strong3.32–18.17ms, but its poor directions erase that component-cost attraction. Constructor, transfers, persistence and physical checks are charged separately and in whole-job time.

D used identical strict coarse-Nyquist removal and normalized Fourier prolongation for teacher and saved reference columns. Source/geometry parameter identity across stored parents was checked. Source squared-residual improvements were .402619 at N128 and .301702 at N256. Defect improvements were −.001369 and+.000311, both below the frozen .02 threshold. The saved original-source teacher is a fixed trial column, not a response label for that defect. Historical acquisition cost for these exact teachers is unknown and is never charged as zero. Prior September coarse64 cost-negative evidence remains a competitor. Retrieving a cached column is not the complete correction cost.

## Complete costs, scope and next ranking

| Process | Entire external process time | Work |
|---|---:|---|
| Snapshot freeze |6.753s|Six immutable fixture files; two explicit GPU H and two captured M applications.|
| A |166.736s|Eight complete certificates, twelve direct paired controls, startup/readback/hash/serialization.|
| B |119.350s|Four complete certificates on shared N256 bytes, including paired gates and all fresh-process/readback costs.|
| C |25.746s|24 result rows,52 explicit H,28 strong M,16 streaming diagnostics,60 persisted fields.|
| D |20.711s|16 rows,36 explicit H,4 strong M,36 persisted fields.|

The processes overlap; their sum is not elapsed batch time. Nested helper/action timers must not be added again to process times. Peak GPU reservation was1.674GiB, below the4GiB limit. A/B CPU peak RSS was about.45GiB each. Warm application times and the14.893% checker-call change are descriptive components under concurrent load. They do not establish complete deployed economics. The new checks still take roughly29–36s per N256 field, while retained classical solve components were around.13s with a different acceptance-cost scope. Certification should therefore serve as explicitly charged offline validation in the next scientifically legitimate benchmark, with separate GPU component timing and fully charged certified throughput reported together. Excluding offline validation cannot support a deployed end-to-end speed claim.

The best supported direction is now **further lowering checker overhead while retaining its proven target and gates**. Upper-only norms are a validated small improvement; future exact singleton dispatch or native stage/reduction fusion would need a new bounded implementation test. This batch does not authorize or run that extension. Neither C nor D supplies a viable learned-preconditioner advantage. A new original-source solver comparison would require a separately frozen protocol and actual immutable baseline/candidate final fields with fresh residual, solution and direct paired certificates. No such comparison was run here.

## Evidence and reproduction

Local evidence: `deliverables/batch_20261004/`, with frozen code/policies, independent preparation/result reviews, `RESULT_AUDIT.json`, `RAW_FILE_HASHES.json` and the scalar-only `audit_batch.py`. Raw scalar receipts are in `receipts/batch_20261004_execution/`; candidate tensors and original project sources remain only on FWI. To recompute the exact scalar acceptance/equivalence/count audit without native code, GPU, data or network:

```sh
python3 deliverables/batch_20261004/audit_batch.py \
  --root receipts/batch_20261004_execution --output /tmp/rte_batch_scalar_audit
```

Frozen execution policy SHA256: `a01ba4cc03b14b0f7f8a78281f82fe98e8610c7329fbb176fd2fc9d50548e07b`. Native library SHA256: `bf5f5d04c2821e508f83cdbd630758b1cc7b3c1ee4a1250d7ef42859a0f84786`. Shared snapshot ledger SHA256: `29f9ec208e25d74204ae8a12663fb3f8b240e99eb304aebe7b04e14e99fa2b83`. The scalar audit recomputes every acceptance predicate and checks numerical A/B equivalence, stage radius propagation, paired positive/negative decisions, physical screen counts and frozen criteria. Further numerical reruns require a new isolated run root; existing outputs must never be overwritten.
