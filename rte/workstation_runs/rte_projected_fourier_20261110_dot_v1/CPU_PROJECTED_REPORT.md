# Projected Fourier CPU qualification: accurate representation, paid-cost NO GO

The finite P8 Fourier representation passes its independent action/inverse bridges and preserves the original solver's call pattern. Its inner phase is21.89% cheaper in the one prescribed forward comparison. Nevertheless, the complete candidate costs17.294708 seconds versus15.746670 seconds for the existing sequential E implementation. The additional paid entry qualification outweighs the online saving. The locked rule therefore stops this implementation without a reverse triplet, threshold change or retry.

This is a qualified representation milestone and a complete-cost NO GO under the frozen four-source protocol. It is not a numerical failure, a broad impossibility result or a reason to discard E's established CPU gains. G's eager GPU branch remains closed.

## Original problem and implementation identity

The problem remains N128, L12, g=.65, the stored amplitude3 inclusion material and source IDs[0,4,7,14]. Material and source-array hashes match the sealed E/B records. The selected background remains normal axis0 and is freshly constructed for every batch. Production files and E's `angular_modes.py` are unchanged.

The new state has shape[1,12,128,17] instead of[1,12,128,128]. Multiplication by each actual stored coefficient uses a17-by17 matrix per normal point with entries FFT_backward(sigma)[(k-l) mod128]/128. This keeps the exact discrete projected product in exact arithmetic. It does not assert that the original coefficient is band-limited, use a17-point cyclic convolution, replace the full angular kernel or regenerate ideal geometry. Normal streaming uses the original stored-derived symbol and its Nyquist convention. Both total and scattering arrays are transformed independently.

The original E mode blocks, original-G repair target1e-10/max3, inner tolerance1e-4/max8 and outer tolerance1e-5/max100 remain fixed. Inner FGMRES imports the original real-stacked implementation unchanged. Every reduced injection is lifted and checked with the full original A for projected residual<=1e-4 inside its timed solve. Original final-A residual<=1e-5, evaluated relative-error bound<=1e-3 and relative field difference<=1e-4 remain mandatory. The evaluated bound is an ordinary FP64 discrete calculation, not an interval or continuous-RTE certificate.

The frozen plan SHA256 is19b95306d355f2fae8f5e46d038d207f49fa49d53f989ac6a8991890cf1ea902. Executed sources:

- `projected_operator.py`:568450e3a3ad80da3ce0cb9d86d4c937a785700228f7ed667cde46aef36b5d45.
- `run_projected_gate.py`:ac460829c7a72fb69bc4c60a6f801235b3a7717274acfdb88281178c7279284b.
- Unchanged E inverse:d152afea90179efa8342a4f12631979951503785330f3aab08d4b7f1fe32a481.

No executable repair was needed. No new physical sample, N256 case, reserved9626 access, training or accurate teacher was used.

## Complete paid costs

The locked order was compact4, original full-grid E, reduced-coordinate E. Every batch rebuilt physical/TSA/factor objects. Both E batches paid all17 original-A0 checks. The reduced batch additionally paid coefficient construction and the full action/inverse/zero/tiny qualification suite, including its diagnostic bookkeeping. No qualification result or useful factor was reused for free in its measured setup.

All values below are CPU seconds from this one forward triplet. First RHS is a direct timestamp from fresh setup through accepted source0, never one quarter of the total.

| Cost | Compact4 | Original E | Reduced E |
| --- | ---: | ---: | ---: |
| Complete observed setup |3.021422|1.616126|3.932535|
| Four-source online phase sum |17.699145|14.116564|13.348090|
| Inner-solve phase within online |0|4.186755|3.270407|
| Observed first RHS including setup |8.040303|5.217405|7.336714|
| Observed complete four RHS |20.734829|15.746670|17.294708|

The reduced path is9.83% slower than E in complete cost, although16.59% faster than compact4. Beating compact4 alone is insufficient when a stronger already-qualified CPU implementation exists. Its online phase saves0.768474 seconds, while setup grows2.316409 seconds. Its inner phase saves0.916347 seconds including the additional four full projected-A checks. These are observed differences, not repeated-performance estimates or additive speedups from separate studies.

The new coefficient construction itself costs.002528 seconds and stores1,606,384 bytes including coefficients, symbol, indices and angular data. The measured action/inverse probe suite costs2.082432 seconds, with further diagnostic setup bookkeeping retained in observed wall. E factor construction is.963744/.952753 seconds for original/reduced, and their17-mode validation costs.547129/.524090 seconds. The raw receipt contains every E chain/corner assembly phase. There is no CPU/GPU transfer in this experiment.

Do not subtract the qualification after seeing this result to announce a deployment win. The paid setup contract was fixed before execution. An alternative validation contract would be a different research/deployment question requiring explicit reasoning about which input-dependent checks remain necessary; this run does not authorize it. No source-count increase or unmeasured reuse is used to manufacture amortization.

## Independent numerical qualification

The preliminary suite and candidate's paid setup repeat the same fixed probes: the existing17 locked vectors, tangent endpoints-8/+8 with normal Nyquist, exact zero,1e-200-scaled vectors and mixed zero/ordinary/tiny columns. All17 vector hashes match E's prior qualification. Across12 probe records:

- Maximum reduced action versus restricted original-A difference:2.936842e-16.
- Maximum lifted action versus original P8 A difference:3.405595e-16.
- Maximum reduced-versus-E inverse field difference:8.599121e-14.
- Maximum original-A image difference of the two inverse outputs:2.679016e-14.
- Exact zero columns remain zero; tiny nonzero columns remain nonzero. Relative norms use scaling, not an absolute tolerance floor.

The preliminary factor and the two measured E factors supply51 independent original-A0 mode checks, maximum3.858052e-12. This checker embeds each mode into the full physical grid and uses original A after subtracting the actual inclusion perturbation, not the angular-band action.

All12 measured source solutions pass. Maximum original-A residual is9.080373e-6, evaluated relative-error bound3.945352e-5 and field difference2.648812e-7. Maximum reduced injection residual independently checked through full original A is8.882625e-5, below the unchanged1e-4 threshold.

Call patterns match exactly:

- Compact4 outer TSA64 calls:[4,3,4,3].
- Both E paths outer TSA64 calls:[2,2,2,2], inner iterations:[5,5,5,4].
- Each E path performs19 inverse applications, covering323 logical modes. Every runtime mode uses one original-G correction. Maximum repaired residual is3.930610e-12 for E and3.934045e-12 for reduced E.
- Original E performs46 full-grid inner A calls. Reduced E performs46 compressed A calls plus four independently checked full-A injection calls. Both use36 outer A calls and four final-A checks at the instrumented high-precision boundaries. These counts do not count the internal low-precision TSA stages as extra high.A_op calls.

The visible high.A_op ledger totals317 calls:51 independent mode checks,36 action/inverse-probe calls,48 ordinary-reference calls and182 measured-batch calls. The first three counts are derived from the frozen executed loops and saved iterations; measured-batch boundaries are instrumented. This ledger deliberately excludes internal low-precision TSA stages and the separate matrix-free G actions in each repair, whose complete profiles are retained.

Thus the projection identity is numerically supported at substantially tighter error than the frozen bridge threshold. It is not bitwise equality, and no mathematical claim about unseen materials or scales follows from these probes.

## Resources and closing status

PID1283946 ran on CPU0 with one thread and CUDA disabled. Initial admission and each stage/batch snapshot showed no other active numerical job; ordinary system processes remained. Schur contention was permitted by the coordinator but is not observed in these snapshots. This is not a claim of physically exclusive machine performance. No GPU was initialized or used.

The preliminary stage cost4.897858 seconds. The ordinary references plus forward triplet stage cost79.223000 seconds; its receipt is marked stopped because the final economic gate ended the campaign, although all three batches completed. All stages are below180 seconds and aggregate research wall is84.163287 seconds, below600. The four ordinary accepted native references cost24.078606 seconds, separately counted as research checks and never used as candidate directions or initial guesses. Peak process RSS was1,197,809,664 bytes, a shared high-water mark rather than a method-isolated memory measurement.

The process exited0. At13:00:39 FWI server time, `ps -p1283946` confirmed it absent. CPU0 is released. No reverse triplet or subsequent numerical experiment was started.

The coordinated Schur diagnostic was later reconstructed from its receipt mtime and9.673875-second elapsed duration as approximately13:01:56-13:02:05 server time. It therefore did not overlap this RTE numerical window. This reconstruction is not an independent timestamp log, but the13:00:39 verified RTE exit is already earlier than that window.

Retain original E as the CPU winner under this paid contract and compact4 as the available warm-GPU decision control from G. CPU and GPU seconds are not combined into an algorithmic speed ratio. The next step is evidence review, not automatic tuning. Reopening would need a materially different computational or verification contract, declared before measurement, while preserving original-A acceptance and the source budget.

## Evidence

`gate_01/results.json` contains all settings, admission snapshots, source/material/background hashes,12 source rows,4 references,51 mode checks, all bridge probes, runtime repair traces, calls and costs. `analyze_gate.py` reads only saved JSON and hashes, asserts the numerical/call/source invariants for the NO-GO case and writes `analysis.json`. `FROZEN_PLAN.md`, executed sources, this report, raw receipts, log and SHA256 manifest form the small evidence package. Sealed B/E/G/H remain unchanged.
