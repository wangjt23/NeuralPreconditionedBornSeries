# E at N256: proposed pure-scale qualification

Status: planning and stored-input inspection only. No new material/source was generated and no A action, factor, solver or GPU work was run for this plan. I and G remain closed. The candidate is the unchanged sealed E sequential angular implementation, not the reduced-coordinate I implementation or batched F.

## 1. Correct identity and available inputs

E's material is the fixed `screen_structured.medium` layered-plus-Gaussian scene, not one of the sixteen seed4626 random materials. Its four sources come from development IDs[0,4,7,14]. Moving to a random development material at N256 would change both material and grid, so that alternative is rejected for this scale question.

The proposed N256 material is a new discretization of the SAME fixed continuous construction: periods1, beta10/3, two-layer profile, transition physical width2/128, inclusion center(.34,.60), width.08, amplitude3 before scaling, total=.05+scatter and scatter=beta*(T-.05). The transition must NOT be changed to2/256. Call the unchanged `screen_structured.medium(256,1,10/3,'cpu')`; do not select a new coefficient geometry, adjust a parameter or reconstruct an idealized angular discretization. No existing N256 copy of this constructed material has been established. It must be called a predeclared new resolution, not a previously stored or globally unseen material.

Existing N256 sources DO exist. `training_priority_20261004/benchmark_joint/neural_field_n256_case{0,4,7,14}.pt` stores `source` separately from `field`, with grid/index/parent hashes. Read-only inspection verified all four:

- The corresponding saved N128 source equals E's original bank q promoted to complex128, byte for byte.
- Each N256 source sliced at[...,::2,::2] equals its corresponding N128 source exactly.
- Each source record names the existing N256 development parent SHA256 fb346212e23752d289a06e7f9086fee94a0ca2fa2b16dc6b840ddd14adc64fa8.
- The N128/N256 parent material arrays are nested exactly; centers, phases, amplitudes, angular codes and geometry codes match exactly between parent files. These parent coefficients are provenance only and are NOT substituted for E's constructed material.

`SAVED_INPUT_AUDIT.json` fixes exact source tensor and containing-file hashes. Reuse only the `source` entries. Do not use the neural `field`, checkpoint, saved solver status or its earlier performance to choose inputs, initialize E, provide a reference solution or claim a teacher. Inspection used CPU memory-mapped loads and hashes/equality only; it did not evaluate the saved solution fields. No source regeneration, seed search, interpolation or CUDA generation is needed.

N256 is an already used development grid. Seed9626 remains unopened under the old protocol. The old16-material study and this one constructed-material/four-source study are distinct workloads. A pass would establish scale transfer on this fixed scene, not material generalization, convergence of the continuous PDE or an untouched test result.

## 2. Debate round one: why a scale test is now useful

Positive case: E's CPU advantage survived the known amplitude3-to6 coupling stress, and its matrix representation is independent of normal-grid size. With fixed17 tangent modes, doubling the normal block size changes factor and application costs differently from compact4's PCR. The next unresolved question is whether reduced construction cost continues to pay at the finer scale, not whether another amplitude can be selected to win.

Strong objection: the inner P8 approximation keeps the same17 physical tangent frequencies while the full spatial space grows. It may require more inner iterations or more outer calls; max8 is unchanged. The approximate angular band also sees larger stored normal frequencies, so one defect correction at N128 does not guarantee one at N256. A good factorization bridge alone is insufficient for a full solver claim.

Three distinct options were compared: unchanged E on a pure resolution extension; a new random-material study on existing seed4626 fields; and changing the inner representation or GPU execution. Select the first. The random-material option changes the question and needs its own unscreened protocol. I/G have preserved cost failures and are not reopened by this test.

## 3. Strong classical controls, including the old N256 winner

The saved global `frozen_selection.json` names TSA16-zero at N256, SHA25667cf2328721db970470643fdae52c7df7db7a191a2db2f56731ffb5c63fb6a49. This selection was on sixteen different development materials with one source each, mostly on GPU. It cannot be imported as a measured winner on E's scene, but it must not be ignored.

Use three freshly paid CPU controls on the SAME new N256 scene and original four sources:

1. Native TSA64-zero, with original `solve_from_initial` and no injection. Its four ordinary1e-5 accepted outputs also supply field checks for the other methods. Record complete setup, first-source and four-source costs, plus independent original residual/bound. This is an ordinary reference/control, not a high-accuracy teacher or free oracle.
2. Compact4 with the existing Nyquist repair, complex64, TSA64 outer preconditioner and one injection after prefix1, exactly the established E comparison architecture.
3. Native TSA16-zero, using `build_base(low,'tsa16')` and the original zero-initialized solver. No inherited balanced initialization, learned correction or uncharged material/cache state.

The decision threshold for E is the least complete accepted cost among all three controls. This is stronger than comparing only with compact4. Native TSA16 has a portable CPU implementation in `ClassicalTransportStack`; its production source has been read and hashed. Existing N128 structured-scene GPU receipts recorded17 TSA16 calls versus5 TSA64 calls for each of the four sources. That is a call-pattern observation, not a CPU timing prediction or a N256 guarantee.

The qualified `DirectSegmentedTSA` backend explicitly requires a CUDA complex64 prototype and uses CUDAGraph/streams. Its improvement was capture/execution overhead, while `_first_stage` and `_subsequent_stage` preserve the native recurrence. It has no qualified CPU capture backend. Do not instantiate it with fake CPU capture, call the CPU native path a direct-backend timing, or copy its GPU milliseconds into this comparison. The native TSA16 CPU control represents the applicable recurrence; a future actual GPU decision comparison is separate and not authorized here.

The attribution control is unchanged E source plus exact material/source nesting and independent original-A0 qualification on N256. N128 receipts remain historical context. No dense N256 LU oracle is needed: it would add an expensive new implementation without strengthening the direct original-G/A0 residual tests. Do not require N128 iteration counts to persist at N256; the discretized operator has changed. Record any increase under the fixed tolerances/maxima as the scientific outcome, not a parameter to repair.

## 4. Exact storage ledger and honest cost estimates

Let N=N_normal=N_tangent, L=12, complex128 E and complex64 compact4. The existing source's logical-byte accounting gives:

  E(N) =17*(992*N^2 +80*N +9416),
  compact4(N) =(2*log2(N)+2)*N^2*144*8 +N*144*8.

These reproduce the sealed N128 figures. They count logical references in the same manner as the old implementation; they are not exact RSS allocations or proof of GPU fit.

| Logical storage | N128 | N256 |
| --- | ---: | ---: |
| E factors/recorded mode data |276,633,928 bytes|1,105,707,336 bytes|
| Compact4 PCR/Nyquist factors |302,137,344 bytes|1,359,249,408 bytes|
| One physical complex128 field |3,145,728 bytes|12,582,912 bytes|

At N256 the four sources plus four ordinary reference outputs occupy100,663,296 bytes. Original physical/TSA caches, factor-construction temporaries, full-grid inner/outer Krylov vectors and the interpreter add more. An approximate3-5 GiB process RSS envelope is plausible, not measured. Propose a6 GiB owned-process RSS ceiling and refuse execution if current host admission cannot safely provide it. Keep only one candidate/control factor alive at a time; no simultaneous E/compact or dense factors.

E's dense normal-block LU/Schur construction has a leading cubic dependence on N, so doubling N may multiply that phase by about8. Vector block solves grow roughly4; full2D FFT work grows approximately4*8/7. Compact4's retained PCR work scales about4*8/7 in this size step, not cubically in N. These are operation-count estimates, not wall-time predictions. Using recent N128 costs only as a budget guide suggests E factor construction on the order of8 seconds, compact construction around13 seconds and full four-source methods on the order of70-130 seconds if calls remain comparable. Higher call counts, memory bandwidth or factor behavior can defeat those estimates. The unchanged limits decide, not the estimate.

This test explicitly asks whether construction, application, original validation or RSS becomes the new bottleneck. Record all four. Do not hide a slower first source behind the four-source total, increase the number of sources or amortize factors across methods.

## 5. Locked proposed execution sequence

All work is in a new directory; sealed E and the prior packages are read-only dependencies. CPU0, one thread, CUDA disabled. Check current admission before each executable stage, recording other numerical work without inventing a whole-host exclusivity requirement. No installs, paid resources, deletion or security changes.

Stage A: input and representation qualification, at most180 seconds.

- Verify E/production/dependency and containing-file hashes, then extract ONLY the four saved N256 sources. Record exact q256 hashes and the already proven q128/E-bank links again. No seed generation.
- Evaluate the unchanged constructed material at128 and256 only for identity/qualification. N128 arrays must match E's stored amplitude3 hashes. Require exact coefficient nesting from256 to128 on even grid points. If the nested construction fails, stop and report the actual discrepancy; do not replace it by interpolation or a different formula. Record new N256 material/background hashes and axis. Require the existing E normal-axis0 contract.
- Instantiate the original PeriodicRTE2D at256 with dx1/256 and the same L12/g/.0 shifts; promote actual stored geometry. Do not compare full operators or Nyquist symbols as though the two discretizations were identical.
- Build one complete E factor using the unchanged `FullAngularModes` class. It supports n through the actual profile length; only the old driver's hardcoded1536-vector/128-grid checker must be replaced by a generic12*N check in the new driver. Do not edit E's source.
- Independently check all17 modes through the full N256 original A0, each with a locked deterministic complex vector of length3072; seed2026111004, same signed order, hashes frozen in the new receipt. Add strict-zero, mixed-zero/ordinary/tiny columns for the fixed modes[-8,0,8]. All original-G/A0 relative thresholds remain1e-10 and repairs at most3. Scaled norms avoid underflow without an absolute tolerance floor. Preserve all outcomes, including a second/third correction or failure.
- Record fully paid construction phases, qualifications, storage and RSS. Delete only in-memory references when the stage ends; no file deletion. Qualification factors cannot be reused by a timed method for free.

Stage B: native TSA64 reference/control batch, at most180 seconds. Fresh material/operator/cache, same zero initialization and original four sources. Ordinary tolerance1e-5/max100, with the original full-A residual and evaluated bound. Save four accepted fields only in memory for subsequent field checks; do not use them in Krylov directions or initial states. Pay and report the complete first/four-source observed wall. No more accurate solve is allowed if this stage fails.

Stage C: compact4 control batch, at most180 seconds. Fresh material/operator/TSA64/compact factors. Retain the established one-shot injection and all original final checks against the ordinary references. Report direct first/four-source wall, factor/application phases, original calls and RSS.

Stage D: TSA16-zero control batch, at most180 seconds. Fresh material/operator/TSA16 cache, original zero-start solve and same four sources/checks. Do not use an old CPU/GPU runtime or discard a failed source. This control is paid even if it loses to compact4.

Stage E: one E candidate batch, at most180 seconds. Fresh material/operator/TSA64 and all17 E factors. Pay the same independent17-mode A0 checks and fixed zero/tiny qualification at this size. Use the original full-grid E inner action, P8 modes-8..8, inner1e-4/max8, prefix1, outer1e-5/max100. Do not substitute I's compressed action or F's batched inverse. Compare every final field with the ordinary references, requiring residual1e-5, evaluated bound1e-3 and field difference1e-4.

Before E's online work, stop if its paid setup lower bound already reaches the smallest complete control cost. Otherwise finish all four original sources, charging every conversion, full-grid projection, repair, original-A check and bookkeeping. Record mode repair histories, inner projected residuals, outer calls and true first-source/four-source observed wall.

This is one forward scale gate, not an unconditional timing campaign. If E fails any original mode/solver acceptance, exhausts budget/RSS or does not beat the cheapest accepted control in complete observed four-source cost, stop and preserve the result. If it passes, report a preliminary N256 scale result and ask for review before any reversed order or broader study. One forward observation cannot promote a repeated-performance claim.

## 6. Debate round two and stop conditions

The cheapest meaningful test is not a full random-material benchmark or a N256 dense inverse. Exact existing-source nesting resolves the main provenance uncertainty without new data. All17 original-A0 checks then decide whether E's representation survives the new size. Only after that should the paid reference/controls/candidate run.

Adding TSA16 is necessary because the older global N256 study identified it as a strong algorithmic alternative. Adding its CUDA capture wrapper is inappropriate in a CPU-only experiment. Reusing ordinary TSA64 outputs as field references avoids a separate accurate-teacher task while keeping their full preparation/solve cost visible as a control. Fresh E/compact factors prevent same-background reuse from inventing a gain.

Proposed hard ceilings:600 seconds aggregate numerical wall across all stages,180 seconds per stage,6 GiB process RSS, one CPU core, no CUDA. The aggregate cap overrides the sum of stage allowances. A simple owned-process monitor can stop only this worker on RSS/elapsed breach; interrupted/partial costs must remain in the receipt. No retry after a real numerical, identity, acceptance, RSS or budget failure. A narrowly mechanical interface mistake may be corrected only with its failed version/cost retained and an independent interface check, without changing mathematics or thresholds.

Material/source hashes, first-RHS/full-batch timing, all setup/validation phases and every failure must be retained. Common input-file identity inspection is research preparation with its cost separately reported, not hidden as process-cold latency. Timed methods receive the same already loaded source arrays and pay their own physical/material/factor construction; no GPU transfer exists. CPU and historical GPU timings remain separate.

The sole conclusion at issue is whether the established E CPU cost benefit persists when the same constructed physical scene and exactly nested four-source family are discretized at256. The outcome cannot unlock reserved9626, establish general material performance, revive I/G or justify more RHS. Execution awaits explicit plan review.
