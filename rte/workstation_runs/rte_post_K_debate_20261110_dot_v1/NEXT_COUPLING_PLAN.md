# After K: coverage, information, or repeated coarse correction?

Status: read-only source audit and two-round direction debate. No new numerical job, factor, mode selection or cutoff sweep has been performed. This proposed gate awaits review. K's single-forward3.9642% cost margin is not treated as robust dominance.

## Audited facts and implementation boundaries

K preserved the actual background bit for bit while moving the coefficient-deviation energy outside P8 from0.0015832% to20.4042%. E's outer calls became3 per source, but inner iterations fell to4. The result does not establish that missing coefficient frequencies alone determine convergence. Higher-frequency streaming and changed source alignment may both matter.

Audited sources and hashes:

- Sealed E angular_modes.py: d152afea90179efa8342a4f12631979951503785330f3aab08d4b7f1fe32a481.
- rte/augmented_fgmres.py:21ddfb628d35738ca88844d6c8587b0b8809956ed000bc7a859237d12b6bc742.
- rte/source_initialization.py:d839a76c104cdedf0017d6b19b3aecb837067d4170819939b137efe07b6543af.
- K driver:4507fe32246b091cc30d7e76961d12f08557fdfbf6a4c39f44e111e22d762198.

AngularBandMode accepts any valid stored tangent FFT index k. Its actual symbol is high.ky.flatten()[k], and original() preserves the full stored angular collision and normal derivative. In contrast, FullAngularModes.__init__ hardcodes range(-8,9). Its project/apply routines iterate over the constructed dictionary and are not intrinsically limited to contiguous modes. A different17-mode set would require an isolated new factory and consistent projection everywhere, not a silent edit to sealed E. It would require original-A0 qualification of every selected mode, including any newly selected large-frequency/Nyquist mode; no positive/negative conjugate reuse has been proved.

augmented_fgmres has a capture_prefix callback. It can expose the current solution, original residual, a basis of search images, iteration counts and residual ratio after the first prescribed classical prefix. K did not enable that callback. K's saved files contain scalar statuses, inner residuals and G-repair profiles, not physical prefix-residual tensors or resumable Krylov states. Older banks from different materials are not replacements for K's missing prefix fields.

The function has no pause/resume state interface. Calling capture_prefix and then restarting from its solution loses the existing Arnoldi/search space and changes work and arithmetic. A pooled four-prefix selector therefore needs either paid duplicated prefix runs or a separately qualified resumable implementation. Neither can be called free reuse of the same algorithm. The ordinary reference solutions cannot supply proposed directions.

The current one-shot limit is explicit: a proposal is attempted when inject is present, `not attempted`, and classical_calls >= inject_after. attempted is increased before acceptance/rejection tests. The solver retains its Krylov basis and checks the true original residual after every candidate. A bounded second proposal can therefore be studied with a small isolated scheduling change while preserving the existing basis; it does not require capture-and-restart.

## Debate round one: three different mechanisms

### A. Increase a fixed classical subspace

One fixed P12 versus P8 comparison would ask whether additional retained frequencies actually lower outer work enough to pay. It is a conventional deterministic coverage control, not a learned method or a new information source. It must pay25 rather than17 mode factors, every additional qualification and inverse action. P12/P16 sweeps or choosing a cutoff after seeing the outcome are excluded.

Positive case: K moved substantial coefficient-deviation energy outside P8 and now needs one extra classical outer call. More coupled modes could correct the remaining components. Counterargument: the inner solve already became shorter; missing coefficient energy is not identical to solution/error content. A larger space could merely add factorization/repair work. It would conflate more information capacity with more computational budget and might miss the issue of re-coupling after a single correction. Do not automatically enlarge P8 to rescue K.

### B. Keep17 modes but change how their indices are chosen

A prespecified classical selector could score the original TSA-prefix residual spectra rather than always using the17 lowest indices. This changes information source and construction timing while keeping factor count fixed. An honest pooled rule could average four normalized residual energy spectra and select a single17-mode set, with deterministic ties, then freeze it for the four-source batch. It is not valid to use ordinary reference solutions or retained teacher fields to choose the set.

The strongest matched null is to perform exactly the same prefix collection/scoring and then use fixed P8. The decision baseline remains the original one-shot E without that overhead plus strong native/compact controls. All original-A0 checks and full-original-A acceptance remain necessary. Residual energy is only a diagnostic: streaming may make a large-energy high-frequency component easy, so energy ordering is not an optimal-mode theorem.

Pooling all four prefixes delays first-source completion. The current interface also makes pooling expensive: either discard/recompute prefixes, paying their duplicated TSA/A work, or implement and bridge a four-state continuation scheduler. A first-source-only lazy selector is a cheaper distinct alternative, because inject already receives the first residual and can build factors there, but it risks fitting source0 at the expense of the other three. It needs a lazy fixed-P8 null to separate selection from construction-order effects. These are not interchangeable implementations to try until one wins. The source audit rules out pretending pooled prefixes are currently saved or resumable. Defer this line until there is a clear reason to pay its additional engineering/information cost.

### C. Keep the original P8 factors and allow one later coarse correction

Test whether a single correction is insufficient after the ordinary outer iteration mixes the error again. Build the same E factors once, use the same original inner solve, and allow at most two coarse-direction attempts at fixed classical-call thresholds1 and2. The second attempt occurs only if the original convergence test still requires work. Preserve the Krylov space, original A, original tolerances and total iteration cap. Do not apply a second correction blindly after convergence.

This is a classical change in coarse/outer coupling, not a wider basis or residual-derived factor. It is smaller than implementing a balanced multiplicative two-level preconditioner, which would change every preconditioner application and require a matched classical two-level baseline. The current proposal is only the bounded second injection, not both alternatives.

Counterargument: the second inner solve, its original-G repairs, original-A images, MGS/least-squares work and diagnostics can cost more than any outer work avoided. An extra injection is not automatically equivalent to one fewer TSA64 call. If it is rejected, nearly dependent, or followed by the same third classical call anyway, that is useful negative evidence. Sharing already paid factors is legitimate within the same four-source method, but the extra applications are fully charged.

## Debate round two: cost headroom and selected question

K's E inner work cost3.689947765 s across four sources, while its entire complete-time advantage over compact4 was only0.944840971 s. If four additional inner corrections cost roughly the same and do not replace real outer work, the margin is gone. The second residual can be easier or harder, so3.68995 s is a budget guide, not a predicted cost or a bound.

The native TSA64 batch used20 classical calls in roughly24.5 s of solve/check work. Its average is only a coarse budget proxy; it is not the marginal cost of the third step of augmented FGMRES, where basis size, extra A calls and orthogonalization differ. No exact saving can be inferred from these aggregate clocks. Avoided work must be measured in the actual continuation.

Select C: the narrow question is whether one second, fixed-schedule P8 injection removes actual subsequent classical work and improves full paid cost on K's existing four sources. It is a cheaper causal distinction than enlarging the space or introducing a pooled residual-state controller. It holds the factors and retained frequencies fixed. A failure says this specific extra-correction schedule is not worth its cost; it does not prove that all coarse/outer coupling strategies fail.

The strongest attribution/null control is the original one-shot E, with identical P8/inner8 and fully paid setup. The same-method second injection is compared against continuing the original classical outer steps, not against an intentionally weakened solver. Native TSA64-zero, actual TSA16-zero and compact4 remain the decision controls on the identical material/sources. A second-injection candidate must beat the cheapest accepted control including one-shot E. Merely remaining slightly faster than compact4 while losing to one-shot E is a NO-GO for this extension.

## Proposed minimal implementation and qualification

Use a new directory; leave all sealed E/G/I/J/K data and source unchanged. The only new solver code is an isolated copy/wrapper of augmented_fgmres with a fixed attempt schedule and an optional read-only observer. Default schedule(1,) reproduces the original one-shot condition; candidate schedule is exactly(1,2). The predicate is equivalent to attempted < len(schedule) and classical_calls >= schedule[attempted]. Preserve the increment-before-acceptance behavior, dependence rejection, residual nonincrease safeguard, MGS, least-squares, restart25, final A check and total maxiter100. No reset of the Krylov basis, altered preconditioner, adapted schedule or third attempt is allowed.

Before real comparison, perform zero-solver AST/interface checks that the one-shot schedule is equivalent to the original guard and that rejected proposals cannot cause an unlimited attempt loop. A small interface check should cover inject=None, zero RHS, rejection and the two-attempt cap. Preserve failures; do not silently fix a failed numerical experiment and rerun it.

Then qualify schedule(1,) against the original function on fixed source0 of the same K material, using the same fully paid P8 factor as a research-only bridge. Use fresh zero initial states. Require matching accepted/rejected/classical counts and full-field relative difference <=1e-12; retain both original residuals and all work. The bridge factor and fields are not reused free by timed methods and are never a teacher. This bridge is overhead in the aggregate research ledger, not a benchmark advantage.

## Proposed fixed execution gate, after review only

1. Verify the approved plan, all E/production hashes, K's exact permuted coefficient hashes and original unpermuted source hashes. Recompute actual min/gather background and require the identical K arrays/axis. No new multiplier, material, source, seed or held-out data is introduced.

2. Execute the bounded one-shot equivalence bridge and all17 original-A0/zero/tiny representation checks. Stop on any true identity or numerical failure. Runtime requested/effective/actual-object TSA depths remain hard-checked; actual TSA16 must be16, others64.

3. One fresh fixed sequence: ordinary TSA64-zero, compact4, TSA16-zero, original one-shot E, then two-shot E. All methods use the original four sources[0,4,7,14] and pay fresh physical/material/native cache/factor construction. Each E batch pays its own17-mode original-A0 and zero/tiny qualification. The ordinary accepted TSA64 outputs are used only for field checks, with their entire cost separately visible.

4. Candidate uses the same E factor data for both scheduled injections and across the four sources within its own batch. Every inner solve remains full-grid P8 A P8 at1e-4/max8; both signed modes-8..8, all original angular coefficients, original normal Nyquist and max3 original-G repairs at1e-10 remain unchanged. Each outer direction still uses the full original A. Original residual1e-5, evaluated relative bound1e-3, field difference1e-4, outermax100 and zero initialization remain unchanged.

5. Record each injection's input residual norm, inner iterations/residual/time, acceptance/rejection, G applications/corrections and added original-A work. A read-only observer may record P8 and complementary residual norms immediately after the first accepted injection, after the second classical call and after the second injection. Its FFT/copy/time is charged to the candidate. This does not steer the fixed schedule. The first-injection input/state should match the original one-shot path to the bridge tolerance; do not substitute a residual from a reference solution. Do not claim an observed norm change uniquely proves the re-coupling mechanism.

6. Report first-source and full four-source observed wall, all setup/qualification/inner/outer work, original A counts, total accepted iteration count, classical calls and both attempts. The setup-stop rule uses the cheapest complete accepted control. A preliminary positive requires all original acceptance gates, a reduction of classical calls on at least one source, and candidate full observed cost below every accepted control including original one-shot E. No call reduction means the proposed mechanism did not remove the targeted work, even if a noisy clock happens to be smaller. Any true numerical/resource/budget failure or final cost failure ends the gate. Do not change the schedule, remove diagnostics/validation, enlarge P8, add RHS or rerun for a favorable result.

Proposed resource ceilings:CPU0 one thread, no CUDA,240 s aggregate numerical controller wall,90 s per stage,4 GiB worker RSS. Five N128 full batches plus the short bridge/qualification plausibly fit the budget based on K, but no runtime guarantee is assumed. Check admission at every executable stage and record shared-host activity. A cap breach stops only this owned worker and is retained, not repaired by increasing limits. A mechanical implementation failure is preserved and reported for review, with no automatic numerical retry.

## Limits and next decision

This test does not settle the value of P12, arbitrary17-mode selection, or a multiplicative two-level method. It distinguishes one precise same-factor coupling change from paying for more frequency coverage or new source information. The existing CPU-constrained positioning remains; separately measured GPU compact4 latency is much lower in the previous N128 workload, and CPU savings here do not imply GPU usefulness. G/I remain closed, reserved9626 remains unopened, and K's narrow single-order result remains unchanged.

On success, report the actual call/cost tradeoff and seek review before confirmation. On failure, preserve whether the cause was no classical work avoided, extra inner/orthogonalization cost, rejection, or accuracy/budget failure. Do not move automatically to P12/P16 or a different second-injection point. No new numerical work has been started for this proposal.
