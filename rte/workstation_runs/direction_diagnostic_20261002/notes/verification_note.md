# Independent verification of the direction follow-up — 2026-10-02

The completed FWI receipts support stopping the **tested fixed nonlinear one-shot correction as a speed mechanism**. None of its three predefined learned candidates meets the development gate: zero failures, at least 5% mean solve-time saving against frozen classical at both N128 and N256, and paired confidence intervals above zero. This conclusion concerns the existing checkpoints and protocol; it does not rule out learned preconditioning in general. No remote files were changed by this verification.

Evidence directory: `receipts/rte/workstation_runs/direction_diagnostic_20261002/`. Execution used `/data/wjt/NeuralPreconditionedBornSeries`, branch `main`, HEAD `028caea3f35d100225f656e2b1e910c1366aaafe`, NVIDIA RTX A6000, Torch 2.5.1. The full diagnostic process exited successfully in 504.449 seconds; its internal wall time was 501.115 seconds.

## Counts, correctness and preservation

I independently parsed the raw tables and recomputed their aggregates rather than relying only on the decision file.

- `runtime_n{64,128,256}.csv` contains **480 + 480 + 528 = 1,488 measured solves**: 16 media, three timing repeats, and 10/10/11 methods. Every `(medium, repeat, method)` key is unique and complete. All 1,488 solves succeeded, with finite timings and original-RHS residuals; the maximum relative true residual was `9.985011571756997e-6`. These counts exclude warmups, field-equivalence runs and the separate synchronized profiles.
- Runtime tables contain **54 deliberate rejected proposals**, all from `tsa16_noop_cost` (18 per grid). Every learned/reference proposal was accepted; the 294 actual learned injection calls include no rejection. Thus “zero rejections” is correct only when explicitly restricted to learned/nonzero proposals.
- `prefixes_n*.json` contains **96 prefix contexts**, formed from the same 16 physical media at three nested resolutions and two classical prefixes. **58 qualify**: TSA8-balanced has 13/13/14 eligible tails; TSA16-zero has 6/6/6. All 38 ineligible contexts have already reached the `1e-5` residual tolerance. Eligible frontiers have unit norm to `2.23e-16`, image Gram defects at most `1.92e-15`, and normalized residual/basis overlap at most `6.64e-11`. These are neither 96 nor 58 independent physical draws. All media remain in full runtime comparisons.
- The 58 eligible contexts each have 14 direction records, yielding **812 records** and **84 aggregate groups**. I recomputed every saved group mean. All 754 nonzero proposals were accepted, with no accepted-frontier breakdown; their largest old-versus-corrected continuation discrepancy was `3.681528534180134e-7`.
- All **192 matched-cost comparisons** (four pairs × 16 media × three grids) report exactly zero relative field difference and equal iteration/failure status. Pairing correctly uses TSA8-reference, rather than frozen TSA16, as the N256 reference-cost baseline. The no-op control correctly preserves TSA16-plain fields while triggering rejection fallback.
- Current checkpoint, frozen-selection and validation-parent hashes equal those in the preceding completed runtime gate. All five executed isolated Python source hashes match the retrieved bytes. The five separately logged original sources match the initial snapshot and its hash map; all 147 original source files present in the compact snapshot also match. The final on-host receipt verifies all **155 original sources and seven inputs**, with no source/input/model/development-parent mismatch. Eight original files and the large original inputs are absent from this compact local snapshot, so that last complete preservation check relies on the recorded on-host hash verification. Reserved seed **9626** was not generated; its test-file list is empty.
- The existing FWI protocol-test receipt reports **15 tests and 61 subtests passed**. Profiling is separately instrumented and synchronized; its sub-times overlap enclosing callback times and must not be added to the uninstrumented latency tables.

## Runtime decision reproduced

I reduced the three repeats to a median per physical medium, recomputed all 54 mean comparisons, and regenerated all 54 paired-media bootstrap intervals with the recorded seed and 10,000 resamples. Every interval matches exactly. Savings below are against frozen classical; positive means faster. Frozen classical is reference-tail at N64/N128 and TSA16-zero at N256.

| Candidate | N64 saving, % (95% interval) | N128 | N256 |
|---|---:|---:|---:|
| TSA8 mixed | 0.597 (−2.659, 3.977) | 0.385 (−2.667, 2.956) | −6.984 (−10.313, −2.896) |
| TSA16 mixed | −6.411 (−13.187, −1.280) | −4.867 (−10.836, −0.247) | −4.345 (−8.493, −0.416) |
| TSA8 four-phase average | −12.435 (−17.772, −8.203) | −11.193 (−17.054, −6.688) | −18.347 (−22.292, −14.287) |

Matched field-preserving controls give direct evidence of inference overhead. A single discarded forward behind TSA8-reference costs about 4.65/3.60/4.03% at N64/N128/N256; four discarded forwards cost about 18.48/18.18/16.25%. A zero proposal after one forward costs about 2.60/2.56/2.09% relative to TSA16-plain. These are empirical paired comparisons, not a causal decomposition of every timing difference. Initialization, reference calls, model calls, projection, operator applications and final true-residual checks are included in warm solve latency; physical/base setup and resident model loading are reported separately.

## What the direction and phase evidence establishes

The mixed checkpoint has genuine useful-direction phase sensitivity. After undoing a quarter-turn input phase, its mean raw proposal defect is 0.176/0.183/0.199 behind TSA8 and 0.285/0.294/0.310 behind TSA16. Projection out of the old **real** image span does not remove it: mean real-line angles to the phase-zero useful image are 23.29°/27.03°/32.38° and 47.31°/50.69°/55.81°, respectively. The angle already ignores positive or negative real rescaling. This establishes a search-direction change, not merely contamination inside old `AZ`.

Four-phase averaging improves the mean squared residual ratio after the proposal and two further classical applications. Behind TSA8, phase-zero ratios 0.002501/0.002413/0.002482 become 0.002249/0.002239/0.002292. At N256 behind TSA16, 0.001163 becomes 0.000813. Those finite-horizon gains do not establish tolerance-dependent solve savings. The separately timed four-forward intervention loses against classical at every grid.

**The saved additivity/homogeneity fields describe the original single-forward model, not the four-phase average.** Although stored in `phase4_average` rows, their `phase0_*` names are accurate: `diagnose_tail_directions.py:177–186` calls `model_proposal` with its default single phase for these probes. Mixed-model mean useful-image additivity defects are 0.02079/0.02610/0.02933 behind TSA8; factor-two positive-homogeneity defects are zero. A zero scaling defect on these probes does not imply additivity or complex linearity. Four-phase averaging enforces quarter-turn equivariance in exact arithmetic; neither arbitrary-phase equivariance nor complex linearity follows. The data establish a geometric limitation but do **not** establish that enforcing complex linearity would remove the runtime gap: no constrained architecture was trained or tested here.

The rejection semantic discrepancy is independently demonstrated and correctly isolated. All 58 deliberate zero directions give the original masked ratios `[1, 1, 1]`, while corrected fallback continues the original Arnoldi frontier and reduces the final squared ratio (range `2.18e-5` to `0.06722`). Because actual learned proposals never reject in this cohort, this repair cannot explain their measured slowdown. It remains a valid correction for offline diagnostics of rejected proposals.

Classical-frontier proposals remain a strong geometric control: their mean immediate squared residual ratios are approximately 0.083–0.088 behind TSA8 and 0.034–0.035 behind TSA16, compared with 0.485–0.510 and 0.668–0.754 for mixed phase-zero proposals. Generating that classical proposal consumes a classical application; it cannot be presented as a free oracle. Combined with the timed field-preserving controls, the evidence supports insufficient saved classical work to repay inference for the tested mechanism.

All inference is limited by one paired training seed, the repeatedly reused 16 development media, this finite-horizon diagnostic, and the periodic discrete RTE setting. The result is a supported development decision and a preserved negative result, not held-out, OOD, continuous-RTE or universal neural-preconditioning evidence.

## Supplementary eligible-branch profiles — independently checked 2026-10-02

`eligible_profiles/settings.json` and `profiles_n*.json` contain **60 complete profiled solves**, all successful with finite stage times and true residuals at most `9.81638902305378e-6`. Each of ten methods has exactly two profiles per grid. IDs match the first two saved eligible contexts, without timing selection: TSA8 `[0,1]`, TSA16 `[0,3]` at each resolution. All injection branches execute; six deliberate no-op proposals reject, while all learned/reference proposals accept. The script hash and mixed checkpoint identity match the recorded execution. The script's internal wall time is 16.625 seconds.

The table gives **mean milliseconds per complete profiled solve**, averaging the two selected IDs. `B` is classical preconditioning; `A` is the separate Arnoldi/proposal action stage; `LS` includes projections and small least-squares work; `check` includes true-residual checks and their operator work. `ref/model` are **nested sub-times already inside callback**. Do not add them to callback or interpret any row as uninstrumented benchmark latency. Profile wall also includes work/bookkeeping outside the timed stage blocks.

| N | Method | B | A | LS | check | callback | ref/model | profile wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|64|tsa8_reference|134.24|8.25|48.04|64.34|0.49|0.27/0.00|290.00|
|64|tsa8_reference_cost|129.17|8.00|14.19|11.43|250.14|0.26/249.65|416.63|
|64|tsa8_reference_phase4_cost|129.13|7.92|14.17|11.41|30.63|0.26/30.11|196.88|
|64|tsa8_mixed|122.34|7.39|13.30|10.95|8.13|0.27/7.63|165.67|
|64|tsa8_phase4|124.87|7.65|13.95|11.21|30.94|0.26/30.44|193.26|
|64|tsa16_plain|177.87|5.01|7.61|8.04|0.00|0.00/0.00|200.88|
|64|tsa16_noop_cost|176.92|4.78|6.93|7.86|8.49|0.26/7.97|207.54|
|64|tsa16_reference|163.74|5.30|8.34|8.39|0.48|0.27/0.00|189.17|
|64|tsa16_reference_cost|164.17|5.41|8.72|8.36|8.23|0.26/7.75|197.97|
|64|tsa16_mixed|177.99|5.80|9.56|8.89|8.17|0.29/7.66|213.51|
|128|tsa8_reference|131.69|8.14|18.74|12.71|0.48|0.27/0.00|175.47|
|128|tsa8_reference_cost|131.40|8.20|16.41|11.41|25.46|0.28/24.94|196.68|
|128|tsa8_reference_phase4_cost|131.58|8.27|16.40|11.48|31.26|0.28/30.74|202.74|
|128|tsa8_mixed|124.17|7.69|14.99|10.91|8.50|0.26/8.01|169.78|
|128|tsa8_phase4|125.88|8.52|15.49|11.17|30.50|0.28/29.99|195.18|
|128|tsa16_plain|165.78|4.47|6.79|7.44|0.00|0.00/0.00|186.42|
|128|tsa16_noop_cost|165.34|4.40|6.59|7.29|8.16|0.26/7.65|194.01|
|128|tsa16_reference|165.45|5.21|8.52|8.13|0.47|0.26/0.00|190.54|
|128|tsa16_reference_cost|165.59|5.22|8.58|8.19|8.01|0.26/7.55|198.28|
|128|tsa16_mixed|178.97|5.58|9.29|8.57|7.79|0.27/7.32|213.08|
|256|tsa8_reference|133.87|13.79|25.07|20.52|0.60|0.36/0.00|197.88|
|256|tsa8_reference_cost|134.65|13.52|24.13|18.83|21.16|0.34/20.57|216.23|
|256|tsa8_reference_phase4_cost|133.53|13.44|24.19|18.75|30.54|0.34/29.94|224.42|
|256|tsa8_mixed|126.52|12.81|22.70|18.01|8.02|0.35/7.42|191.86|
|256|tsa8_phase4|126.50|12.79|22.76|18.06|30.36|0.35/29.76|214.29|
|256|tsa16_plain|168.85|7.71|9.64|12.39|0.00|0.00/0.00|200.85|
|256|tsa16_noop_cost|169.06|7.68|9.58|12.42|8.03|0.34/7.41|209.36|
|256|tsa16_reference|170.69|8.97|12.60|13.83|0.59|0.35/0.00|209.79|
|256|tsa16_reference_cost|168.93|8.97|12.55|13.83|9.11|0.35/8.50|216.52|
|256|tsa16_mixed|182.89|9.66|14.12|14.52|8.02|0.34/7.43|232.43|

This sequential profile run has no dedicated per-method warmups. Its first model call at N64/N128/N256 takes **491.28/41.57/33.70 ms**, versus 8.03/8.31/7.43 ms for the second corresponding call. The first N64 reference profile also has unusually large LS/check times. The table retains these observations; ordering and initialization effects prevent treating the cost-control profile means as steady inference overhead. The uninstrumented, randomized, warmed benchmark above remains the runtime evidence.

On these two eligible cases, classical preconditioning is usually the largest recorded stage. Subsequent mixed-model forwards take about **7.3–8.0 ms**, reference sub-calls about **0.26–0.36 ms**, and four-phase model blocks about **30 ms**. TSA8-mixed uses a mean nine classical calls versus 9.5 for TSA8-reference; TSA16-mixed uses 6.5 versus six for TSA16-reference. Thus the small profile cohort illustrates modest TSA8 work savings and no TSA16 work savings, while inference and projection/check work still cost time. It cannot estimate population stage fractions, explain every benchmark difference, or change the gate decision.

I also checked all **36 unique classical phase records**: three grids × two prefixes × two selected IDs × the phases `i,-1,-i`. Maximum relative covariance defects are `8.9033e-15` for the physical action, `2.5572e-6` for FP32 classical preconditioning, and `3.6748e-8` for homogeneous reference action. These tests use the physical source, rather than the hard tail residual, and are not 36 independent media. They are consistent with the intended arithmetic and much smaller than the neural phase defects. They establish empirical quarter-turn covariance only. General complex linearity requires a structural argument from fixed material-dependent matrices and a fixed sequence of linear updates; it is not proved by these probes, and floating-point implementations are not exactly linear.
