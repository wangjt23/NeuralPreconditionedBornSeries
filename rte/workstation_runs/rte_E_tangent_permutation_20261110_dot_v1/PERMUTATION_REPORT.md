# K: fixed transverse permutation, narrow CPU cost pass

## Outcome

The one predeclared N128 coefficient permutation passed its identity, actual-background, spectrum, original-operator and resource gates. E's complete four-source observed cost was **22.889414172 s**, compared with **23.834255143 s** for compact4, the cheapest accepted control: **3.9642% less time, only0.944841 s**. Native TSA16-zero cost25.066791111 s and TSA64-zero25.410193162 s. This is a narrow single-forward-order pass under the frozen decision rule, not a demonstrated robust performance margin.

The first source still favored native methods: TSA64-zero6.193036367 s, TSA16-zero6.285587403 s, E7.577831109 s and compact4 8.217916712 s. These are direct fresh-setup-to-first-accepted-source clocks, not totals divided by four. No extra source or free factor reuse was used to create the four-source saving.

More informative than the small time difference is the change in work. Against the sealed original N128 scene's E call pattern, outer classical calls increased from[2,2,2,2] to[3,3,3,3], while inner iterations decreased from[5,5,5,4] to[4,4,4,4]. Compact4 calls changed from[4,3,4,3] to[4,4,4,4]. Native TSA64 and TSA16 remained at5 and17 calls per source. Historical original-scene wall times are not treated as current paired timing controls.

## Prespecified transformation and actual identities

FROZEN_PLAN.md SHA2564b24e2d47d714a189c4b6e5ec6b2c4dae1b330409f8ae12fe3ee6c26f0b200d2 fixes N128, multiplier3, inverse43 modulo128, and the sole material transformation:

  total_new[i,j] = total_original[i,(3*j) mod128],
  scatter_new[i,j] = scatter_original[i,(3*j) mod128].

Only coefficients are reordered. Normal layering, stored angular geometry, original streaming and the four source arrays are unchanged. This is a new discrete material, not a continuous-PDE equivalence or conjugation of the entire operator. There was no scan over multipliers, widths, amplitudes, source order or seeds.

The original fixed layered-plus-Gaussian construction reproduced sealed E's material fingerprints. Identity permutation, inverse permutation, and every row's sorted coefficient multiset matched bit for bit for both total and scattering. New total SHA256a8973178e67bdcd78077ec66fe1701d00a7a78494c69d60cc916ecd50db3b5f2; scattering SHA256a78aba8699de073eeb18ff2fa040960e1b59563aa87b030078e5a10dd77931dd. Original total/scattering hashes remain in the full receipt and analysis.

Actual background_from_material was rerun. Its implementation uses total.min, scattering gather at the selected indices and squared-residual axis selection, not a mean approximation. The selected axis remained0. Both selected background arrays were exactly identical: maximum differences0 and unequal-element counts0. The selected score was16470.990234375 for both. Even the selected-axis gather indices matched. The nonselected-axis gather index hash changed, as recorded, although its gathered profiles and score remained equal. Nothing was silently rounded or forced back to an old profile.

The selected full-grid total background hash is2f3f2685c7b7e79bf0fd0f4721aeb10a7afda17a0fbfe5d65d69de752ee441c2; scattering ca41ae0109c4fc04e392ebe57a1f7955d753f6667efa430a49595d92119dbb45. Every fresh method construction reproduced these inputs, but paid for new objects and factors.

For actual coefficient deviations from this background, the tangent FFT identity F_new[k]=F_old[(43*k) mod128] held to relative1.95369e-16 for total and1.91622e-16 for scattering. Total spectral energy ratio was1.0. The fraction outside P8 changed from approximately **0.0015832% to20.40419%**. These percentages refer to coefficient deviations, not the full coefficient norm, solution spectrum, residual spectrum or a condition-number estimate. Modular aliasing is part of this finite-array identity and was not replaced by an ideal continuous spectrum.

Sources remain original IDs[0,4,7,14], unpermuted and exactly promoted from the original complex64 bank. All four known source hashes matched. Source bank SHA25658f8e2f069ea1fa1c0c8f490b441c7acfc91a6592d045a7a7a52eea7bf08aeb4. No saved neural solution, training, teacher, additional RHS, held-out9626 input or GPU was used.

## Unchanged E and numerical contract

E angular_modes.py remains SHA256d152afea90179efa8342a4f12631979951503785330f3aab08d4b7f1fe32a481, imported from its sealed directory. Production/dependency/E source hashes matched before and after. New driver run_permutation_gate.py SHA2564507fe32246b091cc30d7e76961d12f08557fdfbf6a4c39f44e111e22d762198 implements only the proposed material/input gate and driver orchestration. A pre-execution static review corrected an imported-name shadow before any numerical process; the numerical gate ran once without retry. The exact depth-call AST/stub test passed, including rejection of a deliberately inconsistent object. Actual runtime ClassicalTransportStack depths were64/64/16/64, read from real objects with expected128x128 reference and1x128x128x12x12 cache shapes.

Original A retains12 angles,g=.65, stored directions/weights/phase and normal Nyquist. E uses both signed tangent modes-8..8, original full-grid P8 A P8, inner tolerance1e-4/max8, outer tolerance1e-5/max100, zero initial guess and one enrichment after prefix1. The approximate angular representation is corrected against original G with tolerance1e-10 and at most3 repairs. Strict zeros return exact zeros; tiny nonzero columns use scaled relative norms without an absolute tolerance floor.

The disposable preliminary factor and fresh paid E factor each underwent all17 independent original-A0 checks and fixed zero/mixed/tiny boundaries. In total:34 mode records,12 boundary records,70 original-A0 vector checks; maximum residual3.860235557177974e-12. All zero/tiny cases passed. Runtime16 inverse applications covered272 mode inversions, all requiring exactly one correction. Maximum final original-G residual4.030526398294556e-12.

All16 method/source rows passed original-A residual1e-5, evaluated relative error bound1e-3 and field difference1e-4 against ordinary accepted TSA64 outputs. Across methods, maxima were8.614471257951949e-6,3.893999260264176e-5 and2.667691435633883e-7 respectively. E alone:2.0712065361098976e-6,8.998959373417117e-6 and7.620117096914809e-8. Maximum E inner original-projected residual7.744637684610383e-5 remained below1e-4. No threshold was relaxed.

## Complete cost ledger

| Method in fixed order | Fresh setup s | Direct first source s | Complete four sources s | Outer classical calls |
| --- | ---: | ---: | ---: | --- |
| TSA64-zero |0.102865687|6.193036367|25.410193162|5,5,5,5|
| compact4 |3.118904302|8.217916712|23.834255143|4,4,4,4|
| actual TSA16-zero |0.104751324|6.285587403|25.066791111|17,17,17,17|
| E |2.864019810|7.577831109|22.889414172|3,3,3,3|

Compact4 factor construction cost3.014989021 s. E factor construction cost1.025785965 s and its complete original-A0/boundary qualification1.508187017 s; both are in E setup2.864019810 s. E online solve plus checks totaled18.752235661 s, including3.689947765 s inner work. Compact4 online solve plus checks totaled20.323347657 s. These phase sums are explanatory; the decision uses the larger complete observed clocks including inter-source bookkeeping and saves. No validation is removed to make the small margin look larger.

Original outer/native A calls were48/52/144/44 for TSA64/compact/TSA16/E, with40 additional E inner A calls and4 final independent A checks for every method. Original-A0 qualification and G repair work are separate paid activities beyond these solver counters. Logical factor data: E276,633,928 bytes, compact4 302,137,344 bytes. These are logical counts, not isolated physical-allocation measurements.

Shared source/material/spectrum identity preparation cost1.791836774 s. Disposable preliminary E factor/check costs were1.263844243/1.419095042 s. These are separately visible in the full research wall; the disposable factor is not reused for a timed method. All four methods receive the same loaded source arrays and pay fresh material permutation, physical operator, native cache and any factor construction. Ordinary TSA64's25.410193162 s is both a paid control and the separately visible reference-output cost; those outputs are used only for acceptance checks, not as a free teacher, initial state or enrichment. Method costs are not interpreter/process-cold timings. No device transfer exists.

## Resources and release

CPU0, one numerical thread, CUDA disabled. Each stage was limited to90 s, total controller wall240 s and owned-worker RSS4 GiB. Actual stage walls were1.9319,3.1212,25.6671,24.2593,25.3639 and23.2910 s. Worker wall103.679157577 s, controller wall107.409437573 s. Peak worker RSS1,232,494,592 bytes. The shared single-process high-water reading does not establish method-specific peak memory.

All admission snapshots showed no other active numerical process, although normal services, SSH/SFTP and kernel activity were present. This was a shared server, not an isolated whole-host benchmark. The coordinated possibility of a Schur job is not described as actual overlap without evidence. Controller1289806 and worker1289808 exited normally, returncode0, reaped=true. At2026-11-10T14:04:13+08:00 explicit ps inspection found neither process. No reverse pair or second numerical attempt was started.

## Interpretation and stopping point

Changing transverse coefficient ordering while preserving verified background inputs changed the outer/inner call balance. More coefficient energy outside P8 did not make the inner solve harder; it finished in fewer iterations, while one extra outer classical call was needed for every source. This is consistent with work moving beyond the projected inner correction, but it does not identify a unique causal mechanism. Higher-frequency streaming and changed source alignment remain relevant explanations. Coefficient-energy coverage alone is not a validated predictor of convergence or cost.

The locked four-source cost rule passes narrowly, with first-source native methods still preferable in this observation. A3.96% single-order margin should not be promoted to robust dominance. This is one designed discrete material perturbation, not general-material performance, the sixteen-material Q1 benchmark, a continuous-PDE equivalence, or neural gain. The known much faster N128 GPU compact4 measurement remains separate deployment context; this CPU gate does not reopen G or I and supplies no new GPU result.

The authorized gate is complete. If confirmation becomes necessary, a separately reviewed bounded reverse-order comparison now has more value than declaring a robust timing win or expanding to more material choices. Do not repeat automatically, change P8/inner8, add sources, or remove qualification costs. The original and all prior sealed evidence remain unchanged.

Files: gate_01/results.json is the full raw receipt (SHA256794bf6a5271c23dfe35448dfc5b7c814aaa6e7265e65c50b4ae39e16671a08c7); WATCHDOG.json is the controller receipt; analyze_permutation.py independently audits saved evidence without Torch/solver imports, producing analysis.json. FROZEN_PLAN.md, SOURCE_AUDIT.json, depth-test evidence, launch hashes, admission/release evidence and this report are included in the lightweight archive. No large source/solution tensors are duplicated into it.
