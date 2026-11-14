# Independent complex-Krylov entry audit — 2026-10-02

The prespecified entry rule stopped this run correctly. All 112 capture/ownership checks pass, but eight of 108 admissibility records fail the frozen low-H FP32 cap. No complete solver comparison was admitted: **zero full solves, timed solves or warmups**. This is a numerical consistency-cap obstruction, not a measured complex-FGMRES convergence failure or a failed cost-benefit comparison.

The [independent standard-library audit](INDEPENDENT_RESULT_AUDIT.json), computed from [raw receipts](../../receipts/complex_execution/) without Torch/original imports or GPU work, returns `protocol_passed=true`, no protocol errors, and `outcome=completed_entry_obstruction`. It verifies frozen policy `acd73134c63839726d3b699c60e795b8da90d14709a9529fb94e905ce59ebd93`, driver `efc7d6f95598e1a7b02c60c1740ab52660a2a48e19cc9d67e90f76564bacb3bf`, all nine code hashes and all thirteen staged payload hashes. Staging verified 420 existing files and CPU preflight exited successfully. This is staging preservation evidence; subsequent post-run preservation is a separate receipt.

The complete admission cohort comprises four contexts (IDs0/3 at N128/N256), 96 own-literal action checks plus sixteen guard bundles, and 108 records × ten calls = 1080 admissibility action evaluations. Source/start/shared-reference-cache QA passes in every context. All action outputs are finite, correctly typed/shaped, input-preserving and exact on zero; failures are solely normalized identity defects. There are 32 failed scalar identities within eight failed records, not eight independent new physical cases. The same source appears in source/mixed and grazing/source pairs.

| Action/path | Failed records / 12 | Maximum normalized defect | Frozen cap |
|---|---:|---:|---:|
| Original physical H128 | 0 | 9.7807e−15 | 1e−12 |
| Low H64 | 8 | 5.0361e−6 | 2e−6 |
| G64 / R064 | 0 / 0 | 2.6220e−7 / 7.6251e−7 | 5e−6 |
| Strong eager64 / actual backcast deployment | 0 / 0 | 1.5269e−5 / 1.5462e−5 | 1e−4 |
| Strong native128 / segmented native128 | 0 / 0 | 2.8651e−15 / 2.8651e−15 | 1e−12 |
| Physical H after actual deployment | 0 | 5.8973e−4 | 1e−3 |

Each context fails low-H on `source_mixed` and `nyquist_source`; all four random-complex low-H pairs pass, maximum 2.6279e−7. N128 source/mixed maxima are 2.2009e−6/2.4017e−6; N256 maxima are 4.7162e−6/5.0361e−6. At N256/ID3/source_mixed, the maximum witness is phase homogeneity of the mixed source: numerator 0.0158710386, noncancelling reference denominator 3151.43352, quotient 5.036133085e−6. Ratios use the frozen reference-output normalization; the diagnostic symmetric denominator was not used to rescue a failure.

The source-level nominal H is complex-linear on stored coefficients. The observed FP32 defects include native input scalar construction, FFT/action rounding and rounded output prediction; the receipts do not separately apportion those errors. High H and native128 factors agree to near double precision on these same probes, so there is no evidence here of a structural conjugation/nonlinear operation. Crucially, **low H is an auxiliary primitive inside the preconditioner; both the actual deployed map and its physical composite pass**. This neither proves nor disproves the safety or benefit of a future complex flexible solve. Passing finite probes is not global exact linearity; failing this engineering cap is not a theorem that complex Krylov is invalid. The unchanged no-waiver rule nevertheless requires zero full solves in this run.

The real source has zero imaginary norm. Its recorded physical-H imaginary fraction is nonzero (2.9266e−6 to 1.5583e−3), and the preconditioned-source fraction is 1.0251e−5 to 7.0275e−5. These describe actual complex fields, not useful coefficient freedom or continuous real-intensity validity. The selected grazing/Nyquist probe's H image has zero imaginary fraction in all contexts; that particular probe does not establish real-subspace invariance or a universal Nyquist effect. No solver coefficients, trajectories or matched search-space gains were obtained.

Unique mandatory entry acquisition costs 73.558860 s: 14.884354/16.049036 s at N128 and 20.235574/22.389896 s at N256. This includes the frozen probe calculations, metadata/norm work and in-timer receipt saves; it is research acquisition, not solver application latency. Runner84.434398 s and owned process87.909460 s stay within95 s soft/120 s hard; reserved memory2,009,071,616 bytes stays below12 GiB. Before/after GPU receipts show only the unchanged desktop process outside the owned run. The retained setup/prewarm costs cannot be combined with hypothetical warm solves to manufacture a Q1 saving. The result JSON therefore records the economic gate **unevaluated/null**, with no numerical solution-accuracy or sixty-four full-recipe parity results.

The frozen stop rule is satisfied, with no threshold relaxation, operator/cast change, training, new teacher, optional span fit, held-out seed9626, source export or stopped gain-variant rerun. The missing complex-solver economic evidence remains missing. Reopening would require a separately authorized, newly justified precision/error-budget or admission protocol before measurements; these receipts alone do not authorize weakening this cap or infer that every complex/learned route is impossible.

Reproduce the scalar audit:

```sh
python3 deliverables/complex_krylov/audit_complex_receipts.py receipts/complex_execution \
  --output deliverables/complex_krylov/INDEPENDENT_RESULT_AUDIT.json \
  --expected-policy-sha acd73134c63839726d3b699c60e795b8da90d14709a9529fb94e905ce59ebd93 \
  --expected-driver-sha efc7d6f95598e1a7b02c60c1740ab52660a2a48e19cc9d67e90f76564bacb3bf \
  --historical-root receipts/gain_two_execution \
  --historical-summary deliverables/gain_two/RESULT_AUDIT.json
```
