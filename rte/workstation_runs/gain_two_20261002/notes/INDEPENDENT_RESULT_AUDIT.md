# Independent β2 result audit — 2026-10-02

The single prescribed control is numerically valid and fails both cost gates. All eight measured directions improve over gain1, but the complete solver remains slower than the prior-ten comparison envelope. This closes the tested fixed-gain/placement control; it does not establish that arbitrary gains, other correction ranges, or learned maps cannot work.

I computed [the independent audit](INDEPENDENT_RESULT_AUDIT.json) directly from [raw receipts](../../receipts/gain_two_execution/) before consulting the root aggregation. Its primary, warm, worst-case, research-charged, geometry and accuracy statistics agree exactly with `RESULT_AUDIT.json`. The fixed policy SHA is `7adf90dd59c439d5b17e3b47513ebdbaaf1cf3f7b3a5f31d75e44c977e912832`; driver SHA is `040d21a413f6e14a118f6193fde7fa09a359489c41e0d13551cceb28138553cf`. All twelve code hashes and fifteen staged payload hashes match. There was one candidate, no gain sweep, and the inherited RAW/N128 and direct-segmented/N256 mapping remained fixed.

The receipt contains 280 entry checks (240 actions, 40 guard bundles), all before 176 full solves (132 timed, 44 warmups), 220 measured applications, four physical prefixes and eight geometry targets. All full solves pass finite-field, original physical-residual and computed discrete error-bound acceptance. No full solver failures or rejected injections occurred. Ninety-six same-recipe full-solve pairs pass their required parity checks. The four four-step diagnostic prefixes intentionally stop before convergence and are not full-solve failures. Minimum coercivity margin is 0.0170082; maximum relative residual is 9.8222e−6 and maximum evaluated relative solution-error bound is 7.2498e−5. These are ordinary floating-point discrete diagnostics, not interval-certified or continuous-RTE bounds.

Costs use the median of three timed solves within each context, then the mean over IDs 0/3. Negative savings denote a slowdown.

| Grid / fixed backend | Prior-ten Q1 → β2 Q1, ms | Q1 saving | Warm saving | Research-charged Q1 saving |
|---|---:|---:|---:|---:|
| 128 / RAW | 281.474 → 293.142 | −4.145% | −40.332% | −48.133% |
| 256 / direct segmented | 164.934 → 219.274 | −32.946% | −45.161% | −133.096% |

The unchanged requirement is at least 10% mean Q1 **and** warm savings on both grids, with no individual Q1 regression above 10%. Worst Q1 regressions are 35.331% and 38.227%; every aggregate gate fails. Q1 and warm denominators separately minimize eligible prior-ten arms per context. These are comparison envelopes, not a free deployable method selector. Even against gain1 on the same backend, β2 saves only 6.543%/9.049% warm time. Classical calls improve from 20/13 to 19/12 at N128 and 21/14 to 19/13 at N256, but the strong direct-segmented recipe needs only 12/8 and 7/5 calls.

The stricter sensitivity adds each unique prefix-init/solve/verify charge once plus both geometry-target totals to β2 Q1; nested timers are not counted twice. This costs 0.577991 s across four contexts. The previous 0.612794 s acquisition remains historical provenance. Current mean setup held fixed, zero additional acquisition/inference/training cost assumed, and both aggregate 10% conditions enforced, the necessary whole-solve warm ceilings are 36.213/116.626 ms: further cuts of 52.369%/38.000% from observed β2 warm times. Research-charged ceilings are already negative, −87.601/−47.908 ms. These are conditional accounting ceilings, not hardware floors, attainable-oracle budgets, or range impossibility theorems.

All eight unit and real best-line geometry gains are positive. Mean squared best-line improvements over gain1 are 0.01094/0.01644 for N128 source/prefix and 0.01104/0.02287 for N256. This supports a useful omitted-angular input weighting on these targets. It does not identify optimal coefficients, prove full-solve progress from every residual, or isolate angular range from heterogeneity, low/high-operator differences and rounding. The accepted action is the literal `g + R0(2d − Pi3 d)`; its original-H image differs from the reassociated `2H(p_full) − H(p_q3)` by up to 2.1128e−4 relatively. Exact-arithmetic equivalence must not replace measured floating-point action. Zero actual rejections gives no basis to attribute this slowdown to the historical rejected-frontier loss defect.

First-use charges are retained: N128/ID0 physical setup 262.446 ms, margin evaluation 78.892 ms and strong-base setup 73.437 ms versus TSA4's 3.611 ms. Setup-plus-warm ranking is not independently measured process-cold ranking. Only two development media were used; nested grids and timing repeats are not extra independent samples. Owned execution completed in 70.936 s (runner 67.296 s), below 120 s hard/95 s soft limits, with 10,215,227,392 reserved bytes below 12 GiB. [Postpreservation](../../receipts/gain_two_final_preservation.json) verifies 408 hashes, all 155 original sources and nine dirty tracked files unchanged, stable main/HEAD, and no other research jobs.

No training, new teachers, held-out seed 9626, wider gain search or automatic larger confirmation is justified by this failed screen. A future route would need a genuinely cheaper correction representation or stronger cheap complement, with a prespecified complete-cost oracle and original-H acceptance before coefficient acquisition. The current evidence does not supply that route.

Recompute the independent raw audit with:

```sh
python3 deliverables/gain_two/audit_gain_two_receipts.py receipts/gain_two_execution \
  --output deliverables/gain_two/INDEPENDENT_RESULT_AUDIT.json \
  --expected-policy-sha 7adf90dd59c439d5b17e3b47513ebdbaaf1cf3f7b3a5f31d75e44c977e912832 \
  --expected-driver-sha 040d21a413f6e14a118f6193fde7fa09a359489c41e0d13551cceb28138553cf
```
