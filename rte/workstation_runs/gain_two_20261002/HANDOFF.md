# FWI RTE fixed gain-two screen — 2026-10-02

The sole prespecified gain-two control improves actual physical correction geometry and slightly reduces continuation work, but **fails the unchanged 10% complete-cost gate on both grids**. Primary Q1 is 4.145% slower at N128 and 32.946% slower at N256; warm accepted solves are 40.332% and 45.161% slower than the independently best eligible prior controls. Charging this round's unique diagnostic acquisition makes the failure larger. All numerical/parity checks pass. Preserve this negative and stop this fixed-gain/terminal-composition family; no training or further gain sweep follows.

## Exact identity and authorized scope

SSH used the existing alias `FWI`, noninteractive existing credentials, strict known-host checking, disabled configured forwarding and a bounded connection timeout. Verified hostname `FWI-Standard-PC-i440FX-PIIX-1996`, user `wjt`, real repository `/data/wjt/NeuralPreconditionedBornSeries`, RTE subtree `/data/wjt/NeuralPreconditionedBornSeries/rte`, branch `main`, HEAD `028caea3f35d100225f656e2b1e910c1366aaafe`, origin `git@github.com:wangjt23/NeuralPreconditionedBornSeries.git`. This is the RTE (radiative transfer equation) project; the separately paused OL_preconditioner project was untouched.

All substantive numerical work ran on FWI's RTX A6000, 49,140 MiB, using `/data1/software/miniconda3/bin/python` (Python3.12.4, Torch2.5.1/CUDA12.4). Mac work was CPU fixtures, receipt analysis and private artifact handling. No unrelated user research process or GPU compute job was present before launch or after completion; the 8 MiB desktop integration process was left alone. No original source bytes were newly exported, Git writes made, jobs cancelled, packages installed, network/proxy/forwarding/SSH settings changed, paid compute used, or training run. Seed9626 remains sealed.

New isolated remote run: `/data/wjt/NeuralPreconditionedBornSeries/rte/workstation_runs/gain_two_20261002`. Final report is its `HANDOFF.md` and `docs/research/RTE_GAIN_TWO_SCREEN_2026_10_02.md`. Earlier runs and handoffs remain unchanged.

## Prespecified control and mechanism

The exact input-dtype action is `g=TSA4(r)`, `d=r-low.A_op(g)`, `p2=g+R0(2*d-Pi3(d))`. Pi3 projects input angular-index DFT modes0,+1,-1; R0 is the original prepared complex64 full twelve-angle homogeneous-reference inverse and produces all output angles. The captured prefix is cast back before forming the defect/addition, and final solver fields own complex128 storage. Native128 promotes the same inverse/mask, with no rebuild. Independent parity uses the original angular projector and original reference solve. Reassociation as `2*p_full-p_q3` is not the implemented FP32 path.

For nominal fixed exact operators, let H_l be the low stored operator, G its mean streaming inverse, F=G+R0(I-H_l G), and S=I-H_l F=(I-H_l R0)(I-H_l G). The four stationary residual stages satisfy M4=F(I+S+S²+S³), I-H_l M4=S⁴. TSA4 is not F⁴. With C2=2I-Pi3, M2=M4+R0 C2 S⁴ and

    I-H_l M2=(I-H_l R0 C2) S⁴,
    I-H_h M2=(I-H_l R0 C2) S⁴-(H_h-H_l) M2.

Factor order matters: streaming couples angular responses, so Pi3 and R0 generally do not commute. These nominal identities do not remove FP32 rounding or promote the low action into physical H_h. In an ideal matched homogeneous inverse, an arbitrary tentative g leaves `(1-beta)(I-Pi3)d`; gain1 closes that defect while gain2 reverses it. For the ideal TSA primitive itself F=R0 and its residual is already zero. Thus beta2 is an empirical heterogeneity/control hypothesis, not a contraction theorem.

Relative to TSA4, the terminal action adds a low stored-operator defect and one full R0. Gain2 additionally has one angular FFT/ifft pair and local multiply/subtract; it does not use two R0 applications. Verified counts are17 spatial forward/inverse pairs plus one angular pair, versus TSA4's15 spatial pairs. These counts exclude copies/guards and do not establish latency or a reduced-channel implementation. Full/q3 beta1 use the same underlying cached inverse.

The user froze beta=2 as the sole gain, with RAW at N128 and explicit direct-segmented TSA4 at N256, before results. There was no tuning, backend reselection, q expansion, span fit, teacher, H0 rebuild or new oracle. Protocol hash `7adf90dd59c439d5b17e3b47513ebdbaaf1cf3f7b3a5f31d75e44c977e912832`; driver hash `040d21a413f6e14a118f6193fde7fa09a359489c41e0d13551cceb28138553cf`; helper hash `6827ebc351253e32ec075abebd63b9bd4fd03b82ebcf454dd51bbabc6a86bafc`. All12 executable source files and15 staged payloads were verified against their frozen hashes.

## Cohort, controls, correctness and resource use

Existing exploratory development seed4626, IDs0/3 only at N128/N256. Existing parent data hashes and regenerated source/configuration equality were checked. This cohort is reused development data, not independent validation, OOD evidence or sealed test data. The two IDs are the units of comparison; repeated timing rows are not independent media and yield no population confidence claim.

The prior portfolio is eager, context-full, context-segmented, explicit-full, explicit-segmented, TSA4, full-segmented, q3-segmented, full-raw and q3-raw; gain2 is the only added arm. N128 retains TSA8/reference-tail balanced initialization and the original full-reference injection after four classical calls. N256 retains TSA16/zero-start. Same reference cache, source and N128 starting field were explicitly checked bitwise between strong/shallow recipes. Physical H_h, original real-stacked coefficient convention, restart25, max100, input precision and source-normalized stopping are unchanged.

- 19 CPU preflight tests passed on FWI in1.118 s (preflight process4.546 s); meaningful tests include literal precision/backcast, fixed gain/mode/mask/cache guards, retained output storage, capture errors and operation counts.
- 280 GPU checks passed before any complete solve:240 own-literal action probes in both precisions and40 ownership/cache/gain guard bundles. Four source/start/cache QA bundles passed. Different algorithms need accuracy agreement; only same recipes require bitwise field/status equality.
- 176 complete solves (44 warmups/132 timed) and96 same-recipe field/status pairs passed, plus220 measured applications,4 truncated physical-prefix diagnostics and8 actual-H geometry targets. No complete failure or rejected tail injection occurred.
- Largest independent source-normalized physical residual `9.82219429208e-06`, evaluated relative-error bound `7.24980613462e-05`, and paired final field difference `3.95119542932e-07`. Acceptance remains residual≤1e-5, bound≤1e-3, field difference≤1e-4, finite field, zero failures.
- Minimum evaluated finite-discrete coercivity margin `0.0170082185985`. Bound uses delta=||q-H_h x||/margin, then delta/(||x||-delta) when its denominator is positive. It is an ordinary FP64 evaluation for the stored discrete operator, not an interval/continuum certificate.
- Runner67.296 s; owned process70.936 s, versus95 s soft/120 s hard limits. Peak allocated2,725,646,336 B and reserved10,215,227,392 B, within12 GiB. No deadline or memory limit was relaxed.

## Full-cost result

All deployable costs charge material/coercivity setup, appropriate prepared base/inverse, whole mask/action/graph constructor and warmup, casts/copies, original complete solver, accepted/rejected work, and independent residual/error/finite acceptance. Q1 is recorded setup plus accepted warm solve. Source IO, interpreter startup and comparative QA are research process overhead, excluded from primary Q1; it is not a matched process-cold estimate.

Within-medium medians of three timed repeats are averaged across the two media. Prior-ten eligible Q1 and warm costs are minimized independently per case. These envelopes are analysis denominators, not free deployed selectors. Require≥10% improvement in both mean Q1 and warm at BOTH grids, with no individual Q1 regression>10%. Candidate eligibility includes warmups. No candidate selection occurred after outcomes.

| Grid | Fixed backend | Gain2 / prior Q1 (ms) | Gain2 / prior warm (ms) | Q1 / warm saving | Research-charged Q1 saving | Worst individual Q1 regression |
|---|---|---:|---:|---:|---:|---:|
| 128 | raw | 293.142 / 281.474 | 76.028 / 54.177 | -4.145% / -40.332% | -48.133% | 35.331% |
| 256 | direct_segmented | 219.274 / 164.934 | 188.107 / 129.585 | -32.946% / -45.161% | -133.096% | 38.227% |

Every primary and stricter research-charged gate is false. Gain2 is6.543%/9.049% faster than its same-backend full beta1 action, and19.922%/20.505% faster than same-backend q3, at N128/N256. Those are weaker-comparator improvements; they do not pass the strong portfolio gate. Strong explicit-segmented calls are12/8 at N128 and7/5 at N256. Gain2 still needs19/12 and19/13, versus full beta1's20/13 and21/14 respectively. At N128 reported iterations20/13 include the retained reference injection; classical calls19/12 count only preconditioning.

Unique new diagnostic acquisition totals `0.577990965918` s: each prefix initialization+solve+independent verification once, plus both geometry target totals per material. Nested image-acquisition fields are not added again. The stricter prespecified sensitivity adds each material's entire unique acquisition to gain2 Q1 without amortization; warm costs stay unchanged. Research-charged Q1 is48.133%/133.096% slower. This is an explicit conservative research-cost sensitivity, not required online geometry for the fixed user-prescribed map or a hardware floor. Earlier matched-v2 acquisition0.612794369459 s remains historical provenance and was not erased or counted twice.

First-use order remains material: N128/ID0 physical construction262.446 ms, margin/eigensetup78.892 ms, strong base73.437 ms versus shallow3.611 ms. Later contexts are much cheaper. These are recorded setup timings with order/library-startup effects, not intrinsic algorithm costs or matched fresh-process measurements. The large warm failure remains without relying on that setup confound.

## Useful geometry, insufficient affordable continuation

All eight actual gain2 fields were separately passed through original physical H_h and compared bitwise to their own literal. Line fits use the same real-AZ projector and physical target; source and prefix targets remain distinct. All eight unit and best-real-line gains over full beta1 are positive. Stratum means are:

| Grid | Target | q3 line remainder | Full line remainder | Gain2 line remainder | Squared-line gain over full |
|---|---|---:|---:|---:|---:|
| 128 | source | 0.465143 | 0.431538 | 0.417836 | 0.010939 |
| 128 | prefix4 | 0.568144 | 0.536384 | 0.520157 | 0.016444 |
| 256 | source | 0.465965 | 0.432462 | 0.418675 | 0.011044 |
| 256 | prefix4 | 0.459971 | 0.407499 | 0.378757 | 0.022866 |

The measured-image identity H(p2)≈2H(p_full)-H(p_q3) has relative discrepancies up to2.1127654e-4 because those FP32 computations differ; actual H(p2), rather than that reassociation, supplies every geometry result. Maximum independently recomputed prefix-residual gap6.9971e-11 and real-AZ Gram defect7.3369e-16 support the recorded projector only. No independently optimized full-response range, coefficient teacher or deployable fitted field is established. A better diagnostic direction does not guarantee cheaper solver trajectories.

The concrete economic obstruction is the observed total continuation cost against stronger prepared classical factors. With the **current gain2 setup**, zero new acquisition/learning cost and an unchanged10% gate, necessary accepted-whole-solve ceilings are:

- N128: with recorded mean setup 217.114 ms, the necessary entire accepted-warm ceiling is 36.213 ms, versus observed 76.028 ms. That needs a further 52.369% warm reduction even with zero new learning/acquisition cost. The stricter research-charged ceiling is -87.601 ms: charged setup/acquisition already exceeds the mean Q1 allowance before any solve.
- N256: with recorded mean setup 31.167 ms, the necessary entire accepted-warm ceiling is 116.626 ms, versus observed 188.107 ms. That needs a further 38.000% warm reduction even with zero new learning/acquisition cost. The stricter research-charged ceiling is -47.908 ms: charged setup/acquisition already exceeds the mean Q1 allowance before any solve.

These are conditional accounting ceilings, not proved response-range impossibility, an achievable proposal budget, a hardware time floor or a license to train. The current beta2 already has no coefficient-acquisition cost; making its prescribed coefficient free changes none of its observed warm-cost failure. Other coefficient fields could alter trajectories and were not tested. No claim follows about all gains, compression representations, coefficient fields, nonlinear learned maps or every future material/source.

## Step back: ranked directions and stop boundary

1. **Most promising immediate missing-control question: complex-coefficient classical continuation with the already prepared strong transport factors.** The project already contains Hermitian complex FGMRES, whereas these portfolio runs use its real-stacked convention. For a fixed complex-linear action in unrestarted exact arithmetic, the complex coefficient span contains the real span, so residual minimization cannot be worse at the same degree. That is a conditional search-space fact, not a promise of fewer calls/time under restarts, flexible actions, reference injection or rounding. This control has low acquisition cost and would refresh the classical denominator before any broad learned-superiority claim; it is not novel research by itself.
2. A cheaper all-angle fixed-material response representation would attack acquisition/application cost directly. It needs a specified setup/action budget and independent actual-H/accepted-cost witness before teachers or training. Existing line/two-image fits and prior response negatives do not show an optimized-range bound or an affordable representation.
3. Further arbitrary angular/gain expansion or compression training has weak support after the preserved cost negatives. Do not run it on geometry alone.

The complex-coefficient question is a principled proposed next discriminator, **not run in this round**. Real coefficients do not inherently break global phase equivariance. Real-valued sources may offer little imaginary freedom; the actual even-grid Nyquist convention prevents assuming an invariant real subspace. A correct future diagnostic must retain actual solution directions Z, use unprojected W=H_h Z and compare min_R||r-Wa|| with min_C||r-Wc||, equivalently the real span[W,iW], then physically verify the combined field. The existing real-AZ projector generally does not commute with i; fitting one complex scalar to an already real-projected image is not this test. The separate FALLBACK_REVIEW supplies the proof limits and a32-complete-solve proposal retaining the N128 reference-tail and N256 zero recipes, full costs and current gates. Its implementation, confirmation/OOD evidence and any new compute remain outside this sole-control authorization.

## Preservation, independent review and deliverables

Post-run verification checked408 file hashes, including155 original source files, all prior negative runs and the new26-payload run. Original nine tracked dirty paths and HEAD/main were unchanged, with no mismatches or other research jobs. The verification itself is saved as `gain_two_20261002/final_preservation.json`; subsequent new report writes have separate hash readback. Preservation uses hashes only, with no original bytes in this packet.

The independent standard-library auditor reproduces counts, fixed policy/code identities, cost/accuracy arithmetic, own-recipe parity, source/start QA, unique acquisition and all gate failures from these new JSON/code receipts without original source or GPU access. It agrees with the separately written root analysis. Review files are INDEPENDENT_PROTOCOL_AUDIT, INDEPENDENT_RESULT_AUDIT, MECHANISM_AND_STOP_RULE and FALLBACK_REVIEW. The compact private Library bundle includes all26 new run payloads, new analysis/auditors/handoff/preservation receipts and compact prior negative evidence; a SHA256 payload manifest verifies each entry. No original source, data arrays, trained weights, credentials or sealed test bytes are packaged.

Earlier preserved direct-capture16-medium confirmation remains a classical20.784%/19.645% Q1 improvement. Prior learned/runtime, algebraic precision, q3, full-response, full-angle and v1 capture failures remain available. Their unchanged private artifacts are [direct-capture report](https://chatgpt.com/api/library/files/libfile_3a3dbf9299f48191985aecdb19ae0cd9/download), [direct-capture bundle](https://chatgpt.com/api/library/files/libfile_bd99ac07b6688191a8c525c34b928732/download), [matched full/q3 report](https://chatgpt.com/api/library/files/libfile_efe562b9c31c8191b25a24f690448cd7/download) and [matched full/q3 bundle](https://chatgpt.com/api/library/files/libfile_9be8be2b807c819192509db249a3f347/download).

Remaining gaps are independent/OOD confirmation, matched process-cold or genuinely declared multi-RHS economics, any affordable learned representation, and the complex-versus-real missing-control question. They do not block this completed negative screen. No training, seed9626 access, original-source export, Git mutation, broad campaign or another scalar tweak occurred.
