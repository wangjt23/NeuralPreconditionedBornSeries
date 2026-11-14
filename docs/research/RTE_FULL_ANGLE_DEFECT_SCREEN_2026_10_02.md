# FWI RTE matched full-angle defect screen — 2026-10-02

## Outcome and scope

The bounded matched full-angle versus q3 reference-defect test completed on the authorized FWI host. All 252 entry checks and 160 complete solves passed; full input angles improved the frozen direction signal in every source/prefix stratum. Neither tested full nor q3 composition met the unchanged 10% Q1 **and** warm accepted cost goal at both grids. There is useful omitted-input signal for this composition, but no affordable learned or classical superiority witness. No training was started and seed9626 remains sealed.

This is a four-context exploratory screen (N128/N256, existing development IDs0/3), with two independent media per grid. The eight geometry targets include raw source and a four-step physical frontier for each medium, not eight independent media. Development seed4626 has been reused in earlier research. No new holdout or OOD generalization conclusion follows. The verified preceding direct segmented classical result remains separate: 640 solves/864 entry checks on16 development media per grid gave20.784% N128 and19.645% N256 mean recorded-Q1 savings. Its costs and original work are preserved; this round does not replace or pool that confirmation.

## Exact host and repository

- Access: existing SSH alias `FWI`, existing credentials; forwarding disabled for these connections and strict host-key checking retained. No access/security/network configuration changes.
- Actual host/user: `FWI-Standard-PC-i440FX-PIIX-1996` / `wjt`.
- Repository/realpath: `/data/wjt/NeuralPreconditionedBornSeries`; RTE subtree `rte`. RTE means **radiative transfer equation** in this project.
- Branch/HEAD: `main` / `028caea3f35d100225f656e2b1e910c1366aaafe`.
- Origin: `git@github.com:wangjt23/NeuralPreconditionedBornSeries.git`.
- Hardware/runtime: RTX A6000,49,140 MiB;36 CPU/184 GiB previously verified; `/data1/software/miniconda3/bin/python`, Python3.12.4, Torch2.5.1/CUDA12.4.
- New successful run: `rte/workstation_runs/matched_reference_v2_20261002`.
- Retained failed preflight: `rte/workstation_runs/matched_reference_20261002`.

Only FWI ran project numerical experiments. Mac CPU fixtures and artifact handling were auxiliary. The separately paused OL_preconditioner project was not resumed or modified. Original FWI source bytes remain dependencies on FWI and were **not exported** to the Mac or included in this artifact. The prior rejected optional source export remains unexecuted; it was not bypassed.

## Frozen question, arithmetic and comparisons

The tested maps use original FP32 arithmetic and the same prepared complex64 homogeneous inverse cache:

    g=TSA4(r), d=r-low.A_op(g)
    p_full=g+R0(d), p_q3=g+R0(Pi3 d).

Pi3 retains angular-index DFT modes0,+1,-1. It limits correction **input**, while R0 retains all12 output angles and all spatial frequencies. Uniform positive stored weights justify the DFT projector interpretation. The q3 implementation still executes the full inverse and adds an angular FFT pair; it is not a reduced-channel performance implementation. Full/native128 behavior promotes the stored inverse, without rebuilding an ideal or double inverse. A segmented prefix is cast back before the original defect and addition; solver outputs own complex128 storage.

Both grids used their original complete solver recipes: balanced reference initialization and reference-tail injection at N128; zero start at N256. Those common operations, and TSA4 itself, already carry full angular information. Five original strong backends (`eager`, context full/segmented capture, direct full/segmented capture), direct segmented TSA4, and full/q3 with segmented or whole raw capture were included. Different algorithms may use different work; identical-recipe backends must have identical fields and entire status.

Policy SHA256: `67e5cbdfaf734265f9b06ffb7f18cee75a6ab52c22cf8c5a46bce7cee0c30c48`. Driver SHA256: `c82c8050e93c48894f05c7791864ec8794237ea1497ae4862491f52a057dfe4d`. Helper SHA256: `6194942d23831bed3e720f194271acd279abb6176debeb5996067b267445b1f4`. The policy fixes3 timed repeats/1 warmup, development IDs0/3,95 s soft/120 s hard,12 GiB reserved,160 complete solves,4 truncated prefixes and8 geometry targets. It keeps residual1e-5, ordinary-evaluated relative bound1e-3 and paired field difference1e-4. One backend per correction rule/grid is chosen by two-media mean Q1 with deterministic ties; matched same-backend comparisons are also retained. No per-case algorithm was deployed by selecting a timing oracle.

## Actual repair and validation

V1 passed20 CPU tests but CUDA rejected the original projector's temporary-mask scalar assignment during q3 capture. It exited1 after4.698431890 s without timeout, before recorded entry checks or full solves; the GPU returned to desktop-only use. The failed code/policy/trace are retained. This was an implementation failure, supplying no angular or solver result.

V2 preconstructs the identical48-byte q3 mask outside capture, guards object/version/dtype/device/layout and holds it strongly, then uses the same angular FFT/multiply/IFFT and unchanged full-angle inverse. Mask allocation/filling moves into charged setup and is removed from each application. Native128 promotes those exact0/1 entries. No original function was patched or source edited. Independent static review found no blocking semantic, guard, precision or billing issue.

All25 v2 CPU tests passed on FWI (1.965 s tests;5.067 s preflight process). All252 global checks completed before any full solve:216 bitwise complex64/native128 action checks against independently evaluated **original eager recipes**, plus36 ownership/cache-view bundles. The raw captures retained17 spatial FFT/ifft calls each, with one extra angular FFT/ifft pair for q3; segmented capture retained its four-call stage counts. All96 complete-solver same-recipe field/status comparisons passed.200 measured applications,4 original-solver truncated prefixes and8 geometry targets were recorded. The4-step prefixes intentionally stop early and are not counted as failed complete solves.

All160 complete rows (120 measured,40 warmups) passed acceptance. Maximum source-normalized true residual: `9.82219429208e-06`; maximum evaluated discrete relative bound: `7.24980613462e-05`; maximum paired field difference: `3.63052044707e-07`. These are ordinary FP64 evaluations for the stored periodic FFT-pseudospectral finite operator, not directed-rounding, continuum or OOD certificates. The exact-arithmetic coercivity argument assumes the Hermitian weighted scattering margin is positive; actual margin and original-H residual are recorded per context.

Runner/process wall time: 61.918750/65.591788 s, no timeout. Peak allocated/reserved: 2,682,390,528/9,718,202,368 bytes, below12 GiB. Afterward, only the original desktop integration GPU process remained; no other user research job was found.

## What the geometry establishes

The actual promoted-buffer physical H scores the complete proposals against r. Increment diagnostics separately score reference corrections against recomputed physical defect dtrue. The captured real-AZ basis is checked, and projection uses two passes. The predeclared signal is `mean(q3_best_line_ratio²-full_best_line_ratio²)>=0.02` in **each** grid/context stratum; all four passed, and individual signs favor full in all8 targets.

| Grid | Context | Mean q3 best-line remainder | Mean full best-line remainder | Mean squared reduction |
|---|---|---:|---:|---:|
| 128 | source | 0.465143 | 0.431538 | 0.029906 |
| 128 | prefix4 | 0.568144 | 0.536384 | 0.034966 |
| 256 | source | 0.465965 | 0.432462 | 0.029881 |
| 256 | prefix4 | 0.459971 | 0.407499 | 0.045522 |

Full used21/14 iterations on IDs0/3 at both grids; q3 used24/16 at N128 and23/16 at N256; the strong original maps used13/9 and7/5. In matched raw capture, full reduced warm accepted cost relative to q3 by14.318%/12.697% at N128/N256. Its smaller per-application cost and useful input information did not compensate for greater complete-solver work relative to strong controls.

Recorded diagnostic qualifications: maximum low-versus-high defect gap `0.000643901186`; homogeneous full-cache closure `7.87308392e-06`; promoted correction/image additivity gaps `8.2121743e-06`/`8.32991416e-05`; prefix recurrence gap `6.99706893e-11`; real-AZ Gram defect `7.33687549e-16`. H0 is the FP32-assembled reference matrix promoted128: `E=H0R0-I` concerns cache/application closure, while `D=H-H0` includes heterogeneity **and** stored-operator differences. The measurements do not isolate pure heterogeneity or establish global operator norms.

The two-proposal real least-squares fit attained mean measured-image residuals0.413705/0.513217 for N128 source/prefix and0.414399/0.360312 for N256. It is an attained result in those two measured images, with all fits rank2. Its reconstructed field combination has **not** been independently passed through H. It is not a q3/full response-range optimum, optimal solver direction or deployable oracle. Prefix initialization/solve/verification, H0 construction once per context and geometry acquisition/fit are separately recorded: 0.612794 s total; nested component timers are not added twice. A future deployed oracle must pay its acquisition costs.

## Complete costs and negative boundary

Costs charge physical/coercivity setup, appropriate base preparation, the **entire** constructor (including warmups, mask, binding and shallow graph if required), casts/copies, complete solver including initialization/tail and rejected work, and independent acceptance. QA, source I/O and process startup are excluded. Q1 means recorded setup plus median warm accepted solve; it is **not a matched process-cold estimate**. Failed v1 engineering work and both CPU preflights are retained as separate research receipts, not silently erased.

Selected backends below use frozen per-rule/grid mean-Q1 selection. Q1 savings use the independently cheapest successful original-five Q1 per case; warm savings use the independently cheapest original-five warm accepted cost per case. Those can be different backends. Both thresholds remain10% at BOTH grids, with no individual Q1 regression over10%. Benchmark minimum envelopes do not assert a free deployed chooser.

| Grid | Selected rule/backend | Setup ms | Warm accepted ms | Q1 ms | Q1 saving | Warm saving | Worst Q1 regression |
|---|---|---:|---:|---:|---:|---:|---:|
| 128 | full_raw | 210.349 | 72.280 | 282.629 | +0.797% | -42.618% | 34.547% |
| 128 | q3_raw | 211.490 | 84.359 | 295.849 | -3.843% | -66.451% | 47.994% |
| 256 | full_segmented | 29.189 | 210.959 | 240.148 | -45.750% | -61.598% | 54.208% |
| 256 | q3_segmented | 29.394 | 237.730 | 267.124 | -62.122% | -82.106% | 67.982% |

Original-five best mean Q1/warm costs were284.900916/50.680837 ms at N128 and164.766683/130.545220 ms at N256. The strong Q1 winner was direct segmented in all4 contexts. All10 arms are accurate, so subsequent same-case classical comparison envelopes include them. At N128 ID0 full_raw is cheaper in recorded Q1, while ID3 regresses34.55%; it is not a two-grid or confirmed16-medium gain. N256 full_segmented is selected by only0.694 ms mean Q1 over full_raw, so no broad backend ranking follows.

First-use/order effects are explicit: N128 ID0 physical setup264.785 ms, coercivity eigensetup75.995 ms, strong-base preparation70.071 ms versus shallow3.400 ms. Retaining those charges does not establish intrinsic algorithm setup differences. The warm failure is substantial even against the Q1-winning strong map's own warm cost, and it survives this qualification. No process-cold or source-reuse amortization claim follows.

The obstruction is **measured fixed-composition progress and complete cost**, not a proved lack of full/q3 response range and not unbilled teacher acquisition. Current fixed maps use no teacher oracle. Angular truncation contributes to their observed loss, but merely restoring angles does not yield affordable superiority. A large discarded-input fraction or an invertible full reference inverse cannot establish a useful, cheap coefficient map for heterogeneous H. No requirement for residual history or benefit of training is proved.

For reference only, the future all10 comparison envelope gives these necessary **whole-solve** allowances with the selected full map's recorded setup and zero added acquisition/setup/inference/training cost:

| Grid | Selected full backend (held setup) | Maximum allowed whole warm accepted solve | Further reduction needed |
|---|---|---:|---:|
| 128 | full_raw | 32.667 ms | 39.613 ms (54.81%) |
| 256 | full_segmented | 117.491 ms | 93.468 ms (44.31%) |

These are cost-model necessities, not proposal budgets or achievable learned headroom. First-use effects and the strict per-case envelope qualify them; added costs lower the allowance. No affine subtraction of microbenchmark action time is used as an achievable runtime forecast.

## One precise next mechanism; no training yet

The development-informed next falsifier is a **fixed gain on omitted-input response**, once at beta2:

    p2 = g + R0(2*d - Pi3*d).

It retains the q3 input contribution and doubles its complement through one original full-angle inverse. This changes relative angular-channel weights; FGMRES already handles global real rescaling of a proposal. In all8 measured two-image fits, after normalizing `a*p_q3+b*p_full` by nonzero `(a+b)`, the omitted-input gain `b/(a+b)` was1.634857–4.774191. This supplies a finite, development-informed hypothesis, **not** a field-verified beta2 success, optimized-range proof, novel method or justification for a gain sweep/network.

Ranked next directions: (1) one teacher-free beta2 composition control, cheapest distinct direction falsifier; (2) economical heterogeneous coefficient/representation changes only if a charged positive progress budget first appears; (3) trajectory/history models remain lower priority because this evidence does not establish a history deficit or affordable acquisition. Existing classical full-angle/ROM methods remain comparators; adding learning alone is not a contribution.

Before beta2 GPU use, freeze its literal arithmetic/precision, original-H acceptance, same initialization/tail, all5 strong plus existing q3/full controls, identical masks/caches, whole constructor/solve costs, one fixed gain and modest95/120 s/12 GiB envelope after a fresh idle-job check. Use only existing IDs0/3, no new teacher bank, gain sweep or seed9626. It needs bitwise own-literal checks including native128 and changed-state guards. Continue to16-media confirmation only after BOTH cost gates survive; the sealed test and OOD regimes remain untouched in this round. If beta2 cannot pay, retain that bounded negative and stop this homogeneous-coefficient path before training. **Beta2 was not implemented or executed in this round.**

Classical precedent: [Fan–An–Ying angular reduction/homogeneous FFT](https://arxiv.org/abs/1802.03061); transport-ROM comparators [Peng ROMSAD](https://arxiv.org/html/2410.08735v3) and [Tang–Peng trajectory-aware ROM](https://arxiv.org/html/2509.05001v1). Their discretizations/setup assumptions require matching; no priority claim is made here.

## Preservation, artifacts and reproducibility

Post-run preservation checked370 files including155 original sources and all prior direct-capture/failed-v1 work: zero mismatches. HEAD/main/origin and all9 tracked dirty entries stayed unchanged. No original-source export/edit, Git commit/push/merge, new package, access change, new training or unrelated job cancellation. No active user research/GPU compute job remained.

The companion bundle contains only authored code/tests, frozen policies, new receipts, metadata/provenance and research reports. It excludes original source bytes, datasets, models, credentials, signed transfers and Library helper internals. Source/data identities and exact FWI dependency hashes are retained. The previously verified direct-capture Library artifacts remain unchanged; this round creates separate private report/bundle identities.

Key files: `receipts/matched_execution_v2/code/matched_reference.py`, `receipts/matched_execution_v2/code/matched_screen.py`, frozen `receipts/matched_execution_v2/policy.json`, `receipts/matched_execution_v2/results/{checks,setup,applications,geometry,prefixes,solver,settings}.json`, execution/staging receipts, independent result audit, denominator budget and final preservation. V1 failure receipts are under `receipts/matched_execution/`. CPU contracts can be run on FWI with original repo and v2 code on PYTHONPATH. Existing output directories deliberately reject overwrite. A numerical rerun requires a new isolated run directory, the same source/data hashes and frozen policy, and the bounded owned-process launch wrapper; do not directly rerun over either preserved run.

Remaining gaps: no affordable learned witness, no new trained checkpoint, no positive beta2 test, no full16 confirmation for new maps, no unseen holdout/OOD/source-reuse study, no continuum or interval certificate, and no entire correction-range optimum. Primary outcome is the bounded accepted negative with useful angular-information evidence and a precise next no-training control.
