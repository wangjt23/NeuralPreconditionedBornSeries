# RTE: select mechanisms from the bottleneck, then respect the precision gate

Completed 2026-10-02 on FWI. **The strongest near-term direction is cheaper application of the existing classical TSA action, before another learned or coarse correction.** Its cheapest algebraic check found removable work but failed the predeclared FP32 physical-image parity gate. Retain the prototype and negative cases; do not replace the baseline, retrain the rejected tail, or start another campaign. A precision-aware accuracy/cost check is the specific evidence needed to reopen this direction.

## Basis for the decision

The actual project is `/data/wjt/NeuralPreconditionedBornSeries/rte`, branch `main`, HEAD `028caea3f35d100225f656e2b1e910c1366aaafe`; RTE means steady **radiative transfer equation**. This is the periodic 2D Fourier/S1 discrete problem with L12, g=.65 and epsilon=.1. The current sixteen development media are fast for the tuned classical portfolio: homogeneous-reference tail at N64/N128 and TSA16-zero at N256. The completed nonlinear-tail diagnostic saved no mean TSA16 classical calls at N128/N256 and lost runtime. Phase repair by four-forward averaging did not pay its inference cost. There were no actual learned rejections, and projected useful images retained about98–99% of their norm. These observations do not support a dominant rejection or discarded-image explanation; they are not a complete causal cost decomposition.

The saved eligible-case profiles identify classical preconditioning as the dominant category; they do not separate FFT, angular products, layout or launch cost. The homogeneous inverse is **already cached**; warm applications do not refactor it. Existing CT/NTO code already caches coefficient-conditioned residual-linear factors. September controls also include compiled HSS, deeper classical tuning, exact Galerkin/Schur corrections, moment controls and ROM/POD negatives. Recommending caching, linearity or a coarse grid by name would repeat previous work.

## Ranked genuinely different mechanisms

| Rank / mechanism | Attainable benefit and strongest competitor | Setup and inference/amortization | Smallest discriminator and current boundary |
|---|---|---|---|
| **1. Remove redundant work in classical TSA** | Preserve the existing exact-arithmetic action while removing an inner streaming evaluation at each stage. TSA16 has 126 spatial transforms; the conservative identity needs 94. Competitors remain unmodified TSA16-zero and the frozen per-grid portfolio. This is an implementation improvement, not novel transport theory. | Same cached angular inverse and full-angle coverage; no training, neural loading or new material setup. No repeated-source assumption. The prototype's geometry/cache guard and local operations still cost work. | Selected check: actual action/image parity and counted transforms on frozen media, including grazing probes. **Work count verified; FP32 image parity failed five of 48 checks.** No measured speed benefit or deployment qualification yet. |
| **2. Strictly cheaper fixed-material complex-linear replacement** | Replace expensive repeated full-resolution factors, rather than append a neural tail. Preserve exact local collision/streaming where cheap and predict genuinely expensive heterogeneous information. Competitors include optimized TSA, compiled HSS and existing CT/NTO at appropriate depths. | Pay material encoding once, then use fixed residual-linear operations. Full L-by-L cached fields cost O(d L² N²) memory. Offline training and preparation must be charged; learning analytic shifts or the already-cached homogeneous inverse has weak justification. | Before training, require an analytic/structured apply representation with fewer passes **and** useful original-A image progress on actual TSA16 frontiers. Existing CT/NTO pilots and expanded-depth controls prevent assuming that caching or phase symmetry supplies this benefit. No credible current Pareto point justifies training. |
| **3. Material-adapted multilevel/coarse response** | Address heterogeneous long-range/directional error that a cheap full-angle complement leaves. Potentially greater scientific significance under demonstrated stiffness/angular refinement, but no current fast-cohort advantage. Competitors are exact Galerkin/angular corrections, ROM synthetic acceleration, recycling and the strongest full solver. | Assembly, basis acquisition/prediction, verification, rank/storage and transfers are charged. Reuse across **distinct** RHS can amortize setup; timing the same RHS repeatedly cannot. The FFT apply path supplies no ready sparse hierarchy; exact spectral assembly exists, but its scalable cost and hierarchy suitability remain unproved. | An existing-data coarse oracle must first yield useful residual-image coverage and complete cost headroom. Prior flat POD, radial coarse and inverse-compression negatives remain controls. Capturing solution-error energy is insufficient. No new coarse construction or learned campaign was run. |

Current primary work supports these distinctions, without transferring its conclusions to this RTE. [McMg](https://arxiv.org/html/2606.30495v2) separates nonlinear material setup from residual-linear multilevel application in Helmholtz; this design is established and already partly present here. [GMP](https://arxiv.org/html/2607.28456v1) uses assembled sparse hierarchies and exposes substantial setup/application costs against strong ILUT controls; it is an overhead warning, not a direct Fourier-RTE candidate. [Fan–An–Ying](https://arxiv.org/abs/1802.03061) is closer physically, with angular reduction, homogeneous FFT solves and heterogeneous skeletonization. [Peng FGMRES–ROMSAD](https://arxiv.org/abs/2410.08735) and [trajectory-aware ROM](https://arxiv.org/html/2509.05001v1) supply relevant kinetic correction/trajectory competitors. Their discretizations, coefficient access and convergence results do not automatically apply here. Detailed source checks and prior failures are in the companion mechanism notes; no novelty claim is made.

## Selected algebraic contribution

For fixed material, write `A=S+t I-s K`, where `K_ij=phase_matrix_ij*weights_j` uses actual stored quadrature. Let `L=S+gamma I`, `G=L^-1`, with `gamma=base.reference.sigma0`, and let `R` be the existing cached reference action. The original stage is

`g=G d; u=g+R(d-A g)`.

If the stored physical/reference streaming symbols coincide and `LG=I`, then

`d-A g=(L-A)g=(gamma-t)g+s K g`.

Thus the same stage needs only the local defect, with no inner streaming FFT pair. Exactness of `R` as an inverse is not required for this identity. The prototype retains the original outer `d=rhs-A z` updates, depth, relaxation, reference preparation/cache lifecycle and physical angular action. A strict pointer/version guard checks actual streaming symbols, phase and weights; it does not silently regenerate or symmetrize geometry.

At depth d, transform calls are `6d+2(d-1)` originally and `4d+2(d-1)` after the change: **126→94 at depth16**, a 25.4% reduction in this count. All sixteen reference angular applications and fifteen outer physical defects remain. This is not a 25.4% wall-time prediction: internal cost fractions are unmeasured.

Use the reference's actual `sigma0`, including its shift. The physical operator's existing `V_op` may use a different floor/removal coefficient and cannot be substituted blindly. More aggressive Fourier-state/recursive-residual fusion was deliberately deferred: it relies on stronger homogeneous-inverse identities and can suppress refinement of FP32 inverse roundoff.

## Actual cheap check and retained negative result

The isolated directory is `rte/workstation_runs/classical_tsa_algebra_20261002/`. No original source, trained model or physical data was changed. Existing development IDs **0 and 3** were used at N64/N128/N256; regenerated seed-4626 identity fields were checked exactly against all seven saved material/source fields. Inputs were the existing physical source and an analytical Nyquist/grazing stripe, not a new source dataset or held-out evaluation.

Forty-eight checks covered two media × three grids × depths1/16 × complex64/complex128 inputs × two probes. Before execution, the gate required finite outputs, exact transform counts, relative field **and original FP64 physical-image** differences at most `1e-4` for complex64 and `1e-9` for complex128 inputs. The complex128 checks retain the stored FP32 material/geometry/reference cache; separate native-FP64 CPU fixtures test fully double-precision construction.

| Observed quantity | complex64 input | complex128 input, stored FP32 cache |
|---|---:|---:|
| Maximum relative field difference | 5.03965e-6 | 9.39107e-15 |
| Maximum relative original-physical-image difference | 5.38673e-4 | 1.42950e-12 |
| Checks passing frozen parity gate | 19/24 | 24/24 |

All 48 outputs were finite and every transform count matched. **Five failed checks** were the actual physical source at depth16 in complex64: ID0 at all three grids and ID3 at N128/N256. None was the grazing stripe. In those five, field differences were only `1.83e-7–7.76e-7`; the physical operator amplified their relative image discrepancy beyond the gate. This establishes why raw field agreement alone is insufficient. It is not a finding of failed full FGMRES solves, instability, or worse final solution accuracy: those were not run. Nor does it establish whether the transform reduction saves latency.

Finite precision replaces `LGd=d` with an approximate identity. The idealized difference between original and local defects is `d-LGd`; subsequent reference action and stiff/refined physical differentiation can amplify the resulting difference. No threshold was relaxed after seeing results. The process deliberately exited **2** for failed feasibility; all cases and the unsuccessful decision are retained.

Six focused tests and **58 subtests passed on FWI**, covering 32 material/shift/depth/relaxation parity configurations, physical-defect identities, phase/additivity/zero behavior, transform counts, stored-physics rejection, and cache/material refresh. Operator checks took 7.893 s internally, 11.629 s including process startup; tests took 4.412 s including imports. Actual runtime was Python3.12.4/Torch2.5.1/CUDA12.4 on the existing RTX A6000. Check end-of-process occupancy was 938 MiB, not a peak/deployment measurement. Reserved seed9626 was untouched. No full-solver benchmark, training, teacher acquisition, package installation, job disruption, Git commit/index/push, access change or paid resource occurred.

Reproduce from the unchanged full FWI checkout, with a fresh output directory:

```bash
cd /data/wjt/NeuralPreconditionedBornSeries
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:rte/workstation_runs/classical_tsa_algebra_20261002/code \
  /data1/software/miniconda3/bin/python \
  rte/workstation_runs/classical_tsa_algebra_20261002/code/check_frozen_tsa.py \
  --output rte/workstation_runs/NEW_TSA_ALGEBRA_CHECK --device cuda
```

For the tests, use `rte/workstation_runs/classical_tsa_algebra_20261002/code/run_focused_tests.py NEW_TEST_RECEIPT.json` with the same repository/PYTHONPATH. Code, logs, 48 cases, thresholds and hashes are preserved. Helper SHA256: `85ca74f338091c645ca77b198d74f011eca09ec62b86d27d732ba92126de5e81`; check script: `0f6d14f8558497776b4c0382d061a210044d7e52cdcebd62428566d9d3121531`. Final preservation receipt records unchanged originals, inputs, checkpoints, development parents, branch and prior reports.

## What exactly would justify reopening

**For classical optimization:** explain/control the five FP32 image discrepancies, or explicitly evaluate the variant as a changed floating-point preconditioner. A separately authorized bounded paired check must show zero original-RHS convergence failures, unchanged `1e-5` physical tolerance, acceptable solution error, and worthwhile setup-plus-solve and warm latency against the strongest frozen portfolio. Wider arithmetic may recover parity but its cost must be measured. Do not label the current prototype an unchanged deployed action or infer benefit from transform counts. If that check fails, retain the implementation negative and stop it.

**For a learned replacement:** exhibit a genuinely cheaper operation/storage representation and an existing-data oracle showing useful TSA16-trajectory progress sufficient to repay its apply/preparation costs. Then specify matched analytic controls and charge data/training/loading. Complex linearity, existing small-grid positive CT/NTO evidence or another epoch count alone cannot justify reopening.

**For coarse learning:** demonstrate that a verified classical/oracle range and inversion yield a cost advantage across distinct sources, including setup/rank scaling and grazing angular channels. Compare with exact Galerkin, ROM/recycling and optimized classical factors before learning the space. A new physical regime can motivate a new hypothesis but is not evidence of current benefit.

Until that evidence exists, stop new learned/coarse campaigns and baseline replacement. Retain the three mechanism hypotheses and this inexpensive failed discriminator. Earlier computed coercivity bounds remain discrete, ordinary-FP64 estimates under their weighted-phase/skew-streaming/positive-margin assumptions, without interval enclosure; this study adds no solution-error certificate or continuous-RTE claim.
