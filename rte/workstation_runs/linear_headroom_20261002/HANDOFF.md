# FWI RTE continuation: faster exact classical control and angular obstruction

2026-10-02. Actual FWI execution is complete for this bounded continuation. The most useful positive result is an exact classical replay implementation. A learned advantage remains unestablished. The q3 learned-inverse proposal did not earn a training run; its numerical least-squares obstruction was investigated rather than treated as a negative optimum.

## Access, project and preservation

Existing SSH alias `FWI` authenticated as `wjt` on `FWI-Standard-PC-i440FX-PIIX-1996`. Login directory is `/home/wjt`. Repository real path is `/data/wjt/NeuralPreconditionedBornSeries`, origin `git@github.com:wangjt23/NeuralPreconditionedBornSeries.git`, branch `main`, HEAD `028caea3f35d100225f656e2b1e910c1366aaafe`. The RTE workspace is `/data/wjt/NeuralPreconditionedBornSeries/rte`. RTE means **radiative transfer equation**, identified from the project itself. The paused OL_preconditioner project was not resumed.

The original checkout contains nine tracked dirty files and substantial untracked research; HEAD alone is not its reproducible state. Original source, input, checkpoint, development-parent and previous-report hashes are retained. New files and execution are isolated under `rte/workstation_runs/linear_headroom_20261002/`, with `v2/`, `angular/` and `whitening/` subdirectories. No original solver/data/checkpoint was replaced; no commit, push, merge, package installation, system/SSH/proxy change, port forwarding, private-key read, paid compute or unrelated-job cancellation occurred.

SSH used existing credentials with BatchMode, ClearAllForwardings, StrictHostKeyChecking, ConnectTimeout10 and ConnectionAttempts1. Temporary Mac execution-bridge disconnections returned literal `exec-server transport disconnected` errors; subsequent SSH and file receipts confirm recovery. Those errors did not establish an FWI Internet outage. All substantive RTE numerical experiments ran on FWI, using `/data1/software/miniconda3/bin/python` (Python3.12.4, Torch2.5.1/CUDA12.4). No Mac compute target was substituted. Initial and post-run GPU checks show only the existing desktop process (8MiB); no other user research job was disturbed. Hardware remains RTX A6000/49,140MiB, 36CPUs/184GiBRAM. No applicable ancestor/subtree AGENTS.md or project .agents/skills was found in the initial scoped audit.

## Scientific audit carried forward

The accepted discrete operator is the original promoted FP64 periodic 2D FFT transport action, twelve S1 angles, g=.65, epsilon=.1. Stored rounded geometry, quadrature, phase and material buffers are retained; ideal geometry is not regenerated. Original real-stacked flexible GMRES acceptance remains relative to the original RHS, tolerance1e-5, restart25, maximum100 directions. Every current full-solver output has an independent physical residual, finite-field check, positive evaluated accretivity margin, relative field check1e-4 and evaluated discrete solution-error bound1e-3.

For actual K=phase·diag(weights), mu=maxeig(Herm K), m=min(t−s·mu)>0 gives the exact-arithmetic inequality ||H^-1||≤1/m. With delta=||b−Hx||/m and ||x||>delta, relative solution error is at most delta/(||x||−delta). Current checks evaluate these quantities in ordinary FP64; they are not interval certificates or continuum/OOD error statements.

Train3626, development4626 and old heldout8626 are distinct physical draws. Development4626 has been repeatedly reused and is explicitly exploratory. Nested resolutions are the same media, not new samples. Reserved9626 remains sealed. Contemporary comparisons include original tuned TSA/HSS/reference controls and the newly verified exact replay. Classical transport/ROM precedents, discretization differences and learned-linearity caveats are documented in `literature_audit.md`, `oracle_protocol.md` and `math_feasibility.md`; no generic novelty claim is made.

Earlier controlled negatives remain valid: mixed-resolution tail learning failed the gate; four-phase tail averaging saved some work but lost cost; the algebraic TSA and its FP64-sum repair each passed43/48 entry checks and remain rejected. Rejection/frontier semantics were repaired only in an isolated diagnostic helper. Historical acquisition/tuning costs and deployment source reuse remain incomplete evidence.

## Positive result: exact classical replay, full development cohort

`GraphClassicalV2` captures the original prepared TSA floating-point stages unchanged. It removes repeated launch overhead, preserves native complex128 eager fallback, returns separately owned complex128 outputs and holds strong operator/tensor references. It checks object identity, mutation versions, stride, storage offset, cache and settings. The v1 screen's general same-pointer-view guard gap was repaired in a new file; the executed v1 screen itself used unchanged objects and remains valid. Use v2 serially with its validated TSA8/TSA16 configurations. Concurrent replay and untested classical kinds are outside this verification.

On FWI:13 focused CPU tests passed,48/48 source/Nyquist/precision entry comparisons were explicitly bitwise equal, and12 guard/ownership bundles passed, including same-pointer transpose replacement rejection. Only after this gate, all16 saved development media at N64/N128/N256 underwent288 paired full solves:96 retained warmups and192 measured solves. **Every paired full solution is explicitly torch.equal**, with identical calls/iterations and all accuracy checks passing. Maximum physical residual9.9850e-6, minimum evaluated m=.01270603136, maximum evaluated relative solution-error bound.00023688606.

| Grid | Original frozen warm ms | Replay warm ms | Warm saving |
|---|---:|---:|---:|
| N64 | 134.205 | 33.328 | 75.166% |
| N128 | 136.580 | 40.847 | 70.093% |
| N256 | 148.651 | 96.806 | 34.877% |

Means reduce two measured repeats to a per-medium median first. Frozen recipe is reference_tail at N64/N128 and TSA16-zero at N256. All individual warm comparisons improve; this is classical implementation evidence on existing development data, not learned or heldout superiority.

Setup is material. Mean additional capture/warmup setup is120.481/122.534/186.229ms. Conditional setup+Q×same-source warm estimates regress at Q1 by13.802/18.479/84.065%. Grid-mean break-even integer Q is2/2/4. At Q4, estimated gains are51.965/46.937/3.491%; N256 still loses in8/16 cases. At Q16 all tested cases have estimated ≥5% benefit. These estimates assume distinct RHS have the measured latency; repeated timing of one RHS does not demonstrate source reuse. The final host signature-binding step is omitted from recorded capture setup, so the cost estimates are slightly optimistic. Process-cold latency and distinct-RHS amortization have not been measured. Keep eager classical as the successful single-RHS setup-inclusive control.

`graph_v2_full16_handoff.md`, `graph_full16_audit.md` and authoritative `full16/*` receipts give exact setup/Q tables and per-case evidence. Original48-entry v1 screen and shallow-reference failures are also retained: reference1 failed one medium on every grid; reference2 passed but was slower. Aggregate v1 `all_solve_accepted=false` refers to these retained reference1 failures, not replay failure.

## Pursued mathematical direction: full-spatial angular Galerkin and image whitening

The new hypothesis retained every spatial/Nyquist frequency while restricting angular unknowns to q3 Fourier columns0,+1,−1. Aq=Q*H Q uses heterogeneous coefficients and actual H. Its matrix-free implementation and full adjoint passed8 FWI CPU tests plus actual-grid projection/adjoint checks. Coercivity proves exact Aq invertibility, but its oblique residual projector can amplify physical residuals. Adaptive inner GMRES is a nonlinear classical/oracle action, even when capped; it is not a fixed linear learned inverse.

Only development IDs0/3 at N128/N256 were used: two independent media, four grid/material pairs, source and residual after four original classical directions (eight contexts). All eight inner solves achieved independently recomputed coarse residual≤1e-8 within22–66 calls. Yet full physical residual ratios were2.57–7.45. The useful line residual after the old real AZ projection stayed.99962–.99998 for Galerkin tail proposals, demonstrating how coarse convergence can fail to supply a useful tail direction.

Cached directions were inserted through original safeguarded full GMRES. Their acquisition cost was excluded optimistically; owned copy, initialization, physical images, joint projection, continuation and final checks were billed. All112 full solves passed accuracy (84measured,28retainedwarmups). Each arm uses its own contemporaneous timing-block baseline.

| Cached proposal | N128 warm saving | N256 warm saving |
|---|---:|---:|
| Galerkin source initializer | −1.828% | −9.776% |
| Galerkin tail | −20.264% | −31.927% |
| Attained LS source initializer | +1.074% | −9.807% |
| Attained LS tail | −7.062% | −15.125% |

Galerkin acquisition alone took100–293ms/context, before paying coarse preparation. The first least-squares diagnostic used Aq's homogeneous inverse as a right preconditioner. All eight100-step runs were underresolved (normal residual.0321–.3878), with raw-source lower bounds zero. That result was not called an optimal angular-range rejection.

One bounded numerical repair then used the **full rectangular homogeneous image** B0(k)=H0(k)Q and W(k)=[B0*B0]^-1/2. This retains angular streaming components outside Q. The exact identity

    B0*B0 = Aq0* Gram^-1 Aq0 + B0*(I−Q Gram^-1 Q*)B0

shows the positive image term omitted by the Galerkin reference. W makes homogeneous full images isometric in exact arithmetic; it does not guarantee heterogeneous or AZ-projected conditioning. Seven FWI CPU tests and actual-grid whitening/adjoint checks passed; maximum identity defect2.18e-14. No eigenvalue clipping or physical-operator symmetrization was used.

All eight repaired LS runs still reached100 steps, but independently checked normal residual improved to3.76e-7–1.71e-5. The conditional raw-source minimum-residual brackets are now narrow:

| Grid/ID | Ordinary-FP64 conditional raw minimum relative residual interval |
|---|---|
| N128/0 | [.658532558, .658532956] |
| N128/3 | [.602040957, .602042641] |
| N256/0 | [.659301761, .659302539] |
| N256/3 | [.601709587, .601713421] |

For t=r−HQc and g=(HQ)*t, the exact-arithmetic bracket is sqrt(max(0,||t||²−||g||²/(m²lambda_min(Gram))))≤minimum≤||t||. These evaluated intervals remain conditional, not rounding enclosures. AZ projection can destroy the singular-value lower bound; projected prefix results supply attained upper bounds only (.948–.981), with no optimality/range-exclusion claim.

The48 repaired cached full solves all passed accuracy (36measured,12warmups). Source-initializer savings were+1.080%/−9.709%; tail savings−8.720%/−14.814%. Acquisition costs328–780ms/context, whitening setup28–185ms; q3/N256 diagnostic W/B0/eigenvalue cache≈46.5MiB plus existing coarse/operator buffers. The small positive N128 cached budget≈.517ms cannot cover these costs and fails the10% screening margin; N256 has negative budget. No actual expensive-inner full deployment solve, q5 extension, new teacher acquisition or training was launched because the paired headroom gate stayed closed.

An initial whitening CPU staging attempt omitted its shared test fixture. The fixture alone was added before GPU execution; initial failure log/policy/receipt are retained. Numerical algorithm and data were unchanged. Actual repaired whitening process took25.760s within140s; angular process54.285s within260s; full16 replay process218.324s within300s. This continuation totals544 full-solver invocations including160warmups,384measured (v1screen96, v2full16 288, angular112, whitening48). No training ran in this continuation.

## Ranking after the evidence

1. **Retain the exact replay control and make workload costs explicit.** This is the only demonstrated positive development improvement here, with exact fields. It tightens future learned comparisons. A real distinct-RHS workload is needed to demonstrate amortization; do not assume Q16 or count phase-related RHS as independent savings.
2. **Change the physical response range/closure before learning another coarse inverse.** A constant angular q3 range has substantial raw-source residual floors and the tested tail composition has no budget. A bounded oracle using the existing full homogeneous transport-response range R0Q, with heterogeneous correction/Schur information and a cheap full-angle complement, is more justified than increasing FNO capacity or blindly raising q. The project already implements `reference_transport_solve(..., harmonics=K)` in this response range: it must be a classical control, not relabelled novel. Check existing September variants and use the same physical operator/cost gates before any fitting. This is a next hypothesis, not an executed or winning method. A different repeated from-start composition was not excluded by the current one-shot/source results.
3. **Trajectory-aligned fixed-material linear actions remain a conditional longer direction.** The old tail trained behind TSA8 differs from the strongest TSA16-zero trajectory; a new learned representation must supply useful images and save complete time against replay while paying setup/offline costs. Existing per-material polynomials, homogeneous moment/response methods, CT/NTO, POD and hierarchy negatives must be credited. Mathematical/physical significance would come from a compatible streaming-aware correction and a verified cheap action, not a new network name.

The first next step toward a learned win is a predeclared small cost/quality oracle for a physically different response range, not a larger training campaign. Novelty is unresolved against the angular-subspace, transport-ROM and heterogeneous FFT/skeletonization literature. Preserve9626 until a complete frozen development gate succeeds.

## Deliverables, reproduction and remaining gaps

This bundle contains original-context handoff/audit/literature, new mathematical notes, executable isolated scripts/tests/policies, raw JSON/logs, independent audits and hash manifests. Full datasets and old optimizer checkpoints remain on FWI; original source/input hashes and the earlier private Library snapshot provide provenance. The current bundle does not duplicate multi-GB data.

Execute from the recorded repository with the recorded Python, PYTHONPATH containing repository plus the relevant isolated code directory, and new output roots. Completed runners refuse result-directory overwrite. Read each frozen policy and check jobs first. Preserve the 100-direction/tolerance limits; never generate9626 merely by rerunning an older benchmark entry point. Graph v2 requires its complete entry gate before solver timing. Exact driver/model/data hashes are in the receipts.

No access/execution blocker remains. Learned superiority, actual single-RHS/process-cold benefit, distinct-source amortization, heldout/OOD/continuum behavior, historical total teacher/tuning cost and interval numerical certification remain open. All failed gates and capped diagnostics are retained. No learned success is claimed from the classical replay result.
