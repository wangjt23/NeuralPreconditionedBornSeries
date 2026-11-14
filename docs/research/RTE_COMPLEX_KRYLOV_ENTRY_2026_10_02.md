# RTE / FWI complex-Krylov entry obstruction — 2026-10-02

The single frozen discriminator stopped at its numerical entry gate, with **zero full solves**. Only the original FP32 `low_H64` action failed: 8/12 records exceeded its prespecified `2e-6` normalized consistency cap. The worst defect was `5.036133085028879e-6`, or 2.518 times the cap. The physical FP64 operator, all standalone reference factors, both strong transport implementations, the actual cast-back path and its physical images passed their separately frozen caps. This is a completed numerical-admissibility obstruction, not an observed complex-solver convergence failure or a runtime comparison.

No cap was relaxed, no scalar gain variant was resumed, and no full comparison was forced. The unchanged learned 10% gate and sealed seed9626 remain in force. Prior negative results remain preserved. The isolated complex solver and admissibility helper passed 31 CPU tests on FWI; they are available as reviewed infrastructure, not a demonstrated RTE performance improvement.

## Access, project identity and preservation

Access succeeded using existing SSH alias `FWI`, user `wjt`, hostname `FWI-Standard-PC-i440FX-PIIX-1996`. The exact repository and realpath are `/data/wjt/NeuralPreconditionedBornSeries`; project directory `/data/wjt/NeuralPreconditionedBornSeries/rte`. RTE means **radiative transfer equation** in this project. Git branch is `main`, HEAD `028caea3f35d100225f656e2b1e910c1366aaafe`, origin `git@github.com:wangjt23/NeuralPreconditionedBornSeries.git`. This is distinct from the separately paused OL_preconditioner project.

SSH used BatchMode, ClearAllForwardings, StrictHostKeyChecking, a ten-second connection timeout and one connection attempt. Existing credentials were used without reading/exporting key contents, altering trust/configuration, forwarding or network settings. FWI was the numerical execution target; the Mac bridge only prepared authored code, reduced scalar receipts and verified artifacts.

The existing RTX A6000 has 49,140 MiB memory. Preflight found no other user research job, only an 8 MiB desktop-integration GPU process. Post-run hardware was 37 MiB used and 0% utilization, with no user research process. Python was `/data1/software/miniconda3/bin/python` (3.12.4), Torch2.5.1/CUDA12.4. No new package was installed.

All **442** post-run hashes matched: 420 preserved originals/prior artifacts plus 22 new execution payloads. These include all 155 original source files and nine pre-existing tracked dirty files. Branch/HEAD and tracked dirty status were unchanged. No source export, training, seed9626 access, Git write/commit/push/merge, job cancellation, destructive cleanup, new access, paid compute or large campaign occurred. The scoped access receipt found no applicable AGENTS.md or .agents/skills in the checked project locations. The exact inspected scope is in `receipts/complex_krylov_access.json`; this is not a claim about every unrelated directory.

New remote run: `rte/workstation_runs/complex_krylov_20261002`. The final handoff is also saved there as `HANDOFF.md`, and under `docs/research/RTE_COMPLEX_KRYLOV_ENTRY_2026_10_02.md`. Original handoffs were preserved.

## Scientific authority retained from the resumed audit

The stored problem is periodic two-dimensional steady RTE on the unit square, 12 angular ordinates, g=.65 and epsilon=.1. Its nominal discrete action is

    H psi = S psi + sigma_t psi - sigma_s K psi,
    K = phase_matrix diag(weights).

The physical authority is the original `high_precision_copy` of the stored low-precision operator: stored buffers are promoted, with the original streaming-product construction. No ideal geometry, new material, rebuilt reference inverse or continuum operator replaced it.

Original residual acceptance is source-normalized `||q-H psi||/||q|| <= 1e-5`. A finite-discrete coercivity evaluation uses

    m = min_x [sigma_t(x)-sigma_s(x) lambda_max(Herm K)] > 0,
    delta = ||q-H psi||/m,
    relative_error_bound = delta/(||psi||-delta),

with a positive denominator, finite solution, solver-success status, evaluated bound<=1e-3 and paired final-field difference<=1e-4. These are ordinary FP64 evaluations for the stored finite problem; they are not interval, continuum or OOD certificates. No accepted field was produced in this round, so none of these full-solve criteria has a new measured outcome here.

Training seed3626, repeatedly reused exploratory development seed4626 and historical8626 were identified previously. Seed4626 is not a fresh held-out confirmation set after repeated selection. Seed9626 remains sealed. This round used only IDs0/3 from the existing seed4626 N128/N256 development parents; their seven stored/regenerated metadata keys matched exactly, and pinned parent SHA256 values are in the frozen policy. No arrays, model weights or original source bytes are included in the evidence bundle.

The strongest retained classical recipes are TSA8 with balanced original R0 initialization and a single attempt after four global classical calls at N128, and TSA16/zero start/no injection at N256. Both planned coefficient controls use the same existing explicit-segmented factors, reference cache, original H and source. The earlier independent 16-medium direct-capture confirmation achieved Q1 savings of 20.78435%/19.64482% at N128/N256. That strengthens the classical denominator; it does not establish learned superiority.

## Frozen experiment and executed entry evidence

Policy SHA256: `acd73134c63839726d3b699c60e795b8da90d14709a9529fb94e905ce59ebd93`.

Driver SHA256: `efc7d6f95598e1a7b02c60c1740ab52660a2a48e19cc9d67e90f76564bacb3bf`.

Admissibility helper SHA256: `c7a76527ee259b0aca5e9620248574bb46d73864a0719ccaadb65da7591280a4`.

Complex companion SHA256: `d517306d65f796b923c08735be235f7d56b615614e8a068fcd35349a592bfe4e`.

The fixed screen allowed 96 total full solves (24 warm/72 timed), 32 in the principal matched pair and 64 in four extra original-real backend controls. It fixed restart25, max100, tolerance1e-5, precision, recipes, IDs, repeat count and backend before execution. There was no parameter sweep, teacher acquisition, optional span fit or additional candidate. A 95-second soft deadline, 120-second owned-process hard timeout and 12 GiB reserved-memory envelope applied.

All **112** original-eager/captured bitwise, output-ownership and cache-view guard checks passed. All source/start/shared-cache checks passed. Every source was exactly real in complex storage and physical/low source values were bitwise identical. All 108 action records completed, each with ten evaluations (1080 calls total), before any full solve. Every record had finite outputs, exact zero, consistent expected dtype, no input mutation, and no action/contract errors.

Three fixed pairs were tested per action/context: source/mixed source, grazing-Nyquist/source and deterministic unit-norm complex random fields. Identities were zero, i-homogeneity, a fixed general phase, real scaling and one fixed complex combination. Each record checked seven nonzero identities. The pass denominator is the noncancelling reference-output sum; a separately recorded symmetric denominator is diagnostic and never used to relax the gate. All 756 scalar identity ratios and flags can be independently recomputed from receipts.

| Actual action | Frozen cap | Failed records /12 | Maximum normalized defect |
|---|---:|---:|---:|
|Physical H, complex128|1e-12|0|9.780659e-15|
|Original low H, complex64|2e-6|8|5.036133e-6|
|Original G, complex64|5e-6|0|2.622050e-7|
|Original R0, complex64|5e-6|0|7.625112e-7|
|Strong eager TSA, complex64|1e-4|0|1.526864e-5|
|Strong eager TSA, native128|1e-12|0|2.865096e-15|
|Actual128→64→TSA→128 path|1e-4|0|1.546164e-5|
|Direct-segmented TSA, native128|1e-12|0|2.865096e-15|
|Physical H after actual deployed TSA|1e-3|0|5.897304e-4|

In each context exactly the two source-bearing low_H64 records failed; all random-complex low_H64 records passed. There were 32 failed scalar identities in those eight records. The source is reused as the second nyquist_source probe, so these records are overlapping evidence, not eight independent sources. Context maxima were 2.200936839e-6 (N128/ID0), 2.401701059e-6 (N128/ID3), 4.716236796e-6 (N256/ID0) and 5.036133085e-6 (N256/ID3).

The sharpest frozen witness is N256/ID3, source_mixed, phase_y. With the actual complex64 source-derived input y and c=exp(.37i):

    ||low_H64(c y)-c low_H64(y)|| = 0.01587103861992238,
    |c| ||low_H64(y)||            = 3151.4335209096985,
    normalized_defect            = 5.036133085028879e-6 > 2e-6.

The fixed real_y scaling also failed there: numerator0.026673347755984046, denominator5357.436985546487, ratio4.978751561230584e-6. Witnesses include i-homogeneity and complex-combination failures at N256; the complete individual metadata is retained. The record expressions use the actual native input/output arithmetic; only diagnostic differences/norms promote128.

This establishes a reproducible violation of the declared numerical gate for the actual original FP32 action. General scalar identities do not isolate operator roundoff from native input-scalar-expression roundoff; multiplication by i is an exact component swap/sign on these finite inputs, so its N256 failures cannot be explained solely by input-scalar rounding. The receipts do not prove structural conjugation/nonlinearity, give a universal forward-error bound, or constitute an interval-certified exact counterexample. Cancellation in a source-conditioned H image is a plausible explanation, not a verified causal conclusion. The source-level formulas remain nominally complex-linear. Passing uncast128 shadows, the actual deployed cast-back factor and its physical images cannot waive the separate failed auxiliary low_H64 cap. Native128 segmented evaluation uses the original eager128 fallback, not a newly captured wide graph. These engineering caps were fixed before results and are not derived universal error constants.

## Complex versus real geometry and admissibility limits

The original real-stacked solver retains real and imaginary field components but uses real Arnoldi/least-squares coefficients. Its product is Re(u* v). The companion uses Hermitian complex products and complex coefficients, storing actual returned directions; its tiny least-squares solves use rank-aware complex SVD. SVD/rank-loss safeguards differ from the unchanged real control and would be explicitly charged, so the proposed timing experiment was not a pure coefficient-only causal attribution.

For one frozen set of actual directions Z and unprojected images W=H Z, nominal complex-linear H gives

    min_(c in C^k) ||r0-Wc||
      = min_(a,b in R^k) ||r0-[W,iW](a,b)||
      <= min_(a in R^k) ||r0-Wa||.

For fixed exact complex-linear H and M, unrestarted real Krylov corrections are contained in complex corrections at matching degree. These are search-space statements. They do not imply dominance of restarted, rounded, injected or differently safeguarded trajectories, iteration counts or runtime. A fixed real image-space projector need not commute with multiplication by i. No optional measured-span/oracle fit was performed.

Flexible GMRES can accommodate varying or nonlinear preconditioning by retaining each actual returned direction and its physical image. This does not restore a fixed-Krylov inclusion theorem, and it does not waive the user's required pre-entry check. See the primary [GMRES paper](https://epubs.siam.org/doi/10.1137/0907058) and [flexible GMRES paper](https://epubs.siam.org/doi/10.1137/0914028). The finite-precision qualifications and span arguments above are this project's analysis of its inspected implementation.

The physical source is real; the original solver imposes no real-only solution or imaginary projection. Original H applied to actual real sources produced imaginary norm fractions 0.00155832/0.00000336449 at N128 and 0.00142503/0.00000292664 at N256, for IDs0/3. Thus the stored original problem does not preserve a real source subspace on these measured sources. The particular grazing stripe probe had zero imaginary H-image in this run; that does not establish real invariance. Preserve the original spectral/Nyquist convention. Solving the stored complex problem is distinct from proving real continuum intensity or positivity.

## Actual costs and audit conclusion

Owned process87.909460394 seconds; runner84.434397735 seconds; peak allocated1,117,886,976 bytes and peak reserved2,009,071,616 bytes. No timeout occurred. Mandatory entry acquisition consumed **73.558859862 seconds** (14.88435,16.04904,20.23557,22.38990 seconds across the four contexts). The allocation peak includes probe snapshots and all held graph contexts; it is not a solve-memory measurement.

The recorded preparation ledger totals physical/material0.258380434s, base0.082642699s, four graph constructors1.354872691s, coercivity/acceptance setup0.076405972s, original-real tiny-LS preparation0.007326322s and complex tiny-SVD preparation0.063418653s. Mandatory acquisition includes helper snapshots/norms/metadata, per-record receipt serialization and source/start diagnostics within its timer. These measured components need not sum to process wall time; source IO, Python/startup, guard/parity checks and QA outside those timers are excluded from the algorithm ledger and remain in measured wall time.

The prespecified ledger would charge entire material/base/capture/acceptance/solver preparation, changed complex arithmetic, copies, orthogonalization, SVD/rank handling, injections and accepted physical solves. Common mandatory entry acquisition would additionally be charged once per context to every arm without amortization, with operational costs separately reported. Complex vectors use the same underlying two-real-channel field storage as real-stacked vectors; a26×25 complex Hessenberg is10,400 bytes versus5,200 bytes real. The default solver retains no historical raw HZ capture, but a current raw image remains live. Actual solver allocations and extra orthogonalization were to be measured within the complete solve.

Because the gate stopped all96 planned full solves, the charged Q1, warm-solve savings, comparative convergence, physical solution acceptance and solver storage evidence are **unavailable**. No acquisition-only timing is presented as a speedup, hardware floor or proof of economic impossibility. Both the contemporaneous five-backend envelope and the separately labeled historical eleven-recipe sensitivity remained unexecuted. Existing learned10% gate (Q1 and warm at both grids, accuracy and individual regression constraints) is unchanged.

Independent frozen-protocol, mathematical, implementation and scalar-receipt audits accompany the handoff. Root and independent stdlib reductions agreed; the bundle verification re-executes both from extracted receipts without Torch, GPU or original sources. Post-run preservation passed all442 checks with no changed original/tracked work or extra research job.

## Step back and next directions

1. **Resolve the numerical-entry uncertainty analytically before any further complex screen.** Derive an a priori forward/backward error model for the original FP32 FFT/products and native scalar-expression construction, with an explicit conditioning factor for source images. A model-normalized gate would need its constants, acceptance connection and protocol fixed before new results. This has mathematical value and directly addresses the measured obstruction; it is not permission to raise2e-6 after this failure. No such new GPU experiment was run.
2. **Keep exact explicit-segmented classical capture as the working denominator.** Its preserved16-medium evidence is the clearest established runtime improvement. A future physical-recipe complex comparison remains unresolved and requires a newly justified admissibility protocol. Native128 passing here is attribution evidence, not authorization to switch the actual pipeline.
3. **Consider learned or compressed residual directions only after an affordable representation and decisive independent evidence.** The earlier learned gate failures, full/q3 matched negatives and stopped gain-two result give no basis for new training. Require total setup/training/inference/solve economics against the strengthened classical portfolio, fresh held-out/OOD confirmation and original residual/error acceptance. None was done in this entry round.

Gain-two and scalar-gain variants remain stopped. Prior gain-two had0.076028/0.188107s warm accepted means versus0.054177/0.129585s historical strongest means at N128/N256, and failed the frozen strong envelope despite fewer calls than one weaker same-formula control. Its exact verified private artifacts remain [gain-two report](https://chatgpt.com/api/library/files/libfile_4e610149396481918dd1fdb1a1187ded/download) and [gain-two evidence](https://chatgpt.com/api/library/files/libfile_1140a9ee6cd88191aaae2985556deaa3/download). Earlier direct-capture, precision, q3, full-response and matched-reference negatives were preserved and are referenced by the included gain-two handoff.

The authorized bounded experiment is complete under its stop rule. The remaining blocker is the original FP32 consistency-cap violation, not access. There is no claim of learned progress, newly certified complex performance, OOD generalization, continuum accuracy or solved physical-reality ambiguity.

## Evidence and reproduction

Frozen code/policy and execution receipts are in `receipts/complex_execution/`; final preservation is `receipts/complex_final_preservation.json`. `deliverables/complex_krylov/RESULT_ANALYSIS.json` is the independently implemented root reduction. The independent result auditor, mathematical result review and implementation review identify the same eight failed low_H64 records.

Receipt-only reproduction from the supplied bundle root:

    python3 deliverables/complex_krylov/analyze_complex.py receipts/complex_execution /tmp/complex_root_analysis.json
    python3 deliverables/complex_krylov/audit_complex_receipts.py receipts/complex_execution --output /tmp/complex_independent_analysis.json --expected-policy-sha acd73134c63839726d3b699c60e795b8da90d14709a9529fb94e905ce59ebd93 --expected-driver-sha efc7d6f95598e1a7b02c60c1740ab52660a2a48e19cc9d67e90f76564bacb3bf --historical-summary deliverables/gain_two/RESULT_AUDIT.json

These reductions need only the standard library. GPU/CPU experiment re-execution would require the preserved original FWI repository and exact development parents; do not automatically rerun or modify its frozen policy. The evidence ZIP includes only newly authored code, scalar/hash receipts, audit notes and selected prior authored handoffs—no original source files, credentials, model/data tensors or sealed data. Its internal manifest records every payload SHA256 and its ZIP CRC/readback checks are in the verification receipt. The initial scientific audit and primary-source literature comparisons are included as `deliverables/RTE_FWI_HANDOFF_2026_10_02.md` and `deliverables/literature_audit.md`; they supply the wider classical/learned prior-art and provenance context, not new evidence from this stopped entry screen.
