# Discrete TSA finite-precision error budget

Independent source and retained-receipt audit, 2026-10-02. This note derives identities; it does not run remote code, train, generate data, or change the physical operator. The original failed 48-row gate remains a negative result. Diagnostic FP64 evaluations below are empirical estimates, not directed-rounding error certificates.

**Latest verified status:** the root's closed diagnosis resolves the earlier cache-equality and outer-tau accounting gaps. The sum-only repair then passed43/48 checks and failed the same five FP32 physical-source/depth16 image rows under the unchanged gate. No full solves or warm solver timings ran. The final section below supersedes the earlier pending-status paragraphs while retaining their derivation and pre-execution reasoning.

## Observed gap and what is still missing

The first diagnosis in `receipts/precision/diagnosis/` is complete: five previously failing physical-source/depth16/complex64 cases, 80 nominal stages, and 15 common-input replays at stages 0/7/15. Final image ratios reproduce the original five failures, from `1.0497897329e-4` to `5.3867264051e-4`; field ratios remain below `7.762e-7`. The local defect decomposition closes to relative `4.84e-11`–`1.72e-10` in reporting arithmetic.

On those five final discrepancy images, 89.53%–94.80% of Fourier power is in `max(|mode_x|,|mode_y|)>=N/4`, and 22.43%–27.95% is on the Nyquist rows/columns. These are **grid-normalized, overlapping bands**: the cutoff changes with N. They support high-frequency derivative amplification within each grid. They neither establish equal physical-frequency content across grids nor identify the kernel that injected the discrepancy. The initial spectrum helper subtracts complex64 fields before promotion; its spectrum therefore includes fresh subtraction rounding. A new corrected diagnostic should promote each field before subtraction while retaining the original parity metric separately.

The measured cancellation ratios are modest (1.056–1.672 across the two formulas/replay stages). Reference-transported inverse-identity and A-action defects exceed transported local-formula rounding. Nevertheless, they do **not** close the actual stage difference: for N256/ID0/stage0, the measured actual single-update image difference is `0.0151967481`, versus `0.0002018132` for the FP64 cached-reference action on the complete local-defect difference, a factor 75.30. Reference-action and addition rounding must be included before choosing a repair. Norms of individually transported terms cannot be summed to allocate a cause because their vectors can cancel.

## Pin the exact discrete objects

Let S be the exact linear spatial Fourier action of the **already assembled FP32** stream symbol, promoted without regenerating products. Let K be the exact action of the separately stored phase and quadrature-weight constants, C=tI−sK, A=S+C, L=S+gamma I, G=L⁻¹, and D=gamma I−C=L−A. Gamma is `base.reference.sigma0`, including its actual shift convention. It need not equal the physical object's `sigma0`.

Let R be the exact Fourier/matrix action of the stored reference-inverse cache bytes. R need not be an exact inverse of any ideal reference matrix. The reference solver casts this cache to the input dtype; a complex128 call promotes the existing inverse rather than rebuilding it. Cache equality between independently constructed original/fused objects must be verified; otherwise a separate cache-change term is required.

The gate uses H, the unchanged original physical FP64 operator. H promotes stored direction/wavenumber factors and forms their products in FP64. Consequently H is not exactly the same streaming matrix as S assembled in FP32. Keep this small symbol difference explicit; use S for identity attribution and H for acceptance. Relevant source: `remote_reference/rte/models.py:182–221,245–274,315–333` and `transport_baselines.py:25–70`.

The common exact single-stage map is

    F = G + R(I−AG) = G + RDG.

This identity requires common S and gamma, not an exact R inverse, phase symmetry, continuous transport consistency, or contraction of a Born iteration.

## Exact common-input accounting, including nonlinear kernels

For the same stored input d and the same computed g, write the actual rounded kernels as

    Ahat(g) = Ag + a(g),
    ho = fl(d−Ahat(g)) = d−Ag−a(g)+b,
    hf = localhat(g) = Dg+f(g),
    kappa = Lg−d.

Here b is the subtraction error, and f includes coefficient subtraction, products, scattering and local addition. All errors are exact differences from fixed linear maps evaluated on the actual operands. They may depend nonlinearly and discontinuously on those operands. The exact defect identity is

    hf−ho = kappa + f + a − b.

These are precisely the signs in the original and extended diagnostic. FP64 vector closure measures reporting error; a tiny closure establishes accounting, not dominance of any component.

Define the actual FP32 reference map `Rhat(h)=Rh+rho(h)` and final single-stage addition error `beta_p=fl(g+Rhat(hp))−g−Rhat(hp)`. Then the **actual common-input single-update difference** is

    uf−uo = R(hf−ho) + [rho(hf)−rho(ho)] + (beta_f−beta_o).

This remains valid even though Rhat is not globally linear. In the extended script, `rround=(Rhat(hf)−Rhat(ho))−R128(hf−ho)` estimates the bracket, and `sadd` estimates the addition-error difference. Computing a difference of two norms would not estimate either vector. Compute their vectors, H-images, pairwise real inner products, and the closure of their **sum** against the actual promoted single difference. H(R128(delta)) alone is not the complete stage error budget.

In exact arithmetic the triangle inequality supplies the common-input bound

    ||uf−uo|| <= ||R|| (||kappa||+||f||+||a||+||b||)
                  + ||rho(hf)−rho(ho)|| + ||beta_f−beta_o||.

For the fixed cached inverse, `||R||_2=max_mode sigma_max(Rhat_cache(mode))` under the unitary Fourier convention. No bound for an ideal homogeneous inverse may be substituted without including the cache inverse defect. Finite measured component norms without enclosing their reporting errors do not turn this inequality into a verified floating-point certificate.

## Transport inverse/FFT ledger

With a unitary diagnostic Fourier transform U and the fixed diagonal Lambda of L, normalize the actual backward FFT's intermediate errors consistently. If

    qhat = Ud + e_forward,
    vhat = Lambda⁻¹ qhat + e_division,
    g = U* vhat + e_inverse,

then exactly

    g−Gd = U*Lambda⁻¹ e_forward + U*e_division + e_inverse,
    kappa = U*e_forward + U*Lambda e_division + L e_inverse.

Thus inverse-FFT/output rounding in g can be differentiated/amplified by L. High-frequency kappa does not imply that the eliminated streaming FFT itself was its source.

The extended script uses actual backward normalization and its actually assembled denominator lambda. Its three Fourier differences telescope correctly:

    lambda FFT128(g32)−FFT128(d32)
      = lambda[FFT128(g32)−vhat32]
        + [lambda vhat32−qhat32]
        + [qhat32−FFT128(d32)].

Their FP64 inverse transforms estimate kappa. The first term combines inverse-transform/output rounding and FP64 reporting roundoff; the decomposition is not a vendor FFT backward-stability theorem. Recomputing G in complex128 is a separate diagnostic, not promotion of the same g. The denominator's assembly must be pinned; otherwise include an additional denominator-assembly term.

## Individual stage error and accumulated discrepancy

For either branch p, write computed `g=Gd+eG_p(d)`. Relative to F, the exact single-stage errors are

    epsilon_o = (I−RA)eG_o + R(−a+b) + rho(ho) + beta_o,
    epsilon_f = (I+RD)eG_f + Rf      + rho(hf) + beta_f,
    u_p(d) = Fd + epsilon_p(d).

The errors use that branch's actual d, g and h. Comparing already diverged stage inputs as if they were a common-input replay confounds a local formula change with propagated state differences.

Since `eG=G kappa`, `||G||_2=1/min_batch(gamma)` for this imaginary diagonal stream and positive spatially constant gamma (the zero mode attains it). For example,

    ||epsilon_o|| <= (1+||R|| ||A||)||G|| ||kappa||
                     + ||R||(||a||+||b||)+||rho(ho)||+||beta_o||,
    ||epsilon_f|| <= (1+||R|| ||D||)||G|| ||kappa||
                     + ||R|| ||f||+||rho(hf)||+||beta_f||.

Available exact conservative operator bounds are `||A||<=max|Omega dot k|+max(t)+max(s)||K||` and `||D||<=max|gamma−t|+max(s)||K||`, using the pinned stored symbols/constants. They explain why a small field error can have a large derivative image; they are generally too loose to predict the observed parity ratio. Norm estimates from un-enclosed eigenvalue/SVD computations remain empirical.

For the outer loop define

    d_k^p = q−A z_k^p + tau_k^p,
    z_(k+1)^p = z_k^p + omega u_p(d_k^p) + alpha_k^p.

Tau includes the actual outer A-action and subtraction rounding; alpha includes the update multiplication/addition rounding. At stage0 with zero state, the special code path uses d=q and tau=0. For omega=1 there is no relaxation multiplication error. With delta=z_f−z_o and E=I−omega FA,

    delta_(k+1) = E delta_k
                  + omega F(tau_k^f−tau_k^o)
                  + omega[epsilon_f(d_k^f)−epsilon_o(d_k^o)]
                  + alpha_k^f−alpha_k^o.

This is an exact algebraic recurrence for actual nonlinear FP32 kernels once the errors are defined as above. For delta_0=0, let v_k denote its three forcing terms. It gives the rigorous exact-arithmetic inequality

    ||delta_J|| <= sum_(k=0)^(J−1) ||E^(J−1−k)|| ||v_k||,
    ||H delta_J|| <= sum_(k=0)^(J−1) ||H E^(J−1−k)|| ||v_k||.

No contraction, independent noise, zero mean, or spectral-radius-only bound is assumed. The matrices can be nonnormal. The measured difference of separately evaluated H-images also includes their FP64 action-rounding difference. A diagnostic's finite set of measured norms is not a rigorous numerical upper bound on E powers.

For each nominal branch, measure tau from `d−[q−A128(z)]`, alpha from promoted pre/post states, and the complete epsilon from `u−F128(d)`, or a vector decomposition of its constituent kernels. This closes a stagewise vector ledger without assuming FP32 R is linear. The measured common-input ledger isolates local causation; the two nominal ledgers account for propagation. Cache bytes, actual operands, operation order and dtype policy belong in the receipt.

## Diagnostic priority and conditional repair

This section records the pre-repair conditional reasoning. The root execution updates below describe the subsequently closed ledger and unsuccessful selected repair.

The present evidence disfavors a cheap local-affine reordering as the first repair: local cancellation is modest and its transported rounding term is small. A G-only precision change is also not yet justified while actual reference/addition differences exceed the transported local difference by up to 75 times in the checked example. Removing a rounded identity correction can move either branch away from or toward a fixed ideal action; parity supplies no ranking of exact error.

First close the existing extended reference/addition/outer ledger. Split reference rounding only if needed into forward FFT, cached angular matrix multiplication and inverse FFT/output effects, using the same stored inverse. Diagnose output and accumulator quantization separately. A large H-image of `rho(hf)−rho(ho)` supports selective promotion of that reference application; a large alpha image supports a wider accumulator only if its cast-back image is also acceptable. Both have real cost. A large kappa with small remaining terms supports selective G precision, retaining the same assembled denominator and explicitly measuring the cast-back error. Promote only local products/sum if their term is dominant. None of these conditional interventions is a demonstrated repair yet; select one from the completed ledger and freeze it before the unchanged 48 checks.

For frequency attribution, retain within-grid N/4 and Nyquist masks but add common physical-mode bands, angular |Omega dot k|, and H delta = S_phys delta + C delta with its real cross term. Report absolute norms, raw numerator/denominator, and null denominators rather than epsilon-floor relative values when a defect is effectively zero. The exact amplification factorization is

    image_relative/field_relative
      = (||H delta||/||delta||) (||z_o||/||H z_o||),

provided these norms are nonzero. Diagnostic rounding and fresh FP32 subtraction must be excluded or reported separately.

## Gate and bounded-solve acceptance audit

`deliverables/precision/solve_protocol.md` preserves the proper boundary: all48 frozen cases must first pass both original field and original H-image gates, `1e-4` for complex64 and `1e-9` for complex128, with finite results, unchanged parents/probes/depths and intended work counts. Any precision widening/cache/cast cost is explicit. A missing or failed receipt prevents entry into the six-pair solve test; no post hoc threshold relaxation or replacement denominator is warranted.

Only after that entry pass, the proposed two-media/six-grid-case test separately requires every solve's independently recomputed original-RHS residual <=`1e-5`, no failed/nonfinite output, final-field difference <=`1e-4`, and the predeclared evaluated coercivity error bound <=`1e-3` for both arms. With weighted P and mu=lambda_max((P+P*)/2), m=min(t−s mu)>0, delta=||q−Hx||/m gives the exact discrete bound `||x−x_exact||/||x_exact|| <= delta/(||x||−delta)` when `||x||>delta`. Computed FP64 m/residual values without interval enclosure are empirical certificate diagnostics; they do not bound continuum/discretization error. Allowances in the two-solution consistency check must be recorded rather than silently enlarged.

Action parity, final-solution accuracy and warm/setup cost are separate results. Passing action parity does not prove global FGMRES convergence or closer-to-exact inversion. Passing all six solve pairs supplies development evidence on two reused media, not OOD or confirmatory sample evidence. Accuracy is mandatory before cost; the proposed >=5% warm saving at both N128/N256, <=10% individual regression and setup-plus-one-solve conditions do not authorize promotion, additional training or new test access. No further experiment was run by this reviewer.

## Root execution and frozen candidate update, 2026-10-02

The root executed the extended diagnosis and retrieved it under `receipts/precision/diagnosis_extended/`. I independently verified its complete five-case/80-stage/15-replay structure and both diagnosis script hashes against the saved receipts. The extended run reproduces the same five final image failures. Its reference/addition bookkeeping and FFT ledger have correct signs. The common-input constructed reference/addition closure is at reporting-roundoff scale; this is a telescoping accounting check, not an independent proof of FP32 linearity or an FFT error bound.

The stage0 empirical physical-image norms are:

| Case | R128(local delta) | Native R difference minus R128 replay | Single-addition error difference | Outer-addition error difference at stage15 |
|---|---:|---:|---:|---:|
| N64/ID0 | 0.00002056 | 0.00126525 | 0.00005039 | 0.00137887 |
| N128/ID0 | 0.00006537 | 0.00522610 | 0.00025626 | 0.00515192 |
| N128/ID3 | 0.00006585 | 0.00329588 | 0.00022757 | 0.00260974 |
| N256/ID0 | 0.00020181 | 0.01518161 | 0.00062074 | 0.02018251 |
| N256/ID3 | 0.00021214 | 0.01027565 | 0.00072446 | 0.01015248 |

These absolute norms have different stages/operands; they cannot be added to explain the final discrepancy. The native reference discrepancy is substantial, while outer accumulation also has a large measured image. The latter is the difference of update-rounding errors on two already diverged trajectories. The inspected extended/final scripts initially lacked the outer action/subtraction tau ledger and independent cache-equality receipt; the later closed diagnosis below resolves those two accounting gaps.

The root froze one `SumStableAlgebraicTSA` candidate and policy, code SHA256 `274c94dd0ca5fa6e36ffd51ab9b0036a2a7df32b40041d3e9200457378460b6b`. Source review confirms **both** stage `g+R(h)` and outer accumulation/output are complex128; G, local/scattering formula, native cached R and outer A retain the input dtype, with the accumulator cast before outer A. Relaxation multiplication also occurs in complex128 (omega=1 here). It adds no wider FFT or wider inverse cache and preserves the expected 94 spatial transforms at depth16. Wider sums, cast buffers and memory traffic still have costs.

This was a bounded feasibility test of a cheap precision intervention, rather than a demonstrated dominant-cause repair. Its new complex128 output is explicit and matches the deployed wrapper's output precision. The unchanged full48 checker preserves the frozen source/material fields, physical-image normalization and limits. The following final update supplies the candidate result; no candidate computation was run by this reviewer.

## Closed diagnosis and negative repair result, 2026-10-02

The final receipts are `receipts/precision/executed/diagnosis_closed/`, `repair_gate/` and `execution_receipt.json`. I independently checked the closed script's saved SHA256 `d8aae8caaeb581d2d43883cab5d2d63eff29485004505d7143e454886a527146`, repair checker/helper hashes, all three frozen-policy code hashes, policy-to-execution-receipt hash, local-to-executed source equality, and unchanged three development-parent hashes. These checks establish the inspected code/receipt identity; original-source post-execution preservation is a separate root-owned check.

The closed diagnosis is complete on five cases,80 stages and15 selected stage budgets. It asserts bit-identical independently constructed reference-cache tensors in every checked case. Delta spectra now promote both operands before subtraction. Its nominal outer tau, complete native stage error, propagation and accumulation vectors close the physical-image recurrence to maximum absolute `1.9683892896e-11` over the15 budgets. This closes the empirical accounting gaps; the fixed-linear-map recurrence above remains the exact-arithmetic theorem, while its FP64 replay implementation remains rounded.

The stage15 H-image norms show why addition-only attribution was insufficient:

| Case | F(delta tau) | Native stage-error difference | Propagated prior delta | Accumulation difference | Actual next delta |
|---|---:|---:|---:|---:|---:|
| N64/ID0 | 0.00494024 | 0.00001673 | 0.00080651 | 0.00137887 | 0.00518403 |
| N128/ID0 | 0.01828684 | 0.00003097 | 0.00121791 | 0.00515192 | 0.01903240 |
| N128/ID3 | 0.00946001 | 0.00000848 | 0.00062108 | 0.00260974 | 0.00984120 |
| N256/ID0 | 0.07724487 | 0.00006159 | 0.00187840 | 0.02018251 | 0.07989801 |
| N256/ID3 | 0.03845605 | 0.00002342 | 0.00102305 | 0.01015248 | 0.03978533 |

The outer residual-rounding difference after F is the largest term in all five late-stage budgets. Its magnitude is greater than the accumulation term even though both branches call the same outer A/subtraction expression on their different rounded states. This locates an important late discrepancy in the **state-dependent outer residual kernel**, rather than demonstrating that the eliminated inner streaming FFT or a particular outer FFT/subtraction operation alone caused it. Vector norms can cancel; neither the table nor parity ranks the two branches' absolute errors against an exact inverse. No further kernel-level causal conclusion is established.

The complete repair gate contains48 unique expected rows: three grids, two fixed media IDs, two depths, two input dtypes and two probes. I recomputed pass status from the unchanged `1e-4`/`1e-9` limits and independently checked every finite flag and FFT count. **43 pass; five fail**, all the original physical-source/depth16/complex64 rows:

| Case | Repair relative field difference | Repair original-H image difference |
|---|---:|---:|
| N64/ID0 | 6.73799322e-7 | 1.3806063286e-4 |
| N128/ID0 | 8.60362543e-7 | 2.5755760467e-4 |
| N128/ID3 | 2.19343611e-7 | 1.0127564783e-4 |
| N256/ID0 | 3.34066576e-7 | 5.1800296352e-4 |
| N256/ID3 | 1.91200450e-7 | 2.3087521684e-4 |

All field gates pass (complex64 maximum `5.0398679269e-6`). All24 complex128-input rows pass, with maximum field `9.3910707624e-15` and image `1.4294981205e-12`; these promote the original stored FP32 problem/cache rather than rebuilding them. All single-stage and grazing-probe rows pass. Every count is6→4 spatial transforms at depth1 and126→94 at depth16. These are work counts, not latency evidence.

The selected widening failed to recover the required image parity. It preserves native G/local/scattering/R/outer-A precision for complex64 input, so residual-kernel and reference-action errors remain; wider sums do not automatically remove their physical effects. It also incurs memory traffic:32 complex64→complex128 field casts and15 reverse outer-state casts per depth16 complex64 call, a conceptual887,095,296 read/write bytes (846 MiB) at N256, plus a12 MiB accumulator instead of6 MiB. These are the candidate's cast/storage ledger, not measured incremental bandwidth or peak usage. The recorded gate-process peak and setup timings are diagnostic receipts, not warm application/solver comparisons.

The execution receipt records expected gate exit2, `gate_passed=false`, full-solves0, warm-solver-timings0, trainingfalse and test untouched. The conditional six-pair solve protocol was therefore **not entered**. Stop this candidate under the original gate; do not loosen it, drop the five rows, or infer FGMRES failure, an unsafe returned solution, speed benefit, or impossibility of a different complex128 policy. The exact algebraic identity and concrete126→94 work reduction remain valid positives. Any future test of a different policy needs its own frozen mechanism and bounded protocol; these receipts supply no passing result for it.
