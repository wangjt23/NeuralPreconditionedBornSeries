# RTE FWI: fresh original-H residual certifier

Verified 2026-10-02. **Four saved development teachers pass the unchanged residual, solution-error and fixed 1% reserve gates. Both N256 manufactured fixtures have valid residual/solution-error enclosures but fail the reserve, so receive NO_CERTIFICATE.** All four original-source zero candidates are rigorously rejected. No new full solve or training ran. This implements and validates a useful research checker; it does not admit a full complex-Krylov comparison or establish an affordable deployment checker.

## Workspace and access

Existing SSH alias FWI reached user `wjt` on `FWI-Standard-PC-i440FX-PIIX-1996`. The project is `/data/wjt/NeuralPreconditionedBornSeries`, subtree `rte/`, branch `main`, HEAD `028caea3f35d100225f656e2b1e910c1366aaafe`, origin `git@github.com:wangjt23/NeuralPreconditionedBornSeries.git`. RTE denotes the steady radiative transfer equation. The present model is periodic 2D on the unit square, twelve angles, g=.65 and epsilon=.1. The separately paused OL_preconditioner was untouched.

All target computation used FWI's existing Python3.12.4, Torch2.5.1/CUDA12.4, NumPy1.26.4 and RTX A6000(49,140MiB). The Mac served as authorized access bridge, auxiliary tiny CPU-test host, receipt auditor and artifact workspace. SSH disabled configured forwarding, retained existing credentials/host-key checking and used bounded connection attempts. No credential/trust/SSH/proxy/VPN/system/package change occurred. Prior scoped inspection found no applicable AGENTS.md or .agents/skills.

The new owned run is `rte/workstation_runs/residual_certifier_20261002`. All 528 preexisting preservation hashes and the original nine dirty tracked files were unchanged before and after execution. HEAD stayed fixed. No other user research process was active. Afterward the GPU returned to37MiB/0% utilization and no continuation job remained. Original code, fields, models and data were not exported or overwritten.

## Decision and mathematical target

Earlier finite-precision work could not establish uniform GPU FFT/contraction error constants. Fitting those constants to a finite set of probes would not justify a global contract. The selected direction evaluates **one represented candidate directly** with a mathematically specified independent interval transform. It bypasses both that unproved global GPU contract and the need to bound the preceding Arnoldi trajectory.

Fix the actual represented candidate x and source q, actual assembled original complex128 streaming symbol D64, and exact promoted stored material/angular buffers t,s,Phi,w. The nominal target is

    Hx = F^{-1} D64 F x + t*x - s*(Phi @ (w*x)).

F is the exact unnormalized spatial DFT on the last two axes, and its inverse divides by the spatial cell count. D64 already contains i. Its assembly rounding and Nyquist entries remain part of the target; no ideal geometry, FP32 symbol, real projection or rounded preassembled Phi*w matrix is substituted. This is the original stored periodic discretization, rather than a claim about continuum RTE, physical positivity or the rounded GPU map as a black box.

Every candidate/source/coefficient input is captured without bit loss and receives a dtype, shape, byte count and SHA256 identity. The field actually represented in complex128 is certified, including its output representation; no ideal unrounded iterate is assumed. Saved q originally complex64 is promoted exactly to complex128. Scalars from old benchmark receipts cannot recover their missing final fields and do not retrospectively certify them.

Let P=Phi*diag(w). Exact rational absolute row/column sums and a rational square-root upper enclosure give p_U>=sqrt(||P||1||P||infinity). Positive

    m = min_cells(t-s*p_U)

proves Re⟨v,Hv⟩ ≥ m||v||² because D64 is purely imaginary and Fourier streaming is skew-Hermitian, including at Nyquist. Thus H is invertible and ||H^{-1}||<=1/m. Rounded phase symmetry or exact quadrature normalization is unnecessary. The implementation derives m afresh from the captured exact coefficient bits; the exact rational m used in acceptance is included in the receipts, alongside its downward binary64 display value.

An interval rectangle R contains r=q-Hx. Outward norms provide R_L<=||r||<=R_U, Q_L<=||q|| and X_L<=||x||. Choose a represented center c inside R and bound E_U>=||r-c||. Acceptance uses exact rational cross-products with

    tau=1/100000, alpha=1/100, epsilon=1/1000:
    Q_L>0,
    ||c||_U <= (1-alpha)*tau*Q_L,
    E_U <= alpha*tau*Q_L,
    R_U <= tau*Q_L,
    m*X_L-R_U>0,
    R_U/(m*X_L-R_U) <= epsilon.

The direct residual-box upper R_U and triangle upper ||c||_U+E_U are separate recorded bounds. Norm rounding is paid through upper/lower endpoints and exact comparisons. E_U is a vector enclosure radius, not merely the width of a residual-norm interval. A failed upper test is an inconclusive certificate; R_L>tau*Q_U proves residual failure. Individual 1e-3 solution-error bounds do not establish the separate 1e-4 paired-field gate.

## Implementation, proof and trust boundary

New authored code only: `interval_fft.py`, `certified_twiddles.py`, `cpu_interval_primitives.cpp`, `certifier_driver.py` and three focused test files. It implements bit reversal and radix2 butterflies, retains both old butterfly operands, and encloses every complex product/addition, D64 multiplication, inverse scaling, weighted angular contraction, material product and final subtraction. No NumPy/PyTorch FFT, BLAS, ordinary norm reduction or trigonometric routine serves as a certificate oracle.

Twiddle roots are enclosed by exact rational Machin arctangent sums, reduced-angle rational sine/cosine alternating series and a proved Lipschitz allowance. Cardinal roots are exact. Binary64 endpoints are checked against their rational endpoints. Norms use outward primitive squares/pairwise sums and integer/rational square-root brackets. Underflow remains enclosed; nonfinite/overflow endpoints fail closed.

Endpoint add/multiply uses authored separate native functions built with existing g++13.3.0 and `-O2 -std=c++17 -fno-fast-math -ffp-contract=off -shared -fPIC`. Before/after guards require IEC559 binary64, nearest rounding and supported hardware registers with FTZ/DAZ disabled. Control modes are read, not changed; arithmetic can set sticky exception-status flags. Guard bits32/64 repeat register-mode conditions, while separate finite fixtures exercise subnormals and ties.

The proof is conditional on the recorded standard CPU primitive contract: correctly rounded binary64 operations and reliable adjacent-value stepping/exact-bit and integer/rational execution. Source review and finite tests are not formal verification of compiler or silicon. This explicit elementary boundary replaces the unknown GPU-wide FFT contract. It does not establish a global GPU constant.

The frozen policy SHA256 is `2e078e210664ed59e065b3815bb706e4fe933d72b901f36880fae53e7224841f`; it pins all seven executable/test sources before target validation. Interval module SHA256 `a150f0680fa98d7f627f9abbbfae1e45d62c5ad6f89b96a9197134743a88ea41`; native source `8a9a2eccfa3b85c4caa33958a1db0c67db2c97ecd340dd1c8f5294fa0608d53a`; FWI shared library `bf5f5d04c2821e508f83cdbd630758b1cc7b3c1ee4a1250d7ef42859a0f84786`.

Independent mathematical/source/protocol review supported only a bounded no-solve validation, conditional on FWI preflight. FWI compiled successfully and passed 29 tests plus 34 subtests, including an independent exact-Fraction dense DFT/action oracle, algebraic eighth roots, axis/ownership cases, subnormal/tie/cancellation behavior, extreme-scale norms, a nonnormal collision fixture and fail-closed negative cases. An independent stdlib scalar audit reproduced every acceptance boolean and outward scalar conversion from the raw receipts. It cannot reconstruct nominal actions or coefficient bits from scalar-only receipts; those depend on the reviewed code and FWI inputs.

## Actual bounded results

Ten certificates: four saved full teachers at N64/N128 for development IDs0/3; four original-source zero candidates at N128/N256; two N256 manufactured pairs x=M(source), q_fixture=the captured rounded GPUH(x). This is development seed4626, repeatedly reused historically. No training or heldout/reserved numerical evaluation occurred, and no new teacher was solved. Preservation hashing reread frozen existing files without evaluating their cases. The full saved banks are `source_tail_validation_20260926/validation.pt` and `tail_multiresolution_bank_20260926/validation.pt`; these are full q/target pairs, not tail/context banks.

|Input|Grid/ID|Residual / source norm|Radius / source norm|Relative solution-error upper|Complete verification|Outcome|
|---|---|---:|---:|---:|---:|---|
|Saved teacher|64/0|6.720913e-9|6.453091e-9|4.957206e-8|1.2358s|PASS|
|Saved teacher|64/3|3.098814e-9|2.796999e-9|2.567381e-8|0.9651s|PASS|
|Saved teacher|128/0|8.673217e-8|8.650942e-8|6.399975e-7|4.0693s|PASS|
|Saved teacher|128/3|3.771760e-8|3.750229e-8|3.124541e-7|3.8282s|PASS|
|Manufactured M-source|256/0|1.152096e-6|1.152096e-6|8.566724e-6|17.7620s|NO_CERTIFICATE: reserve|
|Manufactured M-source|256/3|5.031491e-7|5.031491e-7|4.181285e-6|16.0520s|NO_CERTIFICATE: reserve|

Ratios use certified source-norm lower bounds. The radius allowance is 1e-7 Q_L, using the certified source-norm lower bound. N128 teacher0 uses86.5094% of it; N256 fixtures exceed it11.5210× and5.03149×. Both N256 direct residual and solution-error upper bounds meet their accuracy thresholds, but the required reserve fails. Their center norms are approximately1.0004e-13 and5.1542e-14 relative to q; conservative rectangle growth dominates the enclosure. These are neither inaccurate-solution proofs nor certified original source-family solves.

All four zero candidates have a positive original source norm and prove R_L>tau*Q_U. They reject residual, center and solution-error acceptance as expected. Their complete verification costs are6.7870/6.8058s at N128 and29.3946/27.4219s at N256. The near-zero interval path creates subnormal endpoints and incurs substantial transform work; the source of its excess cost was not separately profiled, so no causal performance claim is made.

Saved teacher q/x/t/s remain in their original FWI banks. Newly manufactured N256 x/q and captured D64 were exact-bit transient inputs hashed and certified within the owned process, with no new persistent binary snapshot. “Remain on FWI” in the execution metadata means never exported, not proof that every transient buffer was saved. Exact replay of those fixtures is therefore an outstanding provenance gap; ordinary API reexecution is not an established bitwise replay. Historical benchmark final fields remain unavailable for retrospective certification.

## Research cost and preserved solver economics

The frozen driver allowed 120s soft after imports,150s hard for the owned subprocess and 4GiB GPU reserved memory. It completed in118.3359s internally and122.2622s including interpreter/import startup, with no timeout. Peak GPU allocated268,459,520B/reserved513,802,240B(490MiB); peak CPU RSS1,197,424KiB, approximately1.142GiB. It performed exactly 10 explicitH actions and 2 M actions; zero full solves/training steps. No unrelated job was cancelled.

Summed complete candidate verification was114.3216s, including actual D64 capture, CPU transfers, both bit-identity hash passes and the entire checker. The complete certifier-call timers totaled 113.1000s; summed internal helper elapsed time was 113.0532s. Both are nested within complete verification. Its transform/action dominates: N128 teacher3.5269/3.3041s, N256 fixture15.5565/14.0884s. Coefficient/norm/reserve/guard/threshold costs are recorded separately. Twiddle construction is included in first-use interval action time; it was not separately profiled. Input loading/hash0.9190s, physical setup0.3265s, base/capture setup0.1324s, M actions0.0359s, explicit diagnosticH actions0.2296s and JSON persistence0.9242s are separately visible; some helper/subprocess measurements are nested and must not be summed twice. Cold native compile0.4197s; FWI tiny-test subprocess10.8146s. The research preservation sweeps cost18.5930s before and 18.8065s after the driver; these are audit overhead, not per-candidate deployment work.

This checker is substantially more expensive than the established classical solver components. Prior unchanged direct segmented capture confirmed Q1 means223.3476→176.9263ms at N128 and162.1031→130.2582ms at N256, savings20.7844%/19.6448%. Those are previously measured per-material setup plus warm solve with ordinary independent acceptance, not process-cold or newly certified timings. Their absolute component savings remain evidence; adding a research checker does not discard them or certify their missing historical fields.

Any future comparison must charge every arm its complete required verification. An illustrative common additional cost C changes a component saving Delta from Delta/T_prior to Delta/(T_prior+C); it preserves the absolute saving but reduces its percentage. Current teacher/fixture timings do not establish identical costs on solver controls or integrated deployment economics. No checker caching, Q16 amortization, paid workload or learned superiority is claimed.

## Ranked next direction and remaining obstruction

1. **Best supported: stagewise center-plus-norm-radius certification.** The present rectangles grow past the fixed N256 reserve and cost seconds. For exact radix2 stages, ||B||=sqrt(2); enclosing each stage on its represented point center yields local error e and E_next<=sqrt(2)E+e. Exact forward/inverse scaling then gives E_stream<=E_inverse+(max|D64|E_forward+e_D)/N. This removes propagated rectangle wrapping without changing H,tau or the reserve. The theory is supplied, but that fallback is not implemented or executed in this screen. Decisive next evidence is a separately frozen implementation/independent proof, saved fixture snapshots, the same N256 reserve predicates and full charged cost. Optimizing certified constant-zero/cardinal paths can then be considered without altering the target.
2. **Only after a checker passes the required grids:** a freshly frozen complex-Krylov comparison with immutable final candidate/baseline fields, original-H fresh certificates, the separate paired-field gate and equally charged verification. The old failed low-H entry gate remains failed, and the previous uniform GPU error contract is still unproved. This round makes no exemption or full-solver claim.
3. **Learned training remains lower priority:** No affordable learned advantage against the stronger classical portfolio has been demonstrated. No new training/heldout campaign is justified by four teacher certificates or manufactured-input checks. Development selection, OOD/boundary/continuum accuracy and total historical acquisition/train/tuning costs remain gaps.

No further GPU campaign is queued. The precise current obstruction is N256 fixed-reserve failure and16–18s verification cost, together with missing old final fields and unpersisted new manufactured snapshots. Original residual tolerance,1% reserve and earlier negative results remain intact.

## Evidence and reproduction

The evidence bundle contains authored code/tests/theory/reviews, frozen policy, raw scalar/hash receipts and independent reducers. It excludes original project source/data/model/field bytes and native platform binaries. The original FWI repository and exact teacher parents are external dependencies for target reruns. Small CPU fixtures can run with an existing NumPy/system compiler; scalar audit/reduction uses only the standard library and requires neither CUDA nor original source/data.

The remote run's policy, preflight, execution and postrun preservation receipts retain code/data identities, compiler provenance, budgets and results. `RESULT_AUDIT.md/json` and `SCALAR_ANALYSIS.json` give independently reproducible scalar evidence. The handoff is saved on FWI and private Library artifact identities/exact-byte download verification are recorded separately, avoiding self-referential content hashes. No public sharing, Git commit/push/merge, new access, large training or holdout/source export occurred.

Primary proof/implementation references are the [GCC floating optimization contract](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html), [NumPy1.26 adjacent-float specification](https://numpy.org/doc/1.26/reference/generated/numpy.nextafter.html) and [MPFR directed-rounding semantics](https://www.mpfr.org/mpfr-current/mpfr.html#Rounding). The exact twiddle derivation and additional primary references appear in `TWIDDLE_PROOF_AND_SOURCES.md`; no empirical FFT constant is fitted.
