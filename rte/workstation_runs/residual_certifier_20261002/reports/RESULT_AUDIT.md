# Independent scalar certificate result audit

2026-10-02. The stdlib auditor independently reproduces every scalar acceptance predicate using exact Fraction arithmetic and agrees with the sealed receipts. It does not import or rerun RTE, GPU code, the native backend or omitted original arrays.

**Outcome:** 4 certified inputs, 4 rigorous original-source/zero residual rejections, 2 inconclusive NO_CERTIFICATE cases. Zero full solves and training steps. No full-solver comparison is admitted by this result.

| Case | Classification | Residual upper / Q_L | Radius upper / Q_L | Solution-error upper | Complete verification (s) |
|---|---|---:|---:|---:|---:|
| saved_teacher N64 ID0 | CERTIFIED_INPUT | 6.72091265e-09 | 6.45309087e-09 | 4.95720553e-08 | 1.235758 |
| saved_teacher N64 ID3 | CERTIFIED_INPUT | 3.09881355e-09 | 2.79699932e-09 | 2.56738122e-08 | 0.965056 |
| saved_teacher N128 ID0 | CERTIFIED_INPUT | 8.67321741e-08 | 8.6509423e-08 | 6.39997548e-07 | 4.069300 |
| original_source_zero N128 ID0 | RIGOROUS_RESIDUAL_REJECTION | 1 | 1.58839224e-16 | undefined | 6.786959 |
| saved_teacher N128 ID3 | CERTIFIED_INPUT | 3.77175981e-08 | 3.75022936e-08 | 3.1245409e-07 | 3.828175 |
| original_source_zero N128 ID3 | RIGOROUS_RESIDUAL_REJECTION | 1 | 1.6058921e-16 | undefined | 6.805835 |
| manufactured_M_source N256 ID0 | NO_CERTIFICATE | 1.1520961e-06 | 1.15209604e-06 | 8.56672448e-06 | 17.762027 |
| original_source_zero N256 ID0 | RIGOROUS_RESIDUAL_REJECTION | 1 | 1.58940264e-16 | undefined | 29.394562 |
| manufactured_M_source N256 ID3 | NO_CERTIFICATE | 5.03149137e-07 | 5.03149102e-07 | 4.18128542e-06 | 16.051979 |
| original_source_zero N256 ID3 | RIGOROUS_RESIDUAL_REJECTION | 1 | 1.60373217e-16 | undefined | 27.421937 |

The audit verifies 10 records (4 saved teachers, 2 manufactured fixtures, 4 negative controls), 10 explicit H actions, 2 M actions, the declared exact cases, zero prohibited work, matching policy/code/native-library seals, all 29 tiny FWI tests and read-only x86-MXCSR guards with flags 127. Same-context N128 teacher and zero controls share source/operator buffer identities. Shapes, dtypes, byte counts and no-mutation metadata are consistent. Those identities are evidence from receipts, not an independent reconstruction of the fields.

Acceptance is recomputed from rational coercivity m and binary endpoint values: `C_U <= .99*tau*Q_L`, `E_U <= .01*tau*Q_L`, `R_U <= tau*Q_L`, and positive-denominator `R_U/(m*X_L-R_U) <= 1e-3`, with exact `tau=1/100000`. The direct box norm and separately reported triangle bound remain distinct. Every outward scalar conversion is independently reproduced. All negative controls satisfy exact `R_L > tau*Q_U`, so their rejection is proved for the nominal stored operator. Norm-width inflation is already paid through the outward C_U and Q_L; norm width does not replace vector radius.

Across the six positive inputs, maxima are residual 1.1520961e-06, center 7.21586132e-10, radius 1.15209604e-06 and defined relative solution-error upper 8.56672448e-06.

Both N256 fixtures have valid residual upper bounds below tau (1.15209610e-6 and 5.03149137e-7) and valid relative solution-error upper bounds below 1e-3 (8.56672448e-6 and 4.18128542e-6), conditional on the CPU contract. The fixed vector-radius reserve alone fails: E_U/Q_L is 1.15209604e-6 and 5.03149102e-7 against 1e-7. Thus the enclosure establishes those weaker individual accuracy inequalities but does not admit either fixture under the complete frozen policy. This is enclosure uncertainty, not a lower-bound proof of GPU error or an inaccurate candidate.

Owned process wall is 122.262188 s; the driver records 118.335905 s after imports. Complete per-record verification totals 114.321586 s, including actual-D64 capture/CPU transfers, both identity-hash passes and the full certifier. Nested helper timers total 113.053165 s. Explicit rounded GPU H diagnostics total 0.229560 s separately. Cold compilation 0.419747 s and preflight 10.814560 s are outside the owned run and remain billable work. Peak GPU reserved memory is 513802240 B (allocated 268459520 B), within 4 GiB. Do not add nested helper/transfer categories to complete verification a second time. Parent loading, physical/capture setup, persistence and external startup also remain visible; this is not a solver speed measurement.

The N64/N128 inputs are actual stored full teachers and exact promoted original source bits. N256 inputs are new manufactured fixtures with represented q=captured rounded GPU H(x); no prior N256 benchmark field was recovered. The checker certifies each represented nominal residual directly and requires no uniform GPU action-error hypothesis. Its correctness remains conditional on the reviewed exact-bit I/O, integer/Fraction, native IEEE/compiler and nextafter contracts. Finite fixtures and register guards are supporting evidence, not a proof of every CPU operation.

The saved teacher q/x/t/s remain in their original PT banks. The manufactured N256 x/q and actual captured D64 were transient inputs hashed and certified within this owned run, without a proved new binary snapshot. Metadata saying buffers remain on FWI establishes nonexport, not persistence. Bit-for-bit replay by reexecution is unverified; the in-run certificate remains about its precise captured inputs.

The postrun receipt reports all 528 prior hashes preserved, tracked dirty work unchanged and no other research process. The auditor checks its seven retrieved scalar-payload hashes locally, including the stdout and empty stderr. It does not independently rerun the FWI seal. Preservation-before 18.593001 s and preservation-after 18.806471 s are separate experiment provenance work outside the owned driver; they are recorded without conflating them with an intrinsic checker latency.

Individual error bounds do not establish the separate 1e-4 paired-field criterion. No new Krylov convergence, performance advantage, continuum/physical-positivity claim, past unsaved-field accuracy, global GPU contract or reserved-test outcome follows. This audit does not independently repeat FWI preservation checks or rebuild the omitted action/coercivity arrays.

Reproduce: `python3 deliverables/certifier_20261002/audit_certifier_receipts.py --receipts receipts/certifier_execution`. Input SHA256 seals and every independently recomputed predicate are recorded in RESULT_AUDIT.json.
